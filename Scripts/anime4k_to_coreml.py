#!/usr/bin/env python3
"""Build the Core ML models behind the AI upscaler's Neural Engine backend.

Reads the bundled Anime4K mpv shaders (bloc97/Anime4K, MIT) and rebuilds Mode A's two networks —
Restore_CNN_{S,M} then Upscale_CNN_x2_{S,M} — and C (S)'s single Upscale_Denoise_CNN_x2_S as one Core
ML program each, so they can run on the Neural Engine instead of the GPU. What stays on the GPU is the cheap part: YCbCr -> RGB before,
and the fused tail after (depth-to-space, de-ring clamp, 8-bit 4:2:0), which reads the model's output.

    input  "rgb"    [1, 3, H, W] float16  the decoded frame as RGB, 0...1
    output "planes" [1, 8, H, W] float16  0-2  the restored RGB (Restore's output, Upscale's MAIN;
                                               the input itself for C, which has no restore)
                                          3-6  conv2d_last_tf, the 2x2 sub-pixel residuals
                                          7    STATSMAX, the de-ring statistics of the input

Every pass of a network is `result = sum(mat4 * input) + bias`. An input is a texture itself (the first
pass reads MAIN's rgb, alpha column zero) or its CReLU halves max(t, 0) / max(-t, 0), read either at
3x3 offsets (`go_k(dx, dy)`) or at the texel itself (`g_k`, the (M) networks' last pass, which reads
every earlier layer). A mat4 is column-major, so weight[out i][in j] = floats[4 j + i]. The GPU samples
with clamp-to-edge, which is replicate padding before each 3x3 convolution. Restore's last pass adds
MAIN back; the de-ring statistics are a 5-wide horizontal then vertical max of the input's luma, i.e. a
5x5 max pool (a max over a clamped window equals the max over the in-bounds part, so the pool's own
padding is exact).

Usage (needs coremltools 9 and numpy, which have no wheels for Python 3.14; use 3.11):
    python3.11 -m venv env && env/bin/pip install coremltools numpy
    env/bin/python Scripts/anime4k_to_coreml.py --out <dir> [--check]
then compile what it writes:
    for m in A_S C_S; do xcrun coremlcompiler compile <dir>/Anime4K_$m.mlpackage \
        Sources/Anime4KUpscaler/Resources/Anime4K; done
"""

import argparse
import os
import re
import sys

import numpy as np

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Sources", "Anime4KUpscaler",
                          "Resources", "Anime4K")

# The source sizes the models are built for. Each is a separate shape the Neural Engine compiles when
# the model loads, so the list is short: the sizes A is too heavy for on the GPU. Any other size runs
# on the GPU. 1088 is H.264's coded height for 1080p; 1440x1080 is 4:3 at 1080p.
SIZES = [(1920, 1080), (1920, 1088), (1440, 1080), (1280, 720)]

# Model name -> (restore network or None, upscale network). C has no restore: its MAIN is the input.
NETWORKS = {
    "A_S": ("Anime4K_Restore_CNN_S.glsl", "Anime4K_Upscale_CNN_x2_S.glsl"),
    "C_S": (None, "Anime4K_Upscale_Denoise_CNN_x2_S.glsl"),
}


class Pass:
    def __init__(self, desc, binds, save, body):
        self.desc, self.binds, self.save = desc, binds, save
        self.inputs = {}    # macro name -> (texture, kind) with kind "raw", "pos" or "neg"
        self.terms = []     # (macro name, dx, dy, 4x4 weight [out][in])
        self.kernel = 3
        self.bias = None
        self.adds_main = False
        self._parse(body)

    def _parse(self, body):
        def kind_of(expr):
            if expr.startswith("max(-"):
                return "neg"
            return "pos" if expr.startswith("max(") else "raw"

        for m in re.finditer(r"^#define (go_\d+)\(x_off, y_off\) \((.*)\)\s*$", body, re.M):
            tex = re.search(r"(\w+)_texOff\(vec2\(x_off, y_off\)\)", m.group(2)).group(1)
            self.inputs[m.group(1)] = (tex, kind_of(m.group(2)))
        for m in re.finditer(r"^#define (g_\d+) \((.*)\)\s*$", body, re.M):
            tex = re.search(r"(\w+)_tex\(\1_pos\)", m.group(2)).group(1)
            self.inputs[m.group(1)] = (tex, kind_of(m.group(2)))
        for m in re.finditer(r"mat4\(([^)]*)\) \* (go_\d+)\((-?[\d.]+), (-?[\d.]+)\)", body):
            self.terms.append((m.group(2), int(float(m.group(3))), int(float(m.group(4))),
                               self._matrix(m.group(1))))
        for m in re.finditer(r"mat4\(([^)]*)\) \* (g_\d+);", body):
            self.terms.append((m.group(2), 0, 0, self._matrix(m.group(1))))
            self.kernel = 1
        if not self.terms:
            return  # Depth-to-space: no weights, run on the GPU.
        assert not (self.kernel == 1 and any(n.startswith("go_") for n, *_ in self.terms)), self.desc
        bias = re.search(r"result \+= vec4\(([^)]*)\);", body)
        self.bias = np.array([float(v) for v in bias.group(1).split(",")], dtype=np.float64)
        ret = re.search(r"return (.*);", body).group(1).strip()
        if ret == "result + MAIN_tex(MAIN_pos)":
            self.adds_main = True
        elif ret != "result":
            raise ValueError(f"{self.desc}: unexpected return {ret!r}")

    def _matrix(self, text):
        floats = [float(v) for v in text.split(",")]
        assert len(floats) == 16, self.desc
        return np.array(floats, dtype=np.float64).reshape(4, 4).T  # rows: out, columns: in

    def sources(self):
        """The textures read, in order of first use, each as "raw" or "crelu"."""
        order, modes = [], {}
        for name in sorted(self.inputs, key=lambda n: int(n.split("_")[1])):
            tex, kind = self.inputs[name]
            if tex not in modes:
                order.append(tex)
                modes[tex] = set()
            modes[tex].add(kind)
        result = []
        for tex in order:
            if modes[tex] == {"raw"}:
                result.append((tex, "raw"))
            else:
                assert modes[tex] == {"pos", "neg"}, (self.desc, tex, modes[tex])
                result.append((tex, "crelu"))
        return result


def parse_shader(path):
    text = open(path).read()
    passes = []
    for chunk in text.split("//!DESC ")[1:]:
        desc = chunk.splitlines()[0].strip()
        binds = re.findall(r"^//!BIND (\S+)", chunk, re.M)
        save = re.search(r"^//!SAVE (\S+)", chunk, re.M).group(1)
        passes.append(Pass(desc, binds, save, chunk))
    return passes


class Layer:
    """One pass as a convolution over its sources, with the CReLU folded:
    W1 max(t,0) + W2 max(-t,0) = (W1 + W2) max(t,0) - W2 t, so a CReLU source contributes
    [max(t,0), t] (8 channels) and needs one ReLU and no negation — about a quarter faster on the
    Neural Engine than the two halves (M1). A raw source (MAIN) contributes its 3 rgb channels."""

    def __init__(self, p, out_channels):
        self.desc, self.save, self.kernel, self.adds_main = p.desc, p.save, p.kernel, p.adds_main
        self.sources = p.sources()
        self.pass_ = p
        widths = [3 if mode == "raw" else 8 for _, mode in self.sources]
        starts = np.cumsum([0] + widths)
        k = self.kernel
        w = np.zeros((4, int(starts[-1]), k, k), dtype=np.float64)
        seen = set()
        for name, dx, dy, m in p.terms:
            tex, kind = p.inputs[name]
            assert (name, dx, dy) not in seen, p.desc
            seen.add((name, dx, dy))
            index = [s for s, _ in self.sources].index(tex)
            s0 = int(starts[index])
            ky, kx = (dy + 1, dx + 1) if k == 3 else (0, 0)
            if kind == "raw":
                assert np.all(m[:, 3] == 0), f"{p.desc}: alpha column is not zero"
                w[:, s0:s0 + 3, ky, kx] += m[:, :3]
            elif kind == "pos":
                w[:, s0:s0 + 4, ky, kx] += m
            else:  # neg: + m max(-t,0) = m max(t,0) - m t
                w[:, s0:s0 + 4, ky, kx] += m
                w[:, s0 + 4:s0 + 8, ky, kx] -= m
        assert len(seen) == k * k * len(p.inputs), f"{p.desc}: {len(seen)} taps"
        self.weight = w[:out_channels]
        self.bias = p.bias[:out_channels]


def network_layers(passes, last_channels):
    layers = [Layer(p, 4) for p in passes[:-1]] + [Layer(passes[-1], last_channels)]
    assert layers[0].sources == [("MAIN", "raw")], layers[0].desc
    return layers


def load_networks(name):
    restore_file, upscale_file = NETWORKS[name]
    upscale = parse_shader(os.path.join(SHADER_DIR, upscale_file))
    assert "Depth-to-Space" in upscale[-1].desc and upscale[-2].save == "conv2d_last_tf"
    upscale_layers = network_layers(upscale[:-1], 4)
    restore_layers = None
    if restore_file:
        restore = parse_shader(os.path.join(SHADER_DIR, restore_file))
        assert restore[-1].save == "MAIN" and restore[-1].adds_main
        restore_layers = network_layers(restore, 3)
        assert not any(l.adds_main for l in restore_layers[:-1])
    assert not any(l.adds_main for l in upscale_layers)
    return restore_layers, upscale_layers


def build(name, sizes):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    from coremltools.converters.mil.mil import get_new_symbol

    restore_layers, upscale_layers = load_networks(name)
    fp16 = np.float16

    def network(main, layers, prefix):
        tensors = {"MAIN": main}
        relus = {}

        def source(tex, mode):
            if mode == "raw":
                return [tensors[tex]]
            if tex not in relus:
                relus[tex] = mb.relu(x=tensors[tex], name=f"{prefix}_{tex}_relu")
            return [relus[tex], tensors[tex]]

        out = None
        for i, layer in enumerate(layers):
            values = [v for tex, mode in layer.sources for v in source(tex, mode)]
            x = values[0] if len(values) == 1 else mb.concat(values=values, axis=1, name=f"{prefix}{i}_in")
            if layer.kernel == 3:
                x = mb.pad(x=x, pad=np.array([1, 1, 1, 1], dtype=np.int32), mode="replicate",
                           name=f"{prefix}{i}_pad")
            out = mb.conv(x=x, weight=layer.weight.astype(fp16), bias=layer.bias.astype(fp16),
                          pad_type="valid", name=f"{prefix}{i}")
            tensors[layer.save] = out
        return out

    def carried_network(main, layers, prefix):
        """The (M) networks: a chain of 3x3 layers whose 1x1 last pass reads every one of them.
        Kept literally, every layer's output stays alive until the end, which the Neural Engine
        spills to memory: A (M) took 44 ms on an M1 against A (S)'s 11.5, and 39 ms with that pass
        split into one small convolution per layer. Instead (26 ms), each layer's
        share of the last pass is added into a 4-channel running sum carried through the next
        layer's convolution as four extra channels (centre tap only), so the graph stays a chain."""
        convs, dense = layers[:-1], layers[-1]
        n = len(convs)
        cout = dense.weight.shape[0]
        blocks = [np.zeros((4, 8)) for _ in range(n)]
        for j in range(n):
            blocks[j][:cout] = dense.weight[:, 8 * j:8 * j + 8, 0, 0]
        dense_bias = np.zeros(4)
        dense_bias[:cout] = dense.bias
        identity = np.eye(4)

        x = main
        for k, layer in enumerate(convs):
            if k == 0:
                inputs, weight, bias = x, layer.weight, layer.bias
            else:
                t = x if k == 1 else mb.slice_by_index(x=x, begin=[0, 0, 0, 0], end=[0, 4, 0, 0],
                                                       end_mask=[True, False, True, True], name=f"{prefix}{k}_t")
                inputs = mb.concat(values=[mb.relu(x=t, name=f"{prefix}{k}_relu"), x], axis=1, name=f"{prefix}{k}_in")
                cin = 8 if k == 1 else 12
                weight = np.zeros((8, cin, 3, 3))
                weight[:4, :8] = layer.weight
                weight[4:, :8, 1, 1] = blocks[k - 1]
                if k > 1:
                    weight[4:, 8:12, 1, 1] = identity
                bias = np.concatenate([layer.bias, dense_bias if k == 1 else np.zeros(4)])
            padded = mb.pad(x=inputs, pad=np.array([1, 1, 1, 1], dtype=np.int32), mode="replicate",
                            name=f"{prefix}{k}_pad")
            x = mb.conv(x=padded, weight=weight.astype(fp16), bias=bias.astype(fp16), pad_type="valid",
                        name=f"{prefix}{k}")
        t = mb.slice_by_index(x=x, begin=[0, 0, 0, 0], end=[0, 4, 0, 0], end_mask=[True, False, True, True],
                              name=f"{prefix}_last_t")
        inputs = mb.concat(values=[mb.relu(x=t, name=f"{prefix}_last_relu"), x], axis=1, name=f"{prefix}_last_in")
        weight = np.zeros((cout, 12, 1, 1))
        weight[:, :8, 0, 0] = blocks[n - 1][:cout]
        weight[:, 8:12, 0, 0] = identity[:cout]
        return mb.conv(x=inputs, weight=weight.astype(fp16), bias=np.zeros(cout).astype(fp16), pad_type="valid",
                       name=f"{prefix}_last")

    def is_carried(layers):
        convs, dense = layers[:-1], layers[-1]
        return (dense.kernel == 1 and all(l.kernel == 3 for l in convs) and len(convs) >= 2
                and dense.sources == [(l.save, "crelu") for l in convs]
                and all(convs[k].sources == [(convs[k - 1].save, "crelu")] for k in range(1, len(convs))))

    def any_network(main, layers, prefix):
        if is_carried(layers):
            return carried_network(main, layers, prefix)
        return network(main, layers, prefix)

    height, width = get_new_symbol(), get_new_symbol()

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, height, width), dtype=types.fp16)],
                opset_version=ct.target.iOS17)
    def program(rgb):
        if restore_layers:
            restored = mb.add(x=any_network(rgb, restore_layers, "restore"), y=rgb, name="restored")
        else:
            # No restore network: the upscale reads the frame itself, and planes 0-2 carry it through.
            restored = rgb
        residual = any_network(restored, upscale_layers, "upscale")
        luma_w = np.array([0.299, 0.587, 0.114], dtype=np.float64).reshape(1, 3, 1, 1)
        luma = mb.conv(x=rgb, weight=luma_w.astype(fp16), name="luma")
        stats = mb.max_pool(x=luma, kernel_sizes=[5, 5], strides=[1, 1], pad_type="same", name="statsmax")
        return mb.concat(values=[restored, residual, stats], axis=1, name="planes")

    shapes = [(1, 3, h, w) for (w, h) in sizes]
    model = ct.convert(
        program, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        inputs=[ct.TensorType(name="rgb", shape=ct.EnumeratedShapes(shapes=shapes, default=shapes[0]),
                              dtype=np.float16)],
        outputs=[ct.TensorType(name="planes", dtype=np.float16)])
    files = ", ".join(f[:-5] for f in NETWORKS[name] if f)
    model.short_description = (f"Anime4K {name} networks ({files}) and de-ring statistics, "
                               "from bloc97/Anime4K (MIT).")
    return model, restore_layers, upscale_layers


def reference(rgb, restore_layers, upscale_layers):
    """The shaders evaluated literally (CReLU halves, not folded) in float64 numpy. rgb: [3, H, W]."""
    def shifted(t, dx, dy):
        c, h, w = t.shape
        padded = np.pad(t, ((0, 0), (1, 1), (1, 1)), mode="edge")
        return padded[:, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    def run(main, layers):
        tensors = {"MAIN": main}
        out = None
        for layer in layers:
            p = layer.pass_
            n = layer.weight.shape[0]
            h, w = main.shape[1:]
            out = np.zeros((4, h, w))
            for name, dx, dy, m in p.terms:
                tex, kind = p.inputs[name]
                t = tensors[tex]
                if tex == "MAIN":
                    t = np.concatenate([t, np.zeros((1, h, w))])
                v = shifted(t, dx, dy)
                v = v if kind == "raw" else np.maximum(v, 0) if kind == "pos" else np.maximum(-v, 0)
                out += np.einsum("oi,ihw->ohw", m, v)
            out = out[:n] + p.bias[:n, None, None]
            tensors[layer.save] = out
        return out

    restored = run(rgb, restore_layers) + rgb if restore_layers else rgb
    residual = run(restored, upscale_layers)
    luma = np.tensordot([0.299, 0.587, 0.114], rgb, axes=1)
    padded = np.pad(luma, 2, mode="edge")
    h, w = luma.shape
    stats = np.max([padded[dy:dy + h, dx:dx + w] for dy in range(5) for dx in range(5)], axis=0)
    return np.concatenate([restored, residual, stats[None]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory for Anime4K_<name>.mlpackage")
    parser.add_argument("--networks", default=",".join(NETWORKS), help="comma-separated: " + ", ".join(NETWORKS))
    parser.add_argument("--check", action="store_true", help="compare each model with numpy (macOS only)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for name in args.networks.split(","):
        model, restore_layers, upscale_layers = build(name, SIZES)
        path = os.path.join(args.out, f"Anime4K_{name}.mlpackage")
        model.save(path)
        print("wrote", path)
        if not args.check:
            continue
        import coremltools as ct
        rng = np.random.default_rng(1)
        h, w = 720, 1280
        # Smooth shapes and hard edges, like a cel: blocks of flat colour plus a gradient.
        rgb = np.zeros((3, h, w))
        yy, xx = np.mgrid[0:h, 0:w]
        rgb += (xx / w)[None]
        for _ in range(40):
            y0, x0 = rng.integers(0, h - 40), rng.integers(0, w - 40)
            rgb[:, y0:y0 + rng.integers(10, 120), x0:x0 + rng.integers(10, 120)] = rng.random((3, 1, 1))
        rgb = np.clip(rgb, 0, 1)
        expected = reference(rgb, restore_layers, upscale_layers)
        for units in [ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_NE]:
            loaded = ct.models.MLModel(path, compute_units=units)
            got = loaded.predict({"rgb": rgb[None].astype(np.float16)})["planes"][0].astype(np.float64)
            diff = np.abs(got - expected)
            print(name, units, "max abs diff per channel:", np.round(diff.reshape(8, -1).max(axis=1), 4),
                  "mean", float(diff.mean()))


if __name__ == "__main__":
    sys.exit(main())
