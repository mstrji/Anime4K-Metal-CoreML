# Anime4K-Metal-CoreML

Real-time [Anime4K](https://github.com/bloc97/Anime4K) for Apple video players: a Swift package that
takes decoded frames (`CVPixelBuffer`) and returns enhanced or 4K-upscaled ones. The heavy networks
run on the **Neural Engine** through Core ML, so an Apple TV 4K (3rd gen, A15) can upscale 1080p
anime to 4K for a whole episode without its GPU overheating.

## Modes

| Mode | What it does | Output | Runs on |
|---|---|---|---|
| `.enhance` | Anime4K **Restore (S)**: crisper lines, less blur and compression damage | same size | GPU (Metal) |
| `.upscale` | Anime4K **C (S)**: 2× upscale with light denoising | 2× (1080p → 4K) | Neural Engine |
| `.upscaleEnhance` | Anime4K **A (S)**: Restore, then 2× upscale | 2× (1080p → 4K) | Neural Engine |
| `.upscaleEnhanceHigh` | Anime4K **A (M)**: the larger Restore and 2× upscale networks | 2× (1080p → 4K) | Neural Engine + GPU, split |

Each mode watches its own speed. If the device cannot keep up with the video's frame rate,
`.upscaleEnhanceHigh` steps back to `.upscaleEnhance`, `.upscaleEnhance` steps back to `.enhance`, and
the others switch themselves off. Playback never
stops: a frame that is not processed is shown as it was decoded.

Input: SDR 8- or 10-bit 4:2:0 frames (NV12 / P010, video or full range) up to 1920×1088. HDR, Dolby
Vision, 4:2:2/4:4:4 and anything larger pass through untouched. Output is always 8-bit 4:2:0 in the
source's colour range and matrix.

## Measured on an Apple TV 4K (3rd gen, A15)

Real playback of two episodes, one frame sampled per second, 24 frames per row. Times are per 1080p
frame; **Total** is the whole path (colour conversion, networks, conversion back to 4:2:0) and is
what has to fit the frame budget: 39.6 ms at 23.976 fps.

| Source | Mode | Runs on | Total p50 / p95 | GPU p50 / p95 | Neural Engine p50 | Over 16.7 ms (/24) | Ink-line gain vs 2× bilinear |
|---|---|---|---|---|---|---|---|
| Mission: Yozakura Family S02E01, 8-bit H.264 | `.enhance` | GPU | 19.5 / 23.3 ms | 18.8 / 22.7 ms | — | 22 | +30.0% |
| | `.upscale` | Neural Engine | 14.8 / 16.8 ms | 4.6 / 4.7 ms | 9.3 ms | 3 | +9.5% |
| | `.upscaleEnhance` | Neural Engine | 19.6 / 25.5 ms | 4.6 / 6.9 ms | 14.0 ms | 24 | +52.5% |
| Re:ZERO S01E01, 10-bit HEVC | `.enhance` | GPU | 19.9 / 25.0 ms | 19.0 / 24.3 ms | — | 14 | −9.8% |
| | `.upscale` | Neural Engine | 14.4 / 18.3 ms | 4.6 / 7.4 ms | 8.6 ms | 3 | +8.6% |
| | `.upscaleEnhance` | Neural Engine | 20.9 / 24.5 ms | 4.7 / 4.9 ms | 14.6 ms | 24 | +64.4% |

- Every row fits 24 fps comfortably. "Over 16.7 ms" only matters for 60 fps sources, where `.upscale`
  is the one mode that nearly keeps up.
- **Ink-line gain** is the 99th percentile of the 4K luma gradient (central differences, 8-bit code
  values) compared with a plain 2× bilinear upscale of the same frames. It counts grain as edges too,
  so a mode that cleans a grainy source (`.enhance` on Re:ZERO) can score below bilinear while
  looking cleaner. The gain is repeatable to the decimal on the same frames; timings vary by 1–3 ms.
- On the GPU alone, A (S) took ~27 ms on this Apple TV until the GPU throttled after 15–20 s of that
  load (45–60 ms). Moving the networks to the Neural Engine is what made `.upscaleEnhance` sustainable.

### `.upscaleEnhanceHigh` (A (M)) on the same Apple TV

A (M) is about three times A (S)'s work. Whole on the Neural Engine it takes ~45–50 ms per 1080p
frame, over the 39.6 ms budget, so on a 1920×1080 source the frame is **split**: the Neural Engine
runs the model on the top rows while the GPU runs the same networks as Metal shaders on the bottom
rows, at the same time, and the two halves are joined. The split stays at row 720 (720 rows on the
Neural Engine, 360 on the GPU), where this Apple TV settled every time; on hardware with a relatively
stronger GPU set `rebalancesHighSplit = true` and every 2 s it moves to the row where both halves
would finish together (a Mac settles at 540). With **repeated-frame reuse** (on
by default, below) only about half the frames are upscaled at all.

Measured in a player built on this code (5 minutes of A Tale of the Secret Saint, 1080p H.264, 23.976
fps, logged every 2 s):

| | |
|---|---|
| Split row | 720 of 1080 (Neural Engine 720 rows, GPU 360) |
| Neural Engine / GPU half | ~25–31 / ~24–35 ms |
| Frame | ~27–38 ms typical, occasional spikes to 40–45 ms |
| Frames reused as repeats | 50–64% |
| Dropped frames | 0 (the player keeps ~0.3 s of upscaled frames queued) |

- Spikes come from other GPU and memory traffic: drawing the screen, and a player downloading the
  whole file in the background (~40 ms frames while a 1.4 GB file downloaded at full speed).
- The Neural Engine model was built for these band sizes too (1920×564, 624, 684 and 744). Other
  sizes run A (M) whole on the Neural Engine. Set `splitsHigh = false` to run 1080p whole as well.
- The first load of `Anime4K_A_M` on a device compiles it for the Neural Engine, ~15 s; later loads
  take a fraction of a second. Until it is ready the mode runs on the GPU alone (~64 ms a frame on this
  box), so call `setMode(.upscaleEnhanceHigh)` early, e.g. at launch, to get it ready in advance.
- Split against whole on the same frame: luma at most 2 steps apart (mean 0.02), chroma 1 (mean 0.01).

## Install

Swift Package Manager:

```swift
.package(url: "https://github.com/mstrji/Anime4K-Metal-CoreML", branch: "main")
```

Requires iOS 17, tvOS 17 or macOS 14. Without a Neural Engine (the Simulator, Intel Macs) the
upscaling modes run on the GPU instead.

## Use

```swift
import Anime4KUpscaler

let upscaler = Anime4KUpscaler()
Anime4KLog.handler = { print($0) }        // optional: preparation, slow frames, step-downs

// Per video:
upscaler.setMode(.upscaleEnhance)          // compiles shaders / loads models in the background
upscaler.setFrameRate(23.976)              // the watchdog judges against this
upscaler.beginSession()

// Per decoded frame, on your own serial queue (never the decoder's callback):
let upscaleQueue = DispatchQueue(label: "upscale", qos: .userInteractive)
func didDecode(_ frame: CVPixelBuffer, pts: CMTime) {
    upscaleQueue.async {
        let shown = upscaler.process(frame) ?? frame   // nil: show the original
        enqueueForDisplay(shown, pts)                    // e.g. AVSampleBufferDisplayLayer
    }
}

// Optional settings, shown at their defaults:
upscaler.reusesRepeatedFrames = true      // return a repeated drawing's last upscaled frame
upscaler.splitsHigh = true                // .upscaleEnhanceHigh: split 1080p across Neural Engine and GPU
upscaler.rebalancesHighSplit = false      // off by default: keep the split at row 720

// Diagnostics, e.g. in a stats overlay:
let s = upscaler.status   // state, preset, backend ("Neural Engine"/"GPU"), median times, budget, memory
```

### Integration notes (learned on a real Apple TV)

- **Never call `process(_:)` on the decoder's callback.** It waits for the GPU and the Neural
  Engine; on a thread that also feeds audio, that starved the audio renderer.
- **Hold at most `Anime4KUpscaler.maximumFramesInFlight` (16) returned buffers.** Each 4K 8-bit
  frame is ~12 MB. At the cap, `process` waits up to 250 ms for your display to release one, then
  returns nil.
- **Repeated frames.** Anime is mostly drawn on twos and threes. A frame whose every 8×8 block of
  luma and chroma is within 3 code values of the frame last upscaled (compression noise moves less,
  any real motion more) is returned as **that same output buffer**, so `process` can hand back an
  object it returned before. Wrap each one in its own sample buffer with the new frame's timestamp,
  and never write into a returned buffer. Reused frames cost one small GPU comparison and are not
  counted by the watchdog. On a 1080p episode 50–64% of frames were reused, which cuts the work, and
  the heat, roughly in proportion.
- **Memory.** It checks `os_proc_available_memory` and switches itself off rather than letting tvOS
  kill the app.
- **Size.** A returned buffer carries the attachment `Anime4KUpscaler.upscaleFactorAttachmentKey`
  (1 or 2), so your player can keep reporting the source's own resolution.
- **Model sizes.** The Neural Engine models are built for 1920×1080, 1920×1088, 1440×1080 and
  1280×720 (A (M) also for its split bands). Other sizes run the same networks on the GPU.
- **Audio in a sample-buffer player.** If your player renders audio with
  `AVSampleBufferAudioRenderer`, observe
  `AVSampleBufferAudioRendererOutputConfigurationDidChangeNotification` and flush/re-feed from the
  playhead when it fires. It is not about the upscaler itself, but players that switch the TV's
  frame rate or audio format as playback starts can otherwise play static until the next seek.

## How it works

- `MPVHookShader` parses Anime4K's mpv GLSL shaders and translates each pass to a Metal compute
  kernel (half precision for the networks). `.enhance` runs these on the GPU.
- `Scripts/anime4k_to_coreml.py` reads the same GLSL files and rebuilds the networks as Core ML
  programs (`Anime4K_A_S.mlmodelc`, `Anime4K_A_M.mlmodelc`, `Anime4K_C_S.mlmodelc`): 3×3 convolutions with replicate padding
  (the GPU samples clamp-to-edge), the CReLU folded as `(W1+W2)·max(t,0) − W2·t` (a quarter faster on
  the Neural Engine), Restore's residual add, and the de-ring statistics as a 5×5 max pool. Output:
  restored RGB, the four sub-pixel residuals and the statistics, as eight planes.
- Per frame on the Neural Engine path: YCbCr → planar RGB (GPU) → Core ML prediction (Neural Engine)
  → one fused GPU kernel doing depth-to-space, Anime4K's de-ring clamp and the conversion back to
  8-bit 4:2:0. Input and output tensors are IOSurface-backed pixel buffers shared between Metal and
  Core ML, so nothing is copied.
- **A (M)'s split** (`HighSplit.swift`): the GPU converts only the model's band to planar RGB, the
  Core ML prediction runs on that band (a strided view of the frame's tensors) while the GPU runs the
  shaders on the rest with a 24-row overlap (the networks' 14-pixel reach plus the 5×5 statistics) and
  then the fused tail on its own rows. After both finish, the GPU's rows just below the split are
  written into the model's planes and the planar tail runs on the rows above.
- **Same picture either way:** against the GPU path on the same frame, the Neural Engine output
  differs by at most one 8-bit step (mean ~0.04). The tests check this.

## Rebuilding the models

```bash
python3.11 -m venv env && env/bin/pip install coremltools numpy
env/bin/python Scripts/anime4k_to_coreml.py --out build --check
for m in A_S A_M C_S; do xcrun coremlcompiler compile build/Anime4K_$m.mlpackage Sources/Anime4KUpscaler/Resources/Anime4K; done
```

`--check` compares each model with an exact float64 evaluation of the shaders. The script reproduces
the bundled models' networks and weights byte for byte (only the conversion date in the metadata
differs).

## Tests

```bash
swift test
```

Shader parsing and compilation, the watchdog, the frame cap, repeated-frame reuse, and, on a Mac with
a Neural Engine, picture equality between the Neural Engine and GPU paths and between A (M) split and
whole.
`ANIME4K_UPSCALER_BENCH=1 swift test --filter Anime4KUpscalerBenchmark` prints timings.

## Credits and license

- **Anime4K** by bloc97, MIT: the networks, their weights and the shaders in
  `Sources/Anime4KUpscaler/Resources/Anime4K` (licence alongside them). The Core ML models are
  converted from those weights.
- **Anime4KMetal** by Yi Xie, Apache-2.0: the approach of translating mpv hook shaders to Metal.
  `MPVHookShader.swift` is a rewrite, not a copy.
- This package: MIT, see [LICENSE](LICENSE). Third-party notices in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
