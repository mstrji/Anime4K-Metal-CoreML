import Foundation

// The translation approach (parameters injected into every user function and call, the texture
// macros mpv defines spelled as Metal `#define`s) follows Anime4KMetal by Yi Xie, Copyright 2021,
// licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
// This is a rewrite, not a copy: positions are texel centres as mpv defines them, `WIDTH`/`HEIGHT`
// are evaluated as the RPN expressions mpv reads them as, and the hooked texture is also reachable
// under its stage name, which `Anime4K_Clamp_Highlights` relies on.

/// One pass of an mpv user shader (`//!HOOK` block), translated to a Metal compute kernel.
///
/// Only what Anime4K's shaders use is supported: `HOOK`, `BIND`, `SAVE`, `WIDTH`, `HEIGHT`, `WHEN`,
/// `COMPONENTS` and `DESC`, the `NAME_pos/_size/_pt/_tex/_texOff` macros, and the GLSL spellings
/// `vec2/3/4`, `ivec2` and `mat4`. A directive it does not know fails the parse rather than being
/// skipped, so an unsupported shader is refused at preparation instead of drawing garbage.
struct MPVHookShader: Equatable, Sendable {
    let description: String
    /// The stage the pass runs at. `PREKERNEL` passes run after every `MAIN` pass.
    let hook: String
    let binds: [String]
    let save: String?
    /// RPN expressions, as mpv evaluates them. Nil means the hooked texture's size.
    let width: [String]?
    let height: [String]?
    let when: [String]?
    let code: [String]

    /// The texture a pass with no `SAVE` writes: it replaces the hooked one.
    var destination: String { save ?? "MAIN" }

    static let kernelName = "mpv_hook_pass"

    enum ParseError: Error, Equatable {
        case unknownDirective(String)
        case directiveOutsidePass(String)
        case malformed(String)
    }

    // MARK: - Parsing

    static func parse(_ text: String) throws -> [MPVHookShader] {
        var passes: [MPVHookShader] = []
        var current: Builder?

        for rawLine in text.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            if line.hasPrefix("//!") {
                let tokens = line.dropFirst(3).split(separator: " ").map(String.init)
                guard let directive = tokens.first else { throw ParseError.malformed(line) }
                let args = Array(tokens.dropFirst())
                if directive == "DESC" {
                    if let finished = current?.build() { passes.append(finished) }
                    current = Builder(description: args.joined(separator: " "))
                    continue
                }
                guard current != nil else { throw ParseError.directiveOutsidePass(line) }
                switch directive {
                case "HOOK":
                    guard let stage = args.first else { throw ParseError.malformed(line) }
                    current?.hook = stage
                case "BIND":
                    guard let name = args.first else { throw ParseError.malformed(line) }
                    current?.binds.append(name)
                case "SAVE":
                    guard let name = args.first else { throw ParseError.malformed(line) }
                    current?.save = name
                case "WIDTH":
                    guard !args.isEmpty else { throw ParseError.malformed(line) }
                    current?.width = args
                case "HEIGHT":
                    guard !args.isEmpty else { throw ParseError.malformed(line) }
                    current?.height = args
                case "WHEN":
                    guard !args.isEmpty else { throw ParseError.malformed(line) }
                    current?.when = args
                case "COMPONENTS":
                    // Every intermediate is RGBA half float; the component count changes nothing.
                    break
                default:
                    throw ParseError.unknownDirective(line)
                }
                continue
            }
            // Ordinary comments carry nothing the kernel needs (licence text above each file).
            if line.hasPrefix("//") { continue }
            guard current != nil else { continue }
            current?.code.append(line)
        }
        if let finished = current?.build() { passes.append(finished) }
        return passes
    }

    private struct Builder {
        var description: String
        var hook: String?
        var binds: [String] = []
        var save: String?
        var width: [String]?
        var height: [String]?
        var when: [String]?
        var code: [String] = []

        init(description: String) { self.description = description }

        func build() -> MPVHookShader? {
            guard let hook else { return nil }
            return MPVHookShader(description: description, hook: hook, binds: binds, save: save,
                                 width: width, height: height, when: when, code: code)
        }
    }

    // MARK: - Expressions

    /// Evaluates one of mpv's RPN size expressions (`conv2d_last_tf.w 2 *`,
    /// `OUTPUT.w MAIN.w / 1.200 > ...`). `sizes` maps a texture name to its width and height.
    static func evaluate(_ tokens: [String], sizes: [String: (Double, Double)]) -> Double? {
        var stack: [Double] = []
        for token in tokens {
            if let value = Double(token) {
                stack.append(value)
                continue
            }
            if let dot = token.lastIndex(of: "."), dot != token.startIndex {
                let name = String(token[..<dot])
                let axis = token[token.index(after: dot)...]
                if axis == "w" || axis == "h" {
                    guard let size = sizes[name] else { return nil }
                    stack.append(axis == "w" ? size.0 : size.1)
                    continue
                }
            }
            guard stack.count >= 2 else { return nil }
            let rhs = stack.removeLast()
            let lhs = stack.removeLast()
            switch token {
            case "+": stack.append(lhs + rhs)
            case "-": stack.append(lhs - rhs)
            case "*": stack.append(lhs * rhs)
            case "/": stack.append(lhs / rhs)
            case ">": stack.append(lhs > rhs ? 1 : 0)
            case "<": stack.append(lhs < rhs ? 1 : 0)
            case "=": stack.append(lhs == rhs ? 1 : 0)
            case "!": stack.append(lhs != rhs ? 1 : 0)
            default: return nil
            }
        }
        return stack.count == 1 ? stack[0] : nil
    }

    // MARK: - Metal translation

    /// The Metal parameter name of each bound texture, in binding order (texture index = position).
    var textureParameters: [String] { binds }

    var metalSource: String { metalSource(halfPrecision: true) }

    /// - Parameter halfPrecision: run the colour and feature maths (`vec4`, `mat4`) in half floats,
    ///   which Apple GPUs execute at twice the rate. Positions (`vec2`) stay 32-bit: a half cannot
    ///   address a 1920-wide texture to the texel. The intermediates are half-float textures either
    ///   way, as they are in mpv, so the network never had more precision than this to work with.
    func metalSource(halfPrecision: Bool) -> String {
        let scalar = halfPrecision ? "half" : "float"
        var source = """
        #include <metal_stdlib>
        using namespace metal;

        typedef float2 vec2;
        typedef \(scalar)3 vec3;
        typedef \(scalar)4 vec4;
        typedef int2 ivec2;
        typedef \(scalar)4x4 mat4;

        """

        // mpv's per-texture macros. The hooked texture is bound as HOOKED, and mpv also exposes it
        // under the stage's own name: the de-ring statistics pass binds HOOKED and reads MAIN_texOff.
        var macroTargets: [(macro: String, texture: String)] = binds.map { ($0, $0) }
        if binds.contains("HOOKED"), !binds.contains(hook) {
            macroTargets.append((hook, "HOOKED"))
        }
        for target in macroTargets {
            source += """
            #define \(target.macro)_pos mtlPos
            #define \(target.macro)_size float2(\(target.texture).get_width(), \(target.texture).get_height())
            #define \(target.macro)_pt (float2(1.0) / \(target.macro)_size)
            #define \(target.macro)_tex(pos) vec4(\(target.texture).sample(textureSampler, (pos)))
            #define \(target.macro)_texOff(off) \(target.macro)_tex(\(target.macro)_pos + \(target.macro)_pt * float2(off))

            """
        }

        let textureArgs = textureParameters.map { "texture2d<float, access::sample> \($0)" }
        let extraParams = (["float2 mtlPos", "sampler textureSampler"] + textureArgs).joined(separator: ", ")
        let extraArgs = (["mtlPos", "textureSampler"] + textureParameters).joined(separator: ", ")

        // Every user function gets the position, the sampler and the textures, since Metal has no
        // globals to reach them through. Definitions first, so call sites can be rewritten.
        let definition = try! NSRegularExpression(pattern: #"^(\w+)\s+(\w+)\s*\((.*)\)\s*\{$"#)
        var functionNames: [String] = []
        var body: [String] = []
        for line in code {
            let range = NSRange(line.startIndex..., in: line)
            if let match = definition.firstMatch(in: line, range: range),
               let returnRange = Range(match.range(at: 1), in: line),
               let nameRange = Range(match.range(at: 2), in: line),
               let argsRange = Range(match.range(at: 3), in: line),
               !["if", "for", "while", "switch", "return", "else"].contains(String(line[returnRange])) {
                let name = String(line[nameRange])
                let args = line[argsRange].trimmingCharacters(in: .whitespaces)
                functionNames.append(name)
                let params = args.isEmpty ? extraParams : extraParams + ", " + args
                body.append("static \(line[returnRange]) \(name)(\(params)) {")
                continue
            }
            var rewritten = line
            for name in functionNames {
                let call = try! NSRegularExpression(pattern: "\\b\(name)\\s*\\(")
                rewritten = call.stringByReplacingMatches(
                    in: rewritten, range: NSRange(rewritten.startIndex..., in: rewritten),
                    withTemplate: "\(name)(\(extraArgs), ")
                rewritten = rewritten.replacingOccurrences(of: "\(extraArgs), )", with: "\(extraArgs))")
            }
            body.append(rewritten)
        }
        source += body.joined(separator: "\n") + "\n\n"

        var entryParams = textureParameters.enumerated().map {
            "texture2d<float, access::sample> \($0.element) [[texture(\($0.offset))]]"
        }
        entryParams.append("texture2d<float, access::write> mpvOutput [[texture(\(textureParameters.count))]]")
        entryParams.append("sampler textureSampler [[sampler(0)]]")
        entryParams.append("uint2 gid [[thread_position_in_grid]]")
        source += """
        kernel void \(Self.kernelName)(\(entryParams.joined(separator: ",\n    "))) {
            uint2 outSize = uint2(mpvOutput.get_width(), mpvOutput.get_height());
            if (gid.x >= outSize.x || gid.y >= outSize.y) { return; }
            float2 mtlPos = (float2(gid) + 0.5) / float2(outSize);
            mpvOutput.write(float4(hook(\(extraArgs))), gid);
        }

        """
        return source
    }
}
