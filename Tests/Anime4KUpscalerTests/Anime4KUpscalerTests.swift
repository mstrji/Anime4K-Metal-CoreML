import CoreVideo
import Foundation
import Metal
import Testing
@testable import Anime4KUpscaler

/// The Anime4K upscaler. The translation is checked against every bundled shader, and the
/// whole frame path runs on this machine's GPU. Speed here says nothing about an Apple TV.
@Suite("Anime4KUpscaler")
struct Anime4KUpscalerTests {

    static let shaderFiles = [
        "Anime4K_Clamp_Highlights", "Anime4K_Restore_CNN_S", "Anime4K_Upscale_CNN_x2_S",
        "Anime4K_Upscale_Denoise_CNN_x2_S",
    ]

    static func shaders(_ file: String) throws -> [MPVHookShader] {
        let url = try #require(Anime4KUpscaler.shaderURL(file), "\(file) is not bundled")
        return try MPVHookShader.parse(String(contentsOf: url, encoding: .utf8))
    }

    @Test("Parses each pass's directives")
    func parsesDirectives() throws {
        let clamp = try Self.shaders("Anime4K_Clamp_Highlights")
        #expect(clamp.map(\.hook) == ["MAIN", "MAIN", "PREKERNEL"])
        #expect(clamp.map(\.destination) == ["STATSMAX", "STATSMAX", "MAIN"])
        #expect(clamp[1].binds == ["HOOKED", "STATSMAX"])

        let upscale = try Self.shaders("Anime4K_Upscale_CNN_x2_S")
        #expect(upscale.count == 5)
        #expect(upscale.last?.width == ["conv2d_last_tf.w", "2", "*"])
        #expect(upscale.allSatisfy { $0.when != nil })
    }

    @Test("Evaluates mpv's RPN size and condition expressions")
    func evaluatesExpressions() {
        let sizes: [String: (Double, Double)] = ["MAIN": (1920, 1080), "OUTPUT": (3840, 2160),
                                                 "conv2d_last_tf": (1920, 1080)]
        #expect(MPVHookShader.evaluate(["conv2d_last_tf.w", "2", "*"], sizes: sizes) == 3840)
        let when = "OUTPUT.w MAIN.w / 1.200 > OUTPUT.h MAIN.h / 1.200 > *".split(separator: " ").map(String.init)
        #expect(MPVHookShader.evaluate(when, sizes: sizes) == 1)
        let native = ["MAIN": (3840.0, 2160.0), "OUTPUT": (3840.0, 2160.0)]
        #expect(MPVHookShader.evaluate(when, sizes: native) == 0)
        #expect(MPVHookShader.evaluate(["NOPE.w"], sizes: sizes) == nil)
    }

    @Test("Refuses a directive it does not understand")
    func refusesUnknownDirective() {
        #expect(throws: MPVHookShader.ParseError.self) {
            try MPVHookShader.parse("//!DESC x\n//!HOOK MAIN\n//!OFFSET 1 1\nvec4 hook() { return vec4(0); }")
        }
    }

    @Test("Every bundled pass compiles for Metal")
    func compiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        for file in Self.shaderFiles {
            for shader in try Self.shaders(file) {
                let library = try device.makeLibrary(source: shader.metalSource(halfPrecision: false), options: nil)
                #expect(library.makeFunction(name: MPVHookShader.kernelName) != nil, "\(file): \(shader.description)")
                // Every network pass must take the fast path; only the de-ring clamp may fall back.
                if file.contains("CNN") {
                    let half = try device.makeLibrary(source: shader.metalSource(halfPrecision: true), options: nil)
                    #expect(half.makeFunction(name: MPVHookShader.kernelName) != nil, "\(file) half: \(shader.description)")
                }
            }
        }
    }

    @Test("Each mode runs its own presets, in step-down order")
    func modePresets() {
        #expect(Anime4KUpscaler.presets(for: .off).isEmpty)
        #expect(Anime4KUpscaler.presets(for: .enhance).map(\.name) == ["Anime4K Restore (S)"])
        #expect(Anime4KUpscaler.presets(for: .upscale).map(\.name) == ["Anime4K C (S)"])
        #expect(Anime4KUpscaler.presets(for: .upscaleEnhance).map(\.name)
                == ["Anime4K A (S)", "Anime4K Restore (S)"])
        #expect(Anime4KUpscaler.presets(for: .upscaleEnhance).map(\.neuralEngineModel) == ["Anime4K_A_S", nil])
        #expect(Anime4KUpscaler.presets(for: .upscale).map(\.neuralEngineModel) == ["Anime4K_C_S"])
    }

    /// Waits for the shaders and, when the mode wants it, the Neural Engine model.
    static func ready(_ upscaler: Anime4KUpscaler) async throws {
        let deadline = Date().addingTimeInterval(120)
        while Date() < deadline {
            let status = upscaler.status
            if status.state != .preparing, status.neuralEngineNote != Anime4KUpscaler.loadingNote { return }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    @Test("The Neural Engine draws the same picture as the GPU",
          arguments: [(Anime4KMode.upscaleEnhance, 1280, 720), (.upscaleEnhance, 1920, 1080),
                      (.upscale, 1920, 1080)])
    func neuralEngineMatchesGPU(mode: Anime4KMode, width: Int, height: Int) async throws {
        guard MTLCreateSystemDefaultDevice() != nil, NeuralEngineUpscaler.deviceHasNeuralEngine else { return }
        let size = (width, height)
        let source = try Self.busyFrame(width: size.0, height: size.1)
        var planes: [Bool: (luma: [UInt8], chroma: [UInt8])] = [:]
        for neural in [true, false] {
            let upscaler = Anime4KUpscaler()
            upscaler.usesNeuralEngine = neural
            upscaler.setMode(mode)
            try await Self.ready(upscaler)
            #expect(upscaler.status.neuralEngineNote == nil)
            let output = try #require(upscaler.process(source))
            #expect(CVPixelBufferGetWidth(output) == size.0 * 2 && CVPixelBufferGetHeight(output) == size.1 * 2)
            #expect(upscaler.status.backend == (neural ? "Neural Engine" : "GPU"))
            planes[neural] = (Self.bytes(output, plane: 0), Self.bytes(output, plane: 1))
        }
        let neural = try #require(planes[true]), gpu = try #require(planes[false])
        for (name, a, b) in [("luma", neural.luma, gpu.luma), ("chroma", neural.chroma, gpu.chroma)] {
            var maxDiff = 0, total = 0, over1 = 0
            for i in a.indices {
                let d = abs(Int(a[i]) - Int(b[i]))
                maxDiff = max(maxDiff, d)
                total += d
                if d > 1 { over1 += 1 }
            }
            let mean = Double(total) / Double(a.count)
            print("[Anime4KUpscalerTests] \(mode.rawValue) Neural Engine vs GPU \(size.0)x\(size.1) \(name): max \(maxDiff), "
                  + "mean \(String(format: "%.4f", mean)) codes, \(over1) samples over 1")
            // Both compute in half precision, in a different order.
            #expect(maxDiff <= 2, "\(name) max \(maxDiff)")
            #expect(mean < 0.1, "\(name) mean \(mean)")
        }
    }

    @Test("A size the model was not built for runs on the GPU")
    func neuralEngineFallsBackForOtherSizes() async throws {
        guard MTLCreateSystemDefaultDevice() != nil, NeuralEngineUpscaler.deviceHasNeuralEngine else { return }
        let upscaler = Anime4KUpscaler()
        upscaler.setMode(.upscaleEnhance)
        try await Self.ready(upscaler)
        _ = try #require(upscaler.process(try Self.busyFrame(width: 960, height: 540)))
        #expect(upscaler.status.backend == "GPU")
        _ = try #require(upscaler.process(try Self.busyFrame(width: 1920, height: 1080)))
        #expect(upscaler.status.backend == "Neural Engine")
    }

    @Test("Restore only keeps the source size and the picture")
    func restoreOnlyKeepsSize() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        let upscaler = Anime4KUpscaler()
        upscaler.setMode(.enhance)
        let deadline = Date().addingTimeInterval(60)
        while upscaler.status.state == .preparing, Date() < deadline {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        let source = try Self.testFrame(pixelFormat: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, width: 1920, height: 1080)
        var timings: [Double] = []
        var output: CVPixelBuffer?
        for _ in 0..<5 {
            let started = CFAbsoluteTimeGetCurrent()
            output = upscaler.process(source)
            timings.append((CFAbsoluteTimeGetCurrent() - started) * 1000)
        }
        let result = try #require(output)
        #expect(CVPixelBufferGetWidth(result) == 1920 && CVPixelBufferGetHeight(result) == 1080)
        #expect((CVBufferCopyAttachment(result, Anime4KUpscaler.upscaleFactorAttachmentKey, nil) as? NSNumber)?.intValue == 1)
        #expect(abs(Self.lumaCode(result, x: 200, y: 500) - Self.lumaCode(source, x: 200, y: 500)) < 0.02)
        #expect(abs(Self.lumaCode(result, x: 1700, y: 500) - Self.lumaCode(source, x: 1700, y: 500)) < 0.02)
        print("[Anime4KUpscalerTests] Restore (S) 1080p, ms per frame: \(timings.map { String(format: "%.1f", $0) }.joined(separator: ", "))")
    }

    @Test("An ineligible session says why instead of waiting for a frame")
    func ineligibleSessionStatus() async throws {
        let upscaler = Anime4KUpscaler()
        upscaler.setMode(.upscaleEnhance)
        let deadline = Date().addingTimeInterval(60)
        while upscaler.status.state == .preparing, Date() < deadline {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        upscaler.beginSession(ineligibility: "source is larger than 1080p")
        #expect(upscaler.status.state == .bypassed(reason: "source is larger than 1080p"))
        upscaler.beginSession(ineligibility: nil)
        #expect(upscaler.status.state == .waiting)
    }

    @Test("Watchdog: spikes over a healthy median never switch it off")
    func watchdogIgnoresSpikes() {
        var window = UpscaleTimingWindow()
        var tripped = false
        // 28 ms frames with a 10-frame burst at 120 ms every two seconds, for a minute.
        for frame in 0..<(24 * 60) {
            let spike = frame % 48 < 10
            window.record(wall: spike ? 120 : 28, gpu: spike ? 118 : 27)
            if window.isSustainedlyOver(budget: 39.6) { tripped = true }
        }
        #expect(!tripped)
        #expect(window.medianWall == 28)
    }

    @Test("Watchdog: sustained slowness switches it off after the window and a second more")
    func watchdogTripsOnSustainedOverload() {
        var window = UpscaleTimingWindow()
        var trippedAt: Int?
        for frame in 0..<200 {
            window.record(wall: 45, gpu: 44)
            if trippedAt == nil, window.isSustainedlyOver(budget: 39.6) { trippedAt = frame }
        }
        // Full at frame 47; the median is over from then, and 24 consecutive frames take it to 70.
        #expect(trippedAt == UpscaleTimingWindow.capacity + UpscaleTimingWindow.sustainFrames - 2)
    }

    @Test("Eligibility: SDR 4:2:0 up to 1080p only")
    func eligibility() {
        let nv12 = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        #expect(Anime4KUpscaler.ineligibility(width: 1920, height: 1080, pixelFormat: nv12, transfer: nil) == nil)
        #expect(Anime4KUpscaler.ineligibility(width: 1920, height: 1088, pixelFormat: nv12, transfer: nil) == nil)
        #expect(Anime4KUpscaler.ineligibility(width: 3840, height: 2160, pixelFormat: nv12, transfer: nil) != nil)
        #expect(Anime4KUpscaler.ineligibility(
            width: 1920, height: 1080, pixelFormat: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            transfer: kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ as String) == "HDR source")
        #expect(Anime4KUpscaler.ineligibility(
            width: 1920, height: 1080, pixelFormat: kCVPixelFormatType_32BGRA, transfer: nil) != nil)
    }

    @Test("Upscales a 1080p frame to 4K in its own format and keeps the picture",
          arguments: [kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                      kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange])
    func upscalesFrame(pixelFormat: OSType) async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        let upscaler = Anime4KUpscaler()
        upscaler.setMode(.upscaleEnhance)
        let deadline = Date().addingTimeInterval(60)
        while upscaler.status.state == .preparing, Date() < deadline {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        #expect(upscaler.status.state == .waiting)

        let source = try Self.testFrame(pixelFormat: pixelFormat, width: 1920, height: 1080)
        var output: CVPixelBuffer?
        var timings: [Double] = []
        for _ in 0..<5 {
            let started = CFAbsoluteTimeGetCurrent()
            output = upscaler.process(source)
            timings.append((CFAbsoluteTimeGetCurrent() - started) * 1000)
        }
        let result = try #require(output)
        #expect(CVPixelBufferGetWidth(result) == 3840)
        #expect(CVPixelBufferGetHeight(result) == 2160)
        // Always 8-bit: a 10-bit 4K frame would double what every queued frame costs.
        #expect(CVPixelBufferGetPixelFormatType(result) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
        #expect((CVBufferCopyAttachment(result, kCVImageBufferYCbCrMatrixKey, nil) as? String)
                == (kCVImageBufferYCbCrMatrix_ITU_R_709_2 as String))
        #expect(upscaler.status.state == .active)
        print("[Anime4KUpscalerTests] \(pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ? "NV12" : "P010") "
              + "1080p → 4K, ms per frame: \(timings.map { String(format: "%.1f", $0) }.joined(separator: ", "))")

        // The picture survives: the left half was dark and the right half bright, with neutral chroma.
        let darkLeft = Self.lumaCode(result, x: 400, y: 1000)
        let brightRight = Self.lumaCode(result, x: 3400, y: 1000)
        let sourceDark = Self.lumaCode(source, x: 200, y: 500)
        let sourceBright = Self.lumaCode(source, x: 1700, y: 500)
        #expect(abs(darkLeft - sourceDark) < 0.02, "dark \(darkLeft) vs \(sourceDark)")
        #expect(abs(brightRight - sourceBright) < 0.02, "bright \(brightRight) vs \(sourceBright)")
        let chroma = Self.chromaCode(result, x: 1700, y: 500)
        #expect(abs(chroma.0 - 0.5) < 0.02 && abs(chroma.1 - 0.5) < 0.02, "chroma \(chroma)")

        #expect((upscaler.status.memoryBytes ?? 0) > 0)

        upscaler.setMode(.off)
        #expect(upscaler.process(source) == nil)
    }

    @Test("The fused tail draws the same picture as the passes it replaces")
    func fusedTailMatchesPasses() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        let source = try Self.busyFrame(width: 960, height: 540)
        var planes: [Bool: (luma: [UInt8], chroma: [UInt8])] = [:]
        for fused in [true, false] {
            let upscaler = Anime4KUpscaler()
            upscaler.fusesTail = fused
            upscaler.setMode(.upscaleEnhance)
            let deadline = Date().addingTimeInterval(60)
            while upscaler.status.state == .preparing, Date() < deadline {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
            let output = try #require(upscaler.process(source))
            #expect(CVPixelBufferGetWidth(output) == 1920 && CVPixelBufferGetHeight(output) == 1080)
            planes[fused] = (Self.bytes(output, plane: 0), Self.bytes(output, plane: 1))
        }
        let fused = try #require(planes[true]), passes = try #require(planes[false])
        for (name, a, b) in [("luma", fused.luma, passes.luma), ("chroma", fused.chroma, passes.chroma)] {
            var maxDiff = 0, total = 0
            for i in a.indices {
                let d = abs(Int(a[i]) - Int(b[i]))
                maxDiff = max(maxDiff, d)
                total += d
            }
            let mean = Double(total) / Double(a.count)
            print("[Anime4KUpscalerTests] fused vs passes \(name): max \(maxDiff), mean \(String(format: "%.4f", mean)) codes")
            // The passes round the 4K image to half floats between steps; the fused kernel does not.
            #expect(maxDiff <= 2, "\(name) max \(maxDiff)")
            #expect(mean < 0.1, "\(name) mean \(mean)")
        }
    }

    @Test("Holds at most the capped number of upscaled frames, then presents the original")
    func capsFramesInFlight() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        let upscaler = Anime4KUpscaler()
        upscaler.setMode(.upscaleEnhance)
        let deadline = Date().addingTimeInterval(60)
        while upscaler.status.state == .preparing, Date() < deadline {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        let source = try Self.testFrame(pixelFormat: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                                        width: 640, height: 360)
        var held: [CVPixelBuffer] = []
        for _ in 0..<Anime4KUpscaler.maximumFramesInFlight {
            held.append(try #require(upscaler.process(source)))
        }
        // Every pooled frame is still held (queued for display): the next waits, then gives up.
        let started = Date()
        #expect(upscaler.process(source) == nil)
        #expect(Date().timeIntervalSince(started) >= Anime4KUpscaler.outputWaitSeconds * 0.9)
        // One frame shown and released frees a slot.
        held.removeFirst()
        #expect(upscaler.process(source) != nil)
    }

    // MARK: - Fixtures

    /// Line-art-like content: hard-edged shapes over gradients, in both chroma directions.
    static func busyFrame(width: Int, height: Int) throws -> CVPixelBuffer {
        let frame = try testFrame(pixelFormat: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, width: width, height: height)
        CVPixelBufferLockBaseAddress(frame, [])
        defer { CVPixelBufferUnlockBaseAddress(frame, []) }
        for plane in 0..<2 {
            let base = CVPixelBufferGetBaseAddressOfPlane(frame, plane)!
            let stride = CVPixelBufferGetBytesPerRowOfPlane(frame, plane)
            let w = CVPixelBufferGetWidthOfPlane(frame, plane), h = CVPixelBufferGetHeightOfPlane(frame, plane)
            for y in 0..<h {
                let row = base.advanced(by: y * stride)
                for x in 0..<w {
                    let edge = ((x / 17 + y / 11) % 4 == 0)
                    if plane == 0 {
                        row.storeBytes(of: UInt8(edge ? 30 : 60 + (x * 150 / w)), toByteOffset: x, as: UInt8.self)
                    } else {
                        row.storeBytes(of: UInt8(100 + (x * 60 / w)), toByteOffset: x * 2, as: UInt8.self)
                        row.storeBytes(of: UInt8(150 - (y * 50 / h)), toByteOffset: x * 2 + 1, as: UInt8.self)
                    }
                }
            }
        }
        return frame
    }

    static func bytes(_ buffer: CVPixelBuffer, plane: Int) -> [UInt8] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let base = CVPixelBufferGetBaseAddressOfPlane(buffer, plane)!.assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane)
        let rowBytes = CVPixelBufferGetWidthOfPlane(buffer, plane) * (plane == 0 ? 1 : 2)
        var out: [UInt8] = []
        for y in 0..<CVPixelBufferGetHeightOfPlane(buffer, plane) {
            out.append(contentsOf: UnsafeBufferPointer(start: base + y * stride, count: rowBytes))
        }
        return out
    }

    static func testFrame(pixelFormat: OSType, width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
                                         kCVPixelBufferMetalCompatibilityKey as String: true]
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, pixelFormat, attributes as CFDictionary, &buffer)
        let frame = try #require(buffer)
        CVBufferSetAttachment(frame, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
        CVPixelBufferLockBaseAddress(frame, [])
        defer { CVPixelBufferUnlockBaseAddress(frame, []) }
        let tenBit = pixelFormat == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
        for plane in 0..<2 {
            let base = CVPixelBufferGetBaseAddressOfPlane(frame, plane)!
            let stride = CVPixelBufferGetBytesPerRowOfPlane(frame, plane)
            let planeWidth = CVPixelBufferGetWidthOfPlane(frame, plane)
            for y in 0..<CVPixelBufferGetHeightOfPlane(frame, plane) {
                let row = base.advanced(by: y * stride)
                for x in 0..<planeWidth {
                    let bright = x >= planeWidth / 2
                    if plane == 0 {
                        let code = bright ? 180 : 40
                        if tenBit { row.storeBytes(of: UInt16(code * 4) << 6, toByteOffset: x * 2, as: UInt16.self) }
                        else { row.storeBytes(of: UInt8(code), toByteOffset: x, as: UInt8.self) }
                    } else {
                        if tenBit {
                            row.storeBytes(of: UInt16(512) << 6, toByteOffset: x * 4, as: UInt16.self)
                            row.storeBytes(of: UInt16(512) << 6, toByteOffset: x * 4 + 2, as: UInt16.self)
                        } else {
                            row.storeBytes(of: UInt8(128), toByteOffset: x * 2, as: UInt8.self)
                            row.storeBytes(of: UInt8(128), toByteOffset: x * 2 + 1, as: UInt8.self)
                        }
                    }
                }
            }
        }
        return frame
    }

    /// Normalised sample of a plane at (x, y), whatever its depth.
    static func sample(_ buffer: CVPixelBuffer, plane: Int, x: Int, y: Int, component: Int) -> Double {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let base = CVPixelBufferGetBaseAddressOfPlane(buffer, plane)!
        let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane)
        let components = plane == 0 ? 1 : 2
        let tenBit = CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
        let row = base.advanced(by: y * stride)
        if tenBit {
            return Double(row.load(fromByteOffset: (x * components + component) * 2, as: UInt16.self)) / 65535
        }
        return Double(row.load(fromByteOffset: x * components + component, as: UInt8.self)) / 255
    }

    static func lumaCode(_ buffer: CVPixelBuffer, x: Int, y: Int) -> Double {
        sample(buffer, plane: 0, x: x, y: y, component: 0)
    }

    static func chromaCode(_ buffer: CVPixelBuffer, x: Int, y: Int) -> (Double, Double) {
        (sample(buffer, plane: 1, x: x / 2, y: y / 2, component: 0),
         sample(buffer, plane: 1, x: x / 2, y: y / 2, component: 1))
    }
}

/// Opt-in GPU timing (`ANIME4K_UPSCALER_BENCH=1 swift test --filter Anime4KUpscalerBenchmark`): each
/// preset in full and half precision, GPU time per 1080p → 4K frame, and how far half drifts from full.
@Suite("Anime4KUpscalerBenchmark", .enabled(if: ProcessInfo.processInfo.environment["ANIME4K_UPSCALER_BENCH"] == "1"),
       .serialized)
struct Anime4KUpscalerBenchmark {
    /// The whole frame path, 1080p NV12 in → 4K out, with the fused tail and without.
    @Test func timesFusedTail() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        let source = try Anime4KUpscalerTests.busyFrame(width: 1920, height: 1080)
        for fused in [false, true] {
            let upscaler = Anime4KUpscaler()
            upscaler.fusesTail = fused
            upscaler.usesNeuralEngine = false
            upscaler.setMode(.upscaleEnhance)
            let deadline = Date().addingTimeInterval(60)
            while upscaler.status.state == .preparing, Date() < deadline {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
            // Past the warm-up, so the status averages are populated.
            for _ in 0..<(Anime4KUpscaler.warmupFrames + 24) {
                _ = try #require(upscaler.process(source))
            }
            let status = upscaler.status
            print("[Anime4KUpscalerBenchmark] \(fused ? "fused tail" : "passes as written"): "
                  + String(format: "wall %.1f ms, GPU %.1f ms, held %d MB",
                           status.averageFrameMilliseconds ?? -1, status.averageGPUMilliseconds ?? -1,
                           (status.memoryBytes ?? 0) / 1_048_576))
        }
    }

    /// A (S) and A (M) per 1080p frame, the whole frame path, with their networks on the Neural Engine
    /// and on the GPU.
    @Test(arguments: [Anime4KMode.upscaleEnhance, .upscale])
    func timesNeuralEngine(mode: Anime4KMode) async throws {
        guard MTLCreateSystemDefaultDevice() != nil, NeuralEngineUpscaler.deviceHasNeuralEngine else { return }
        let source = try Anime4KUpscalerTests.busyFrame(width: 1920, height: 1080)
        for neural in [true, false] {
            let upscaler = Anime4KUpscaler()
            upscaler.usesNeuralEngine = neural
            upscaler.setMode(mode)
            try await Anime4KUpscalerTests.ready(upscaler)
            for _ in 0..<(Anime4KUpscaler.warmupFrames + 48) {
                _ = try #require(upscaler.process(source))
            }
            let status = upscaler.status
            print("[Anime4KUpscalerBenchmark] \(status.preset ?? "?") on the \(status.backend ?? "?"): "
                  + String(format: "wall %.1f ms, GPU %.1f ms, Neural Engine %.1f ms, held %d MB",
                           status.averageFrameMilliseconds ?? -1, status.averageGPUMilliseconds ?? -1,
                           status.averageNeuralEngineMilliseconds ?? -1, (status.memoryBytes ?? 0) / 1_048_576))
        }
    }

    @Test func timesPresets() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        let sampler = try #require(device.makeSamplerState(descriptor: samplerDescriptor))

        // A deterministic busy picture: hard-edged shapes over a gradient, like line art.
        let width = 1920, height = 1080
        var pixels = [Float16](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                let ring = ((x / 37 + y / 23) % 5 == 0) ? Float(0.05) : Float(0.85)
                pixels[i] = Float16(Float(x) / Float(width) * ring)
                pixels[i + 1] = Float16(Float(y) / Float(height) * ring)
                pixels[i + 2] = Float16(ring)
                pixels[i + 3] = 1
            }
        }
        let inputDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        inputDescriptor.usage = [.shaderRead]
        let input = try #require(device.makeTexture(descriptor: inputDescriptor))
        input.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                      withBytes: pixels, bytesPerRow: width * 8)

        for preset in Anime4KUpscaler.allPresets {
            let shaders = try preset.files.flatMap { try Anime4KUpscalerTests.shaders($0) }
            var results: [Bool: [Float16]] = [:]
            for half in [false, true] {
                let pipeline = try MPVHookPipeline(device: device, shaders: shaders, halfPrecision: half)
                var pool = TexturePool()
                pool.prepare(forInput: (width, height))
                var gpu: [Double] = []
                var last: MTLTexture?
                for _ in 0..<8 {
                    let commandBuffer = try #require(queue.makeCommandBuffer())
                    last = try pipeline.encode(commandBuffer: commandBuffer, input: input,
                                               output: (width * 2, height * 2), sampler: sampler,
                                               pool: &pool, device: device)
                    commandBuffer.commit()
                    commandBuffer.waitUntilCompleted()
                    gpu.append((commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000)
                }
                let steady = gpu.dropFirst(2).sorted()
                print("[Anime4KUpscalerBenchmark] \(preset.name) \(half ? "half" : "full"): "
                      + String(format: "median GPU %.1f ms", steady[steady.count / 2]))
                // Read the result back for the precision comparison.
                let out = try #require(last)
                let readDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .rgba16Float, width: out.width, height: out.height, mipmapped: false)
                readDescriptor.storageMode = .shared
                let readable = try #require(device.makeTexture(descriptor: readDescriptor))
                let blitBuffer = try #require(queue.makeCommandBuffer())
                let blit = try #require(blitBuffer.makeBlitCommandEncoder())
                blit.copy(from: out, to: readable)
                blit.endEncoding()
                blitBuffer.commit()
                blitBuffer.waitUntilCompleted()
                var values = [Float16](repeating: 0, count: out.width * out.height * 4)
                readable.getBytes(&values, bytesPerRow: out.width * 8,
                                  from: MTLRegionMake2D(0, 0, out.width, out.height), mipmapLevel: 0)
                results[half] = values
            }
            let full = try #require(results[false]), half = try #require(results[true])
            var maxDiff: Float = 0, sumDiff: Float = 0
            for i in stride(from: 0, to: full.count, by: 4) {
                for c in 0..<3 {
                    let d = abs(Float(full[i + c]) - Float(half[i + c]))
                    maxDiff = max(maxDiff, d)
                    sumDiff += d
                }
            }
            print("[Anime4KUpscalerBenchmark] \(preset.name) half vs full: "
                  + String(format: "max %.4f, mean %.6f (8-bit step is 0.0039)", maxDiff, sumDiff / Float(full.count / 4 * 3)))
        }
    }
}
