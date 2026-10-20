import CoreGraphics
import CoreVideo
import Foundation
import Metal
import os

/// What `Anime4KUpscaler` does to each frame. Every mode steps back, then switches itself off, when
/// the device cannot keep up with the source's frame rate (see `Anime4KStatus`).
public enum Anime4KMode: String, Sendable, Equatable, CaseIterable {
    case off
    /// Restore (S): Anime4K's restore network and de-ring clamp at the source size. Sharper lines, less
    /// blur and compression damage; the frame keeps its size. GPU (Metal). Off when it cannot keep up.
    case enhance
    /// C (S): Anime4K's upscale-and-denoise network, 2× (1080p → 4K). Neural Engine where the device
    /// has one and the source size is one the model was built for, the GPU otherwise. Off when it
    /// cannot keep up.
    case upscale
    /// A (S): the restore network, then Anime4K's 2× upscale network (1080p → 4K). Neural Engine
    /// where possible, as `upscale`. Steps back to `enhance` when it cannot keep up, then off.
    case upscaleEnhance
}

/// What the upscaler is doing right now, for diagnostics. Read `Anime4KUpscaler.status`; it is a
/// snapshot, not a publisher.
public struct Anime4KStatus: Sendable, Equatable {
    public enum State: Sendable, Equatable {
        case off
        /// Shaders are compiling; frames pass through untouched meanwhile.
        case preparing
        /// Turned on, but no frame has reached it: the session is on the native path, or paused.
        case waiting
        case active
        /// The current frames are not eligible (HDR, larger than 1080p, an unsupported pixel format).
        case bypassed(reason: String)
        /// Gave up for the rest of this mode: no Metal device, a shader failed, or the GPU cannot keep up.
        case failed(reason: String)
    }

    public var state: State
    /// The preset in use, e.g. "Anime4K A (M)". Steps down to the lighter preset before failing.
    public var preset: String?
    /// Where the networks ran for the last frame: "Neural Engine" or "GPU".
    public var backend: String?
    /// Why a preset that can use the Neural Engine is on the GPU instead, e.g. "no Neural Engine".
    public var neuralEngineNote: String?
    public var inputSize: CGSize?
    public var outputSize: CGSize?
    /// Median wall time to upscale one frame over the last couple of seconds, GPU work and the CPU
    /// encode included. A median, so a burst of slow frames does not move it.
    public var averageFrameMilliseconds: Double?
    /// The same median for the time the GPU itself spent on a frame. The rest of the wall time is
    /// encoding the work and waiting for it to be scheduled.
    public var averageGPUMilliseconds: Double?
    /// The same median for the Neural Engine's prediction, when the networks ran there.
    public var averageNeuralEngineMilliseconds: Double?
    /// The per-frame time the watchdog allows: the source's frame interval times `budgetShare`.
    public var budgetMilliseconds: Double?
    /// GPU memory the upscaler holds: its working textures and its pool of upscaled frames.
    public var memoryBytes: Int?
    /// How far the app is from tvOS's per-process memory limit, when the system reports it.
    public var availableMemoryBytes: Int?

    public static let off = Anime4KStatus(state: .off)
}

/// Runs Anime4K (bloc97/Anime4K) on decoded video frames: 8- or 10-bit 4:2:0 `CVPixelBuffer`s (NV12 /
/// P010, video or full range) up to 1920×1088, SDR only. Returns a new 8-bit 4:2:0 buffer in the same
/// range and matrix — the source's size for `.enhance`, twice it for the upscaling modes — or nil, in
/// which case present the original: the upscaler can cost sharpness, never a picture.
///
/// How a player should drive it (each point was learned on an Apple TV 4K):
/// - Call `process(_:)` on a serial queue of your own, **never on the decoder's callback**. It waits
///   for the GPU and, on the Neural Engine path, for a Core ML prediction; on a thread that also feeds
///   audio that starved the audio renderer.
/// - Keep at most `maximumFramesInFlight` returned buffers alive (queued for display). At the cap,
///   `process` waits up to 250 ms for one to be released, then returns nil.
/// - Call `setFrameRate(_:)` per session: the watchdog judges against that frame interval.
/// - Call `beginSession()` when a new video starts, so the status forgets the last one.
/// - A returned buffer carries `upscaleFactorAttachmentKey` (1 or 2), so a player can still report
///   the source's own size.
/// - `setMode(_:)` prepares in the background (shaders compile, models load); frames return nil until
///   it is ready.
public final class Anime4KUpscaler: @unchecked Sendable {

    /// The pixel buffer attachment carrying the factor a frame was upscaled by, so the renderer can
    /// report the source's size rather than the upscaled one. Never propagated.
    public static var upscaleFactorAttachmentKey: CFString { "Anime4KUpscaleFactor" as CFString }

    struct Preset: Equatable {
        let name: String
        let files: [String]
        /// How much the chain enlarges the frame: 2 for the upscaling chains, 1 for restore only.
        var scale: Int = 2
        /// The Core ML model that runs its networks on the Neural Engine (`NeuralEngineUpscaler`), if any.
        var neuralEngineModel: String?
    }

    /// Anime4K's "Mode A" chain for 1080p → 4K: de-ring statistics, CNN restore, CNN 2× upscale and
    /// the de-ring clamp.
    static let smallPreset = Preset(name: "Anime4K A (S)", files: [
        "Anime4K_Clamp_Highlights", "Anime4K_Restore_CNN_S", "Anime4K_Upscale_CNN_x2_S"],
        neuralEngineModel: "Anime4K_A_S")
    /// One network instead of two: Anime4K's upscale-and-denoise (its "Mode C" building block) with
    /// the de-ring clamp. About half the work of A (S), with less restoration.
    static let lightPreset = Preset(name: "Anime4K C (S)", files: [
        "Anime4K_Clamp_Highlights", "Anime4K_Upscale_Denoise_CNN_x2_S"],
        neuralEngineModel: "Anime4K_C_S")
    /// A (S)'s first half alone: the restore network and the de-ring clamp at the source size. The
    /// output is not enlarged, so no 4K frame exists and the work is a fraction of A (S)'s.
    static let restoreOnlyPreset = Preset(name: "Anime4K Restore (S)", files: [
        "Anime4K_Clamp_Highlights", "Anime4K_Restore_CNN_S"], scale: 1)

    /// The step-down order a mode runs.
    static func presets(for mode: Anime4KMode) -> [Preset] {
        switch mode {
        case .off: return []
        case .enhance: return [restoreOnlyPreset]
        case .upscale: return [lightPreset]
        case .upscaleEnhance: return [smallPreset, restoreOnlyPreset]
        }
    }

    /// Every preset bundled, for the benchmark.
    static let allPresets: [Preset] = [smallPreset, lightPreset, restoreOnlyPreset]

    /// Largest source it upscales: 2× of it is the 4K canvas (1088 covers coded-height H.264).
    public static let maximumSourceSize = (width: 1920, height: 1088)
    /// Frames averaged before the watchdog judges, past the first frames that include shader warm-up.
    static let warmupFrames = 12
    /// Share of the frame interval an upscale may take on average before the watchdog steps down.
    /// 95%: on an Apple TV 4K (3rd gen) S measured 36 ms against a 41.7 ms frame and played smoothly,
    /// while 85% (35.5 ms) switched it off. The upscale runs beside decoding, not in series with it.
    static let budgetShare = 0.95
    /// Upscaled frames allowed in flight (the renderer's reorder buffer plus its queue). At the cap
    /// the decode thread waits for the screen to release one rather than allocating more: unbounded,
    /// the pool grew with the renderer's queue and took an app past tvOS's 2 GB limit on device.
    /// 16: the software path keeps about 14 frames between its reorder buffer and the layer's queue,
    /// so a cap of 10 made almost every frame wait for one to be shown. 16 × 12.4 MB at 4K 8-bit.
    public static let maximumFramesInFlight = 16
    /// How long the decode thread waits for a free output frame before presenting the original.
    static let outputWaitSeconds = 0.25
    /// Memory the app must still have free beyond what the upscaler is about to allocate, and the
    /// floor under which a running upscaler gives its memory back. `os_proc_available_memory` is
    /// the distance to the per-process jetsam limit, so this is the margin tvOS kills at.
    static let startHeadroomBytes = 250 * 1_048_576
    static let runningFloorBytes = 150 * 1_048_576

    private let stateLock = NSLock()
    private var mode: Anime4KMode = .off
    private var resources: Resources?
    private var preparing = false
    private var failure: String?
    private var presetIndex = 0
    private var bypassReason: String?
    private var lastInputSize: CGSize?
    private var lastOutputSize: CGSize?
    private var timing = UpscaleTimingWindow()
    private var framesTimed = 0
    /// Slow frames since the last slow-frame line, and when that line was written: one line per
    /// burst rather than one per frame, so a log of a long episode stays readable.
    private var slowFramesSinceLog = 0
    private var lastSlowLog = Date.distantPast
    private var frameIntervalSeconds: Double?
    private var processedAny = false
    private var heldBytes: Int?
    /// Where the last frame's networks ran, for the status.
    private var lastBackend: String?
    /// The Neural Engine models by name, once loaded; each loaded the first time a mode wants it.
    private var neuralEngines: [String: NeuralEngineUpscaler] = [:]
    private var neuralEngineLoadsStarted: Set<String> = []
    /// Why a model is not in use (not loaded yet, unavailable, or failed), by name.
    private var neuralEngineNotes: [String: String] = [:]

    /// Whether the last passes run as the one fused kernel. Only the tests turn it off, to compare
    /// against the passes as written.
    var fusesTail = true
    /// Whether a preset that can use the Neural Engine does. Only the tests turn it off, to compare
    /// the two backends.
    var usesNeuralEngine = true

    /// Serialises GPU work and owns the texture pools.
    private let workLock = NSLock()
    private var intermediates = TexturePool()
    private var outputPool: (key: OutputKey, pool: CVPixelBufferPool)?
    /// The Neural Engine's input and output tensors for the current source size.
    private var neuralBuffers: (key: NeuralEngineUpscaler.Size, input: CVPixelBuffer, output: CVPixelBuffer)?

    public init() {}

    // MARK: - Control

    public var currentMode: Anime4KMode {
        stateLock.lock(); defer { stateLock.unlock() }
        return mode
    }

    public func setMode(_ newMode: Anime4KMode) {
        stateLock.lock()
        let changed = newMode != mode
        mode = newMode
        if changed {
            // A fresh attempt: the watchdog's verdict and the preset step-down were for the last one.
            presetIndex = 0
            timing = UpscaleTimingWindow()
            framesTimed = 0
            bypassReason = nil
            processedAny = false
            if resources != nil { failure = nil }
        }
        let needsPreparation = newMode != .off && resources == nil && !preparing && failure == nil
        if needsPreparation { preparing = true }
        let modelsToLoad = Self.presets(for: newMode).compactMap(\.neuralEngineModel)
            .filter { !neuralEngineLoadsStarted.contains($0) }
        for model in modelsToLoad {
            neuralEngineLoadsStarted.insert(model)
            neuralEngineNotes[model] = Self.loadingNote
        }
        stateLock.unlock()

        if newMode == .off { releaseIntermediates() }
        if needsPreparation {
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in self?.prepare() }
        }
        if !modelsToLoad.isEmpty {
            // Separately from the shaders: the first load on a device compiles a model for its
            // Neural Engine, seconds, and the GPU path can run meanwhile. In step-down order.
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                for model in modelsToLoad { self?.loadNeuralEngine(model) }
            }
        }
        Anime4KLog.emit("[Upscaler] mode \(newMode.rawValue)")
    }

    /// The source's frame interval, which sets the watchdog's budget. Nil assumes 24 fps.
    public func setFrameRate(_ fps: Double?) {
        stateLock.lock()
        frameIntervalSeconds = fps.flatMap { $0 > 0 ? 1 / $0 : nil }
        stateLock.unlock()
    }

    /// Drop the per-size textures and the output pool (hundreds of MB at 1080p). The compiled
    /// shaders stay; they are rebuilt only by a new process.
    public func releaseIntermediates() {
        workLock.lock()
        intermediates = TexturePool()
        outputPool = nil
        neuralBuffers = nil
        workLock.unlock()
        stateLock.lock()
        heldBytes = nil
        stateLock.unlock()
    }

    public var status: Anime4KStatus {
        stateLock.lock(); defer { stateLock.unlock() }
        // The limit the watchdog enforces, so the stats line and its "too slow" verdict agree.
        let budget = (frameIntervalSeconds ?? 1 / 24) * 1000 * Self.budgetShare
        var status = Anime4KStatus(state: .off)
        let modePresets = Self.presets(for: mode)
        guard mode != .off, presetIndex < modePresets.count else { return status }
        status.preset = modePresets[presetIndex].name
        status.backend = lastBackend
        if let model = modePresets[presetIndex].neuralEngineModel { status.neuralEngineNote = neuralEngineNotes[model] }
        status.budgetMilliseconds = budget
        status.averageFrameMilliseconds = timing.medianWall
        status.averageGPUMilliseconds = timing.medianGPU
        status.averageNeuralEngineMilliseconds = timing.medianNeuralEngine
        status.inputSize = lastInputSize
        status.outputSize = lastOutputSize
        status.memoryBytes = heldBytes
        status.availableMemoryBytes = Self.availableMemory()
        if let failure {
            status.state = .failed(reason: failure)
        } else if preparing || resources == nil {
            status.state = .preparing
        } else if let bypassReason {
            status.state = .bypassed(reason: bypassReason)
        } else if processedAny {
            status.state = .active
        } else {
            status.state = .waiting
        }
        return status
    }

    // MARK: - Preparation

    private final class Resources {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let textureCache: CVMetalTextureCache
        let yuvToRGB: MTLComputePipelineState
        let rgbToLuma: MTLComputePipelineState
        let rgbToChroma: MTLComputePipelineState
        let fusedTail: MTLComputePipelineState?
        /// The Neural Engine path's two kernels: YCbCr to the model's planar RGB, and the fused tail
        /// reading the model's planes. Optional: without them A (S) stays on the GPU.
        let yuvToPlanar: MTLComputePipelineState?
        let planarTail: MTLComputePipelineState?
        /// Compiled chains by preset name: every preset any mode can run.
        let presets: [String: MPVHookPipeline]
        let sampler: MTLSamplerState

        init(device: MTLDevice, queue: MTLCommandQueue, textureCache: CVMetalTextureCache,
             yuvToRGB: MTLComputePipelineState, rgbToLuma: MTLComputePipelineState,
             rgbToChroma: MTLComputePipelineState, fusedTail: MTLComputePipelineState?,
             yuvToPlanar: MTLComputePipelineState?, planarTail: MTLComputePipelineState?,
             presets: [String: MPVHookPipeline], sampler: MTLSamplerState) {
            self.device = device
            self.queue = queue
            self.textureCache = textureCache
            self.yuvToRGB = yuvToRGB
            self.rgbToLuma = rgbToLuma
            self.rgbToChroma = rgbToChroma
            self.fusedTail = fusedTail
            self.yuvToPlanar = yuvToPlanar
            self.planarTail = planarTail
            self.presets = presets
            self.sampler = sampler
        }
    }

    enum PreparationError: Error, CustomStringConvertible {
        case noDevice
        case noQueue
        case noTextureCache
        case missingShader(String)
        case missingFunction(String)

        var description: String {
            switch self {
            case .noDevice: return "no Metal device"
            case .noQueue: return "no Metal command queue"
            case .noTextureCache: return "no Metal texture cache"
            case .missingShader(let name): return "shader \(name) is missing from the bundle"
            case .missingFunction(let name): return "kernel \(name) did not compile"
            }
        }
    }

    private func prepare() {
        let started = Date()
        let result = Result { try Self.buildResources() }
        stateLock.lock()
        preparing = false
        switch result {
        case .success(let built):
            resources = built
            failure = nil
        case .failure(let error):
            failure = "could not prepare the shaders: \(error)"
        }
        stateLock.unlock()
        switch result {
        case .success:
            Anime4KLog.emit(
                "[Upscaler] shaders ready in \(Int(Date().timeIntervalSince(started) * 1000)) ms")
        case .failure(let error):
            Anime4KLog.emit("[Upscaler] preparation failed: \(error)")
        }
    }

    /// What the status says while a model loads; the tests wait for it to go.
    static let loadingNote = "loading the Neural Engine model"

    private func loadNeuralEngine(_ model: String) {
        let started = Date()
        let result = Result { try NeuralEngineUpscaler(modelName: model) }
        let elapsed = Int(Date().timeIntervalSince(started) * 1000)
        stateLock.lock()
        switch result {
        case .success(let loaded):
            neuralEngines[model] = loaded
            neuralEngineNotes[model] = nil
        case .failure(let error):
            neuralEngineNotes[model] = "Neural Engine unavailable: \(error)"
        }
        stateLock.unlock()
        switch result {
        case .success(let loaded):
            let sizes = loaded.sizes.map { "\($0.width)x\($0.height)" }.sorted().joined(separator: ", ")
            Anime4KLog.emit("[Upscaler] Neural Engine model \(model) ready in \(elapsed) ms (\(sizes))")
        case .failure(let error):
            Anime4KLog.emit("[Upscaler] Neural Engine model \(model) not used: \(error)")
        }
    }

    /// A bundled Anime4K shader file, by name without extension.
    static func shaderURL(_ file: String) -> URL? {
        Bundle.module.url(forResource: file, withExtension: "glsl", subdirectory: "Anime4K")
    }

    private static func buildResources() throws -> Resources {
        guard let device = MTLCreateSystemDefaultDevice() else { throw PreparationError.noDevice }
        guard let queue = device.makeCommandQueue() else { throw PreparationError.noQueue }
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache) == kCVReturnSuccess,
              let textureCache = cache else { throw PreparationError.noTextureCache }

        let library = try device.makeLibrary(source: conversionSource, options: nil)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw PreparationError.missingFunction(name)
            }
            return try device.makeComputePipelineState(function: function)
        }

        var shadersByFile: [String: [MPVHookShader]] = [:]
        let runnable = allPresets
        let pipelines = try runnable.map { preset -> MPVHookPipeline in
            let shaders = try preset.files.flatMap { file -> [MPVHookShader] in
                if let parsed = shadersByFile[file] { return parsed }
                guard let url = shaderURL(file) else {
                    throw PreparationError.missingShader(file)
                }
                let parsed = try MPVHookShader.parse(String(contentsOf: url, encoding: .utf8))
                shadersByFile[file] = parsed
                return parsed
            }
            return try MPVHookPipeline(device: device, shaders: shaders)
        }

        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.sAddressMode = .clampToEdge
        samplerDescriptor.tAddressMode = .clampToEdge
        samplerDescriptor.normalizedCoordinates = true
        guard let sampler = device.makeSamplerState(descriptor: samplerDescriptor) else {
            throw PreparationError.noQueue
        }

        return Resources(
            device: device, queue: queue, textureCache: textureCache,
            yuvToRGB: try pipeline("upscaler_yuv_to_rgb"),
            rgbToLuma: try pipeline("upscaler_rgb_to_luma"),
            rgbToChroma: try pipeline("upscaler_rgb_to_chroma"),
            // Optional: without it every frame takes the passes as written.
            fusedTail: try? pipeline("upscaler_fused_tail"),
            yuvToPlanar: try? pipeline("upscaler_yuv_to_planar"),
            planarTail: try? pipeline("upscaler_planar_tail"),
            presets: Dictionary(uniqueKeysWithValues: zip(runnable.map(\.name), pipelines)), sampler: sampler)
    }

    // MARK: - Per frame

    struct PlaneFormat: Equatable {
        let luma: MTLPixelFormat
        let chroma: MTLPixelFormat
        let tenBit: Bool
        let fullRange: Bool
    }

    static func planeFormat(for pixelFormat: OSType) -> PlaneFormat? {
        switch pixelFormat {
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            return PlaneFormat(luma: .r8Unorm, chroma: .rg8Unorm, tenBit: false, fullRange: false)
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
            return PlaneFormat(luma: .r8Unorm, chroma: .rg8Unorm, tenBit: false, fullRange: true)
        case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange:
            return PlaneFormat(luma: .r16Unorm, chroma: .rg16Unorm, tenBit: true, fullRange: false)
        case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
            return PlaneFormat(luma: .r16Unorm, chroma: .rg16Unorm, tenBit: true, fullRange: true)
        default:
            return nil
        }
    }

    /// A new session: forget the last one's frames and reasons. With `ineligibility`, the session
    /// will not be upscaled at all and the status says why, rather than "waiting for a frame" that
    /// never comes (a 4K file plays on the native path, which no upscaler can see).
    public func beginSession(ineligibility: String? = nil) {
        stateLock.lock()
        bypassReason = ineligibility
        processedAny = false
        lastInputSize = nil
        lastOutputSize = nil
        stateLock.unlock()
    }

    /// Why a frame is not upscaled, or nil when it can be.
    static func ineligibility(width: Int, height: Int, pixelFormat: OSType, transfer: String?) -> String? {
        if planeFormat(for: pixelFormat) == nil { return "unsupported pixel format" }
        if let transfer,
           transfer == (kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ as String)
            || transfer == (kCVImageBufferTransferFunction_ITU_R_2100_HLG as String) {
            return "HDR source"
        }
        if width > maximumSourceSize.width || height > maximumSourceSize.height {
            return "source is larger than 1080p"
        }
        if width < 32 || height < 32 { return "source is too small" }
        return nil
    }

    /// The parameters the conversion kernels share: range offsets and scales for the plane's bit
    /// layout, the YCbCr matrix coefficients, and the dither step for an 8-bit output.
    struct ConversionParams {
        var yOffset: Float
        var yScale: Float
        var cOffset: Float
        var cScale: Float
        var kr: Float
        var kb: Float
        var dither: Float
        var padding: Float = 0

        init(format: PlaneFormat, matrix: String?, height: Int) {
            // 10-bit samples sit in the top bits of a 16-bit word (P010 / x420).
            let unit: Float = format.tenBit ? 64.0 / 65535.0 : 1.0 / 255.0
            if format.tenBit {
                yOffset = format.fullRange ? 0 : 64 * unit
                yScale = (format.fullRange ? 1023 : 876) * unit
                cOffset = 512 * unit
                cScale = (format.fullRange ? 1023 : 896) * unit
            } else {
                yOffset = format.fullRange ? 0 : 16 * unit
                yScale = (format.fullRange ? 255 : 219) * unit
                cOffset = 128 * unit
                cScale = (format.fullRange ? 255 : 224) * unit
            }
            switch matrix {
            case .some(let m) where m == (kCVImageBufferYCbCrMatrix_ITU_R_601_4 as String):
                kr = 0.299; kb = 0.114
            case .some(let m) where m == (kCVImageBufferYCbCrMatrix_ITU_R_2020 as String):
                kr = 0.2627; kb = 0.0593
            case .some(let m) where m == (kCVImageBufferYCbCrMatrix_ITU_R_709_2 as String):
                kr = 0.2126; kb = 0.0722
            default:
                // Untagged: SD is BT.601, everything else BT.709. The output is written with the
                // same matrix, so a wrong guess shifts only what the network sees, not the result's colour.
                if height < 720 { kr = 0.299; kb = 0.114 } else { kr = 0.2126; kb = 0.0722 }
            }
            dither = format.tenBit ? 0 : 1.0 / 255.0
        }

        /// The same matrix and range, written as 8-bit with dither: the upscaled frame. SDR is shown
        /// at 8 bits per channel, and a 10-bit 4K frame would double the memory every queued frame costs.
        func eightBitOutput(fullRange: Bool) -> ConversionParams {
            var out = ConversionParams(
                format: PlaneFormat(luma: .r8Unorm, chroma: .rg8Unorm, tenBit: false, fullRange: fullRange),
                matrix: nil, height: 1080)
            out.kr = kr
            out.kb = kb
            return out
        }
    }

    static func eightBitFormat(fullRange: Bool) -> OSType {
        fullRange ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
    }

    /// Upscale `source`, or nil to have the caller present it as it is.
    public func process(_ source: CVPixelBuffer) -> CVPixelBuffer? {
        stateLock.lock()
        let modePresets = Self.presets(for: mode)
        guard mode != .off, failure == nil, let resources, presetIndex < modePresets.count,
              let pipeline = resources.presets[modePresets[presetIndex].name] else {
            stateLock.unlock()
            return nil
        }
        let scale = modePresets[presetIndex].scale
        let neuralEngineModel = usesNeuralEngine ? modePresets[presetIndex].neuralEngineModel : nil
        let neuralEngine = neuralEngineModel.flatMap { neuralEngines[$0] }
        stateLock.unlock()

        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        let pixelFormat = CVPixelBufferGetPixelFormatType(source)
        let transfer = CVBufferCopyAttachment(source, kCVImageBufferTransferFunctionKey, nil) as? String
        if let reason = Self.ineligibility(width: width, height: height, pixelFormat: pixelFormat, transfer: transfer) {
            noteBypass(reason, size: CGSize(width: width, height: height))
            return nil
        }
        guard let format = Self.planeFormat(for: pixelFormat) else { return nil }
        let matrix = CVBufferCopyAttachment(source, kCVImageBufferYCbCrMatrixKey, nil) as? String
        var inputParams = ConversionParams(format: format, matrix: matrix, height: height)
        var outputParams = inputParams.eightBitOutput(fullRange: format.fullRange)

        if let refusal = memoryRefusal(width: width, height: height) {
            fail(refusal)
            return nil
        }

        // The Neural Engine runs the networks when the preset can use it, the model is loaded and was
        // built for this size; otherwise the same preset runs on the GPU.
        let onNeuralEngine = scale == 2 && neuralEngine?.supports(width: width, height: height) == true
            && resources.yuvToPlanar != nil && resources.planarTail != nil

        workLock.lock()
        let result: FrameResult?
        do {
            if let neuralEngine, onNeuralEngine {
                result = try encodeNeuralEngine(source: source, width: width, height: height, format: format,
                                                inputParams: &inputParams, outputParams: &outputParams,
                                                resources: resources, model: neuralEngine)
            } else {
                result = try encodeAndWait(source: source, width: width, height: height, format: format,
                                           inputParams: &inputParams, outputParams: &outputParams,
                                           resources: resources, pipeline: pipeline, scale: scale)
            }
        } catch let error as NeuralEnginePredictionFailed {
            // The model refused this frame: present the original and keep A (S) on the GPU for the
            // rest of the process rather than switching the upscaler off.
            workLock.unlock()
            stateLock.lock()
            if let neuralEngineModel {
                neuralEngines[neuralEngineModel] = nil
                neuralEngineNotes[neuralEngineModel] = "Neural Engine failed: \(error.underlying)"
            }
            stateLock.unlock()
            Anime4KLog.emit("[Upscaler] Neural Engine failed, using the GPU: \(error.underlying)")
            return nil
        } catch {
            workLock.unlock()
            fail("GPU error: \(error)")
            return nil
        }
        let held = intermediates.allocatedBytes + outputBytesHeld + neuralBytesHeld
        workLock.unlock()
        stateLock.lock()
        heldBytes = held
        stateLock.unlock()
        guard let result else { return nil }
        let output = result.buffer

        Self.copyAttachments(from: source, to: output, scale: scale)
        noteFrame(milliseconds: result.workMilliseconds, gpuMilliseconds: result.gpuMilliseconds,
                  neuralEngineMilliseconds: result.neuralEngineMilliseconds,
                  input: CGSize(width: width, height: height),
                  output: CGSize(width: CVPixelBufferGetWidth(output), height: CVPixelBufferGetHeight(output)))
        return output
    }

    private struct FrameResult {
        let buffer: CVPixelBuffer
        /// Everything after a free output frame was found: GPU, Neural Engine and the waits between.
        let workMilliseconds: Double
        let gpuMilliseconds: Double
        /// The model's prediction, when the networks ran on the Neural Engine.
        var neuralEngineMilliseconds: Double?
    }

    private struct NeuralEnginePredictionFailed: Error {
        let underlying: Error
    }

    /// Why there is not enough memory to go on, or nil. Before the first frame at a size it asks for
    /// the whole working set plus a margin; once running, only that the app keeps a floor free, since
    /// what it already holds is counted in what the system reports as used.
    private func memoryRefusal(width: Int, height: Int) -> String? {
        // No answer on the Mac test host, or for a process without a memory limit.
        guard let available = Self.availableMemory() else { return nil }
        workLock.lock()
        let running = intermediates.isPrepared(forInput: (width, height))
            || neuralBuffers?.key == NeuralEngineUpscaler.Size(width: width, height: height)
        workLock.unlock()
        let needed = running
            ? Self.runningFloorBytes
            : Self.estimatedBytes(width: width, height: height) + Self.startHeadroomBytes
        guard available < needed else { return nil }
        return "not enough memory (\(available / 1_048_576) MB free, needs \(needed / 1_048_576) MB)"
    }

    static func availableMemory() -> Int? {
        #if os(macOS)
        return nil
        #else
        let available = os_proc_available_memory()
        return available > 0 ? available : nil
        #endif
    }

    /// The working set at a source size: about a dozen half-float RGBA textures at the source size and
    /// the capped pool of 8-bit 4:2:0 output frames. The fused tail writes no upscaled-size half-float
    /// image; the unfused fallback adds two (~130 MB at 1080p), which the running floor absorbs.
    static func estimatedBytes(width: Int, height: Int) -> Int {
        let source = width * height * 8
        let upscaled = width * 2 * height * 2
        return source * 12 + upscaled * 3 / 2 * maximumFramesInFlight
    }

    private var outputBytesHeld: Int {
        guard let key = outputPool?.key else { return 0 }
        return key.width * key.height * 3 / 2 * Self.maximumFramesInFlight
    }

    /// The Neural Engine's input and output tensors (half floats, 3 + 8 planes at the source size).
    /// The model's own working memory is the system's and is not counted.
    private var neuralBytesHeld: Int {
        guard let key = neuralBuffers?.key else { return 0 }
        return key.width * key.height * 2
            * (NeuralEngineUpscaler.inputChannels + NeuralEngineUpscaler.outputChannels)
    }

    /// A (S) with its networks on the Neural Engine: YCbCr → planar RGB on the GPU, the model, then
    /// the fused tail on the GPU reading the model's planes. Three steps in series; each waits for
    /// the one before, so the work is timed as a whole.
    private func encodeNeuralEngine(
        source: CVPixelBuffer, width: Int, height: Int, format: PlaneFormat,
        inputParams: inout ConversionParams, outputParams: inout ConversionParams,
        resources: Resources, model: NeuralEngineUpscaler
    ) throws -> FrameResult? {
        defer { CVMetalTextureCacheFlush(resources.textureCache, 0) }
        guard let toPlanar = resources.yuvToPlanar, let tail = resources.planarTail else { return nil }
        let key = NeuralEngineUpscaler.Size(width: width, height: height)
        if neuralBuffers?.key != key {
            neuralBuffers = nil
            guard let input = NeuralEngineUpscaler.makeTensorBuffer(
                    width: width, height: height, channels: NeuralEngineUpscaler.inputChannels),
                  let output = NeuralEngineUpscaler.makeTensorBuffer(
                    width: width, height: height, channels: NeuralEngineUpscaler.outputChannels) else {
                throw TexturePool.AllocationFailed(key: "Neural Engine tensor")
            }
            neuralBuffers = (key, input, output)
        }
        guard let tensors = neuralBuffers else { return nil }

        guard let destination = makeOutputBuffer(pixelFormat: Self.eightBitFormat(fullRange: format.fullRange),
                                                 width: width * 2, height: height * 2) else {
            return nil
        }
        let workStarted = CFAbsoluteTimeGetCurrent()
        guard let sourceLuma = Self.planeTexture(source, plane: 0, format: format.luma, cache: resources.textureCache),
              let sourceChroma = Self.planeTexture(source, plane: 1, format: format.chroma, cache: resources.textureCache),
              let inputPlanes = Self.wholeTexture(tensors.input, format: .r16Float, cache: resources.textureCache),
              let outputPlanes = Self.wholeTexture(tensors.output, format: .r16Float, cache: resources.textureCache),
              let destLuma = Self.planeTexture(destination, plane: 0, format: .r8Unorm, cache: resources.textureCache),
              let destChroma = Self.planeTexture(destination, plane: 1, format: .rg8Unorm, cache: resources.textureCache),
              let before = resources.queue.makeCommandBuffer() else {
            return nil
        }
        try Self.dispatch(before, toPlanar, textures: [sourceLuma.texture, sourceChroma.texture, inputPlanes.texture],
                          params: &inputParams, grid: (width, height))
        before.commit()
        before.waitUntilCompleted()
        if let error = before.error { throw error }

        let predictionStarted = CFAbsoluteTimeGetCurrent()
        do {
            try model.predict(input: tensors.input, output: tensors.output, width: width, height: height)
        } catch {
            throw NeuralEnginePredictionFailed(underlying: error)
        }
        let prediction = (CFAbsoluteTimeGetCurrent() - predictionStarted) * 1000

        guard let after = resources.queue.makeCommandBuffer() else { return nil }
        try Self.dispatch(after, tail, textures: [outputPlanes.texture, destLuma.texture, destChroma.texture],
                          params: &outputParams, grid: (width, height))
        after.commit()
        after.waitUntilCompleted()
        withExtendedLifetime((sourceLuma, sourceChroma, inputPlanes, outputPlanes, destLuma, destChroma)) {}
        if let error = after.error { throw error }
        let gpu = (max(0, before.gpuEndTime - before.gpuStartTime) + max(0, after.gpuEndTime - after.gpuStartTime)) * 1000
        return FrameResult(buffer: destination, workMilliseconds: (CFAbsoluteTimeGetCurrent() - workStarted) * 1000,
                           gpuMilliseconds: gpu, neuralEngineMilliseconds: prediction)
    }

    private func encodeAndWait(
        source: CVPixelBuffer, width: Int, height: Int, format: PlaneFormat,
        inputParams: inout ConversionParams, outputParams: inout ConversionParams,
        resources: Resources, pipeline: MPVHookPipeline, scale: Int
    ) throws -> FrameResult? {
        // Textures the cache made for earlier frames hold those frames' IOSurfaces until it is
        // flushed; without this, decoded and upscaled frames outlive their use.
        defer { CVMetalTextureCacheFlush(resources.textureCache, 0) }
        let device = resources.device
        let target = (width: width * scale, height: height * scale)
        intermediates.prepare(forInput: (width, height))
        let rgb = try intermediates.texture(device: device, key: "input-rgb", width: width, height: height, avoiding: [])

        // Waiting for a free output frame is back-pressure from the screen, not upscaling cost, so it
        // is outside what the watchdog times. Timed together, the waits that come with a full queue
        // (after a start, a seek or a pause) read as a slow GPU: on device the upscaler stepped down
        // and switched itself off over and over until the queue settled.
        guard let destination = makeOutputBuffer(pixelFormat: Self.eightBitFormat(fullRange: format.fullRange),
                                                 width: target.width, height: target.height) else {
            return nil
        }
        let workStarted = CFAbsoluteTimeGetCurrent()
        guard let sourceLuma = Self.planeTexture(source, plane: 0, format: format.luma, cache: resources.textureCache),
              let sourceChroma = Self.planeTexture(source, plane: 1, format: format.chroma, cache: resources.textureCache),
              let destLuma = Self.planeTexture(destination, plane: 0, format: .r8Unorm, cache: resources.textureCache),
              let destChroma = Self.planeTexture(destination, plane: 1, format: .rg8Unorm, cache: resources.textureCache),
              let commandBuffer = resources.queue.makeCommandBuffer() else {
            return nil
        }

        try Self.dispatch(commandBuffer, resources.yuvToRGB, textures: [sourceLuma.texture, sourceChroma.texture, rgb],
                          params: &inputParams, grid: (width, height))
        let outputSize: (Int, Int)
        if scale == 2, fusesTail, pipeline.hasFusableTail, let fused = resources.fusedTail {
            // Everything but the last two passes, then one kernel that does the depth-to-space, the
            // de-ring clamp and the 8-bit conversion straight from the source-sized results: no 4K
            // half-float image is written or read at all.
            let textures = try pipeline.encodeTextures(
                commandBuffer: commandBuffer, input: rgb, output: target, sampler: resources.sampler,
                pool: &intermediates, device: device, skipping: MPVHookPipeline.isFusedTailPass)
            guard let restored = textures["MAIN"], let residual = textures["conv2d_last_tf"],
                  let stats = textures["STATSMAX"],
                  residual.width * 2 == target.width, residual.height * 2 == target.height else {
                throw MPVHookPipeline.EncodeError.unboundTexture("fused tail inputs")
            }
            try Self.dispatch(commandBuffer, fused,
                              textures: [restored, residual, stats, destLuma.texture, destChroma.texture],
                              params: &outputParams, grid: (residual.width, residual.height))
            outputSize = (residual.width * 2, residual.height * 2)
        } else {
            let upscaled = try pipeline.encode(
                commandBuffer: commandBuffer, input: rgb, output: target, sampler: resources.sampler,
                pool: &intermediates, device: device)
            try Self.dispatch(commandBuffer, resources.rgbToLuma, textures: [upscaled, destLuma.texture],
                              params: &outputParams, grid: (destLuma.texture.width, destLuma.texture.height))
            try Self.dispatch(commandBuffer, resources.rgbToChroma, textures: [upscaled, destChroma.texture],
                              params: &outputParams, grid: (destChroma.texture.width, destChroma.texture.height))
            outputSize = (upscaled.width, upscaled.height)
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        // The CVMetalTextures must outlive the GPU's use of their IOSurfaces.
        withExtendedLifetime((sourceLuma, sourceChroma, destLuma, destChroma)) {}
        if let error = commandBuffer.error { throw error }
        // A pipeline whose output is not exactly 2× would be enqueued at the wrong size.
        guard outputSize.0 == target.width, outputSize.1 == target.height else { return nil }
        let gpu = max(0, commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000
        return FrameResult(buffer: destination, workMilliseconds: (CFAbsoluteTimeGetCurrent() - workStarted) * 1000,
                           gpuMilliseconds: gpu)
    }

    private static func dispatch(
        _ commandBuffer: MTLCommandBuffer, _ pipeline: MTLComputePipelineState, textures: [MTLTexture],
        params: inout ConversionParams, grid: (Int, Int)
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PreparationError.missingFunction("compute encoder")
        }
        encoder.setComputePipelineState(pipeline)
        for (index, texture) in textures.enumerated() { encoder.setTexture(texture, index: index) }
        encoder.setBytes(&params, length: MemoryLayout<ConversionParams>.stride, index: 0)
        MPVHookPipeline.dispatchCovering(encoder: encoder, pipeline: pipeline, width: grid.0, height: grid.1)
        encoder.endEncoding()
    }

    private struct PlaneTexture {
        let cvTexture: CVMetalTexture
        let texture: MTLTexture
    }

    private static func planeTexture(_ buffer: CVPixelBuffer, plane: Int, format: MTLPixelFormat,
                                     cache: CVMetalTextureCache) -> PlaneTexture? {
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, buffer, nil, format,
            CVPixelBufferGetWidthOfPlane(buffer, plane), CVPixelBufferGetHeightOfPlane(buffer, plane),
            plane, &cvTexture)
        guard status == kCVReturnSuccess, let cvTexture,
              let texture = CVMetalTextureGetTexture(cvTexture) else { return nil }
        return PlaneTexture(cvTexture: cvTexture, texture: texture)
    }

    /// A single-plane buffer as one texture (the Neural Engine's tensors).
    private static func wholeTexture(_ buffer: CVPixelBuffer, format: MTLPixelFormat,
                                     cache: CVMetalTextureCache) -> PlaneTexture? {
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, buffer, nil, format,
            CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer), 0, &cvTexture)
        guard status == kCVReturnSuccess, let cvTexture,
              let texture = CVMetalTextureGetTexture(cvTexture) else { return nil }
        return PlaneTexture(cvTexture: cvTexture, texture: texture)
    }

    private struct OutputKey: Equatable {
        let pixelFormat: OSType
        let width: Int
        let height: Int
    }

    private func makeOutputBuffer(pixelFormat: OSType, width: Int, height: Int) -> CVPixelBuffer? {
        let key = OutputKey(pixelFormat: pixelFormat, width: width, height: height)
        if outputPool?.key != key {
            let attributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
            let poolAttributes: [String: Any] = [kCVPixelBufferPoolMinimumBufferCountKey as String: 4]
            var pool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttributes as CFDictionary,
                                    attributes as CFDictionary, &pool)
            guard let pool else { return nil }
            outputPool = (key, pool)
        }
        guard let pool = outputPool?.pool else { return nil }
        let aux = [kCVPixelBufferPoolAllocationThresholdKey as String: Self.maximumFramesInFlight] as CFDictionary
        let deadline = CFAbsoluteTimeGetCurrent() + Self.outputWaitSeconds
        while true {
            var buffer: CVPixelBuffer?
            let status = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(kCFAllocatorDefault, pool, aux, &buffer)
            if status == kCVReturnSuccess { return buffer }
            // At the cap: the screen releases a frame every frame interval, so a short wait finds one.
            // Past the deadline (paused with a full queue, a stall) the original is presented instead.
            guard status == kCVReturnWouldExceedAllocationThreshold,
                  CFAbsoluteTimeGetCurrent() < deadline else { return nil }
            usleep(4_000)
        }
    }

    /// Colour tags and pixel aspect carry over; a clean aperture is in source pixels, so it doubles.
    static func copyAttachments(from source: CVPixelBuffer, to destination: CVPixelBuffer, scale: Int = 2) {
        CVBufferPropagateAttachments(source, destination)
        if let aperture = CVBufferCopyAttachment(source, kCVImageBufferCleanApertureKey, nil) as? [String: Any] {
            var scaled = aperture
            for key in [kCVImageBufferCleanApertureWidthKey, kCVImageBufferCleanApertureHeightKey,
                        kCVImageBufferCleanApertureHorizontalOffsetKey, kCVImageBufferCleanApertureVerticalOffsetKey] {
                if let value = aperture[key as String] as? NSNumber {
                    scaled[key as String] = NSNumber(value: value.doubleValue * Double(scale))
                }
            }
            CVBufferSetAttachment(destination, kCVImageBufferCleanApertureKey, scaled as CFDictionary, .shouldPropagate)
        }
        CVBufferSetAttachment(destination, upscaleFactorAttachmentKey, NSNumber(value: scale), .shouldNotPropagate)
    }

    // MARK: - Bookkeeping

    private func noteBypass(_ reason: String, size: CGSize) {
        stateLock.lock()
        let changed = bypassReason != reason
        bypassReason = reason
        lastInputSize = size
        lastOutputSize = nil
        stateLock.unlock()
        if changed { Anime4KLog.emit("[Upscaler] passing frames through: \(reason)") }
    }

    private func fail(_ reason: String) {
        stateLock.lock()
        failure = reason
        stateLock.unlock()
        releaseIntermediates()
        Anime4KLog.emit("[Upscaler] off for this session: \(reason)")
    }

    private func noteFrame(milliseconds: Double, gpuMilliseconds: Double, neuralEngineMilliseconds: Double?,
                           input: CGSize, output: CGSize) {
        let backend = neuralEngineMilliseconds == nil ? "GPU" : "Neural Engine"
        stateLock.lock()
        let backendChanged = lastBackend != nil && lastBackend != backend
        lastBackend = backend
        bypassReason = nil
        processedAny = true
        lastInputSize = input
        lastOutputSize = output
        framesTimed += 1
        let budget = (frameIntervalSeconds ?? 1 / 24) * 1000 * Self.budgetShare
        var verdict: String?
        if framesTimed > Self.warmupFrames {
            timing.record(wall: milliseconds, gpu: gpuMilliseconds, neuralEngine: neuralEngineMilliseconds)
        }
        // Spikes over the budget are what the watchdog has to see past; log them so a device log
        // shows when they happen and whether the GPU's own time is what rose.
        var slowLine: String?
        if milliseconds > budget {
            slowFramesSinceLog += 1
            if Date().timeIntervalSince(lastSlowLog) >= 0.5 {
                slowLine = String(format: "[Upscaler] slow frame: %.1f ms (GPU %.1f", milliseconds, gpuMilliseconds)
                    + (neuralEngineMilliseconds.map { String(format: ", Neural Engine %.1f", $0) } ?? "")
                    + String(format: ") against %.1f ms", budget)
                    + (slowFramesSinceLog > 1 ? " (\(slowFramesSinceLog) slow frames since the last line)" : "")
                    + String(format: "; median %.1f ms", timing.medianWall ?? 0)
                slowFramesSinceLog = 0
                lastSlowLog = Date()
            }
        }
        if timing.isSustainedlyOver(budget: budget), let average = timing.medianWall {
            let modePresets = Self.presets(for: mode)
            if presetIndex + 1 < modePresets.count {
                presetIndex += 1
                framesTimed = 0
                timing = UpscaleTimingWindow()
                verdict = "stepping down to \(modePresets[presetIndex].name): "
                    + String(format: "%.1f ms per frame against a %.1f ms budget", average, budget)
            } else {
                failure = String(format: "too slow: %.1f ms per frame (GPU %.1f) against a %.1f ms budget",
                                 average, timing.medianGPU ?? 0, budget)
                    + (timing.medianNeuralEngine.map { String(format: " (Neural Engine %.1f)", $0) } ?? "")
                verdict = failure
            }
        }
        let first = framesTimed == 1 && presetIndex == 0
        stateLock.unlock()
        if let slowLine { Anime4KLog.emit(slowLine) }
        if first {
            Anime4KLog.emit("[Upscaler] first frame \(Int(input.width))x\(Int(input.height)) → "
                           + "\(Int(output.width))x\(Int(output.height)) in \(String(format: "%.1f", milliseconds)) ms"
                           + " on the \(backend)")
        }
        if backendChanged {
            Anime4KLog.emit("[Upscaler] now running on the \(backend)")
        }
        if let verdict {
            Anime4KLog.emit("[Upscaler] \(verdict)")
            if failure != nil { releaseIntermediates() }
        }
    }

    // MARK: - Conversion kernels

    static let conversionSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct ConversionParams {
        float yOffset; float yScale; float cOffset; float cScale;
        float kr; float kb; float dither; float padding;
    };

    constant float bayer4[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };

    static float ditherOffset(uint2 gid, float step) {
        return (bayer4[(gid.y & 3) * 4 + (gid.x & 3)] / 16.0 - 0.46875) * step;
    }

    kernel void upscaler_yuv_to_rgb(texture2d<float, access::read> lumaTex [[texture(0)]],
                                    texture2d<float, access::sample> chromaTex [[texture(1)]],
                                    texture2d<float, access::write> rgbTex [[texture(2)]],
                                    constant ConversionParams& p [[buffer(0)]],
                                    uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= rgbTex.get_width() || gid.y >= rgbTex.get_height()) { return; }
        constexpr sampler chromaSampler(coord::normalized, address::clamp_to_edge, filter::linear);
        float2 pos = (float2(gid) + 0.5) / float2(rgbTex.get_width(), rgbTex.get_height());
        float y = (lumaTex.read(gid).r - p.yOffset) / p.yScale;
        float2 c = (chromaTex.sample(chromaSampler, pos).rg - p.cOffset) / p.cScale;
        float kg = 1.0 - p.kr - p.kb;
        float r = y + 2.0 * (1.0 - p.kr) * c.y;
        float b = y + 2.0 * (1.0 - p.kb) * c.x;
        float g = (y - p.kr * r - p.kb * b) / kg;
        rgbTex.write(float4(saturate(float3(r, g, b)), 1.0), gid);
    }

    kernel void upscaler_rgb_to_luma(texture2d<float, access::read> rgbTex [[texture(0)]],
                                     texture2d<float, access::write> lumaTex [[texture(1)]],
                                     constant ConversionParams& p [[buffer(0)]],
                                     uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= lumaTex.get_width() || gid.y >= lumaTex.get_height()) { return; }
        float3 rgb = saturate(rgbTex.read(gid).rgb);
        float y = dot(rgb, float3(p.kr, 1.0 - p.kr - p.kb, p.kb));
        float code = saturate(y * p.yScale + p.yOffset + ditherOffset(gid, p.dither));
        lumaTex.write(float4(code), gid);
    }

    // Anime4K's last two passes and the conversion back to 8-bit 4:2:0, in one: one thread per
    // source texel, which is one 2×2 block of the output and exactly one chroma sample.
    //  - Depth-to-space (Upscale_CNN_x2): the output pixel at sub-position (dx, dy) of source texel t is
    //    the restored image, bilinearly sampled at the output pixel's centre, plus component
    //    dy * 2 + dx of the residual at t, added to every channel.
    //  - De-ring clamp (Clamp_Highlights, PREKERNEL): luma no brighter than STATSMAX, the source's
    //    local maximum, bilinearly sampled at the same centre; the excess comes off every channel.
    kernel void upscaler_fused_tail(texture2d<float, access::sample> restoredTex [[texture(0)]],
                                    texture2d<float, access::read> residualTex [[texture(1)]],
                                    texture2d<float, access::sample> statsTex [[texture(2)]],
                                    texture2d<float, access::write> lumaTex [[texture(3)]],
                                    texture2d<float, access::write> chromaTex [[texture(4)]],
                                    constant ConversionParams& p [[buffer(0)]],
                                    uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= residualTex.get_width() || gid.y >= residualTex.get_height()) { return; }
        constexpr sampler linearSampler(coord::normalized, address::clamp_to_edge, filter::linear);
        float2 outSize = float2(lumaTex.get_width(), lumaTex.get_height());
        float4 residual = residualTex.read(gid);
        float3 kY = float3(p.kr, 1.0 - p.kr - p.kb, p.kb);
        float3 sum = float3(0.0);
        for (uint dy = 0; dy < 2; dy++) {
            for (uint dx = 0; dx < 2; dx++) {
                uint2 o = gid * 2 + uint2(dx, dy);
                float2 pos = (float2(o) + 0.5) / outSize;
                float3 rgb = restoredTex.sample(linearSampler, pos).rgb + residual[dy * 2 + dx];
                float luma = dot(float3(0.299, 0.587, 0.114), rgb);
                float clamped = min(luma, statsTex.sample(linearSampler, pos).x);
                rgb = saturate(rgb - (luma - clamped));
                float y = dot(rgb, kY);
                lumaTex.write(float4(saturate(y * p.yScale + p.yOffset + ditherOffset(o, p.dither))), o);
                sum += rgb;
            }
        }
        float3 rgb = sum * 0.25;
        float y = dot(rgb, kY);
        float cb = (rgb.b - y) / (2.0 * (1.0 - p.kb));
        float cr = (rgb.r - y) / (2.0 * (1.0 - p.kr));
        float d = ditherOffset(gid, p.dither);
        chromaTex.write(float4(saturate(cb * p.cScale + p.cOffset + d),
                               saturate(cr * p.cScale + p.cOffset + d), 0.0, 0.0), gid);
    }

    kernel void upscaler_rgb_to_chroma(texture2d<float, access::read> rgbTex [[texture(0)]],
                                       texture2d<float, access::write> chromaTex [[texture(1)]],
                                       constant ConversionParams& p [[buffer(0)]],
                                       uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= chromaTex.get_width() || gid.y >= chromaTex.get_height()) { return; }
        uint2 limit = uint2(rgbTex.get_width() - 1, rgbTex.get_height() - 1);
        uint2 base = gid * 2;
        float3 rgb = (saturate(rgbTex.read(min(base, limit)).rgb)
                    + saturate(rgbTex.read(min(base + uint2(1, 0), limit)).rgb)
                    + saturate(rgbTex.read(min(base + uint2(0, 1), limit)).rgb)
                    + saturate(rgbTex.read(min(base + uint2(1, 1), limit)).rgb)) * 0.25;
        float y = dot(rgb, float3(p.kr, 1.0 - p.kr - p.kb, p.kb));
        float cb = (rgb.b - y) / (2.0 * (1.0 - p.kb));
        float cr = (rgb.r - y) / (2.0 * (1.0 - p.kr));
        float d = ditherOffset(gid, p.dither);
        chromaTex.write(float4(saturate(cb * p.cScale + p.cOffset + d),
                               saturate(cr * p.cScale + p.cOffset + d), 0.0, 0.0), gid);
    }

    // The Neural Engine path. Its tensors are one-component half-float textures W wide and C × H
    // tall: channel c of pixel (x, y) is at (x, c * H + y).

    // upscaler_yuv_to_rgb, written as the model's input: R, G and B planes.
    kernel void upscaler_yuv_to_planar(texture2d<float, access::read> lumaTex [[texture(0)]],
                                       texture2d<float, access::sample> chromaTex [[texture(1)]],
                                       texture2d<float, access::write> planesTex [[texture(2)]],
                                       constant ConversionParams& p [[buffer(0)]],
                                       uint2 gid [[thread_position_in_grid]]) {
        uint w = planesTex.get_width(), h = planesTex.get_height() / 3;
        if (gid.x >= w || gid.y >= h) { return; }
        constexpr sampler chromaSampler(coord::normalized, address::clamp_to_edge, filter::linear);
        float2 pos = (float2(gid) + 0.5) / float2(w, h);
        float y = (lumaTex.read(gid).r - p.yOffset) / p.yScale;
        float2 c = (chromaTex.sample(chromaSampler, pos).rg - p.cOffset) / p.cScale;
        float kg = 1.0 - p.kr - p.kb;
        float r = y + 2.0 * (1.0 - p.kr) * c.y;
        float b = y + 2.0 * (1.0 - p.kb) * c.x;
        float g = (y - p.kr * r - p.kb * b) / kg;
        float3 rgb = saturate(float3(r, g, b));
        planesTex.write(float4(rgb.r), gid);
        planesTex.write(float4(rgb.g), uint2(gid.x, gid.y + h));
        planesTex.write(float4(rgb.b), uint2(gid.x, gid.y + 2 * h));
    }

    // Bilinear with clamp-to-edge from one plane, at `s` in texel units (texel i's centre at i): the
    // texture's own linear filter, with the row clamped to the plane's first and last texel centres
    // so a sample at a plane's edge never blends in the neighbouring plane. Clamping to the centre is
    // exactly clamp-to-edge there; across the width the sampler's own edge clamp does the same.
    static float planeSample(texture2d<float, access::sample> t, sampler linear, uint plane, float2 s, int2 size) {
        float row = clamp(s.y, 0.0, float(size.y - 1)) + float(plane * uint(size.y));
        return t.sample(linear, float2((s.x + 0.5) / float(size.x), (row + 0.5) / float(t.get_height()))).r;
    }

    // upscaler_fused_tail reading the model's output: planes 0-2 the restored RGB, 3-6 the
    // residuals (conv2d_last_tf), 7 STATSMAX.
    kernel void upscaler_planar_tail(texture2d<float, access::sample> planesTex [[texture(0)]],
                                     texture2d<float, access::write> lumaTex [[texture(1)]],
                                     texture2d<float, access::write> chromaTex [[texture(2)]],
                                     constant ConversionParams& p [[buffer(0)]],
                                     uint2 gid [[thread_position_in_grid]]) {
        int2 size = int2(planesTex.get_width(), planesTex.get_height() / 8);
        int2 g = int2(gid);
        if (g.x >= size.x || g.y >= size.y) { return; }
        constexpr sampler linear(coord::normalized, address::clamp_to_edge, filter::linear);
        float3 kY = float3(p.kr, 1.0 - p.kr - p.kb, p.kb);
        float3 sum = float3(0.0);
        for (uint k = 0; k < 4; k++) {
            // Output pixel (dx, dy) of source texel g has its centre at g + (dx, dy) / 2 - 1/4.
            uint2 d = uint2(k & 1, k >> 1);
            uint2 o = gid * 2 + d;
            float2 s = float2(g) + float2(d) * 0.5 - 0.25;
            float residual = planesTex.read(uint2(g.x, g.y + int(3 + k) * size.y)).r;
            float3 rgb = float3(planeSample(planesTex, linear, 0, s, size), planeSample(planesTex, linear, 1, s, size),
                                planeSample(planesTex, linear, 2, s, size)) + residual;
            float luma = dot(float3(0.299, 0.587, 0.114), rgb);
            float clamped = min(luma, planeSample(planesTex, linear, 7, s, size));
            rgb = saturate(rgb - (luma - clamped));
            float y = dot(rgb, kY);
            lumaTex.write(float4(saturate(y * p.yScale + p.yOffset + ditherOffset(o, p.dither))), o);
            sum += rgb;
        }
        float3 rgb = sum * 0.25;
        float y = dot(rgb, kY);
        float cb = (rgb.b - y) / (2.0 * (1.0 - p.kb));
        float cr = (rgb.r - y) / (2.0 * (1.0 - p.kr));
        float d = ditherOffset(gid, p.dither);
        chromaTex.write(float4(saturate(cb * p.cScale + p.cOffset + d),
                               saturate(cr * p.cScale + p.cOffset + d), 0.0, 0.0), gid);
    }
    """
}

// MARK: - Watchdog timing

/// The watchdog's view of recent frames: medians over a rolling window, and a verdict only for
/// sustained overload. On an Apple TV 4K (3rd gen) S ran at 27–30 ms with brief spikes; a running
/// average let a spike carry it past the budget and switched it off while playback was smooth.
struct UpscaleTimingWindow {
    /// Frames in the window: two seconds at 24 fps.
    static let capacity = 48
    /// Further frames the median must stay over budget, once the window is full, before it counts.
    static let sustainFrames = 24
    /// Samples before a median is shown at all.
    static let minimumForMedian = 6

    private(set) var wall: [Double] = []
    private(set) var gpu: [Double] = []
    /// Neural Engine prediction times, for the frames whose networks ran there.
    private(set) var neuralEngine: [Double] = []
    private var overStreak = 0

    mutating func record(wall: Double, gpu: Double, neuralEngine: Double? = nil) {
        self.wall.append(wall)
        self.gpu.append(gpu)
        if let neuralEngine { self.neuralEngine.append(neuralEngine) } else { self.neuralEngine.removeAll() }
        if self.wall.count > Self.capacity { self.wall.removeFirst(self.wall.count - Self.capacity) }
        if self.gpu.count > Self.capacity { self.gpu.removeFirst(self.gpu.count - Self.capacity) }
        if self.neuralEngine.count > Self.capacity {
            self.neuralEngine.removeFirst(self.neuralEngine.count - Self.capacity)
        }
    }

    var medianWall: Double? { Self.median(wall) }
    var medianGPU: Double? { Self.median(gpu) }
    var medianNeuralEngine: Double? { Self.median(neuralEngine) }

    /// True once the window is full and its median has been over `budget` for `sustainFrames`
    /// consecutive frames. Called once per recorded frame.
    mutating func isSustainedlyOver(budget: Double) -> Bool {
        guard wall.count >= Self.capacity, let median = medianWall else { return false }
        overStreak = median > budget ? overStreak + 1 : 0
        return overStreak >= Self.sustainFrames
    }

    static func median(_ values: [Double]) -> Double? {
        guard values.count >= minimumForMedian else { return nil }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        return sorted.count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
    }
}

// MARK: - Pass execution

/// Intermediate textures, kept across frames and keyed by name, size and a variant that lets a pass
/// write a name it also reads (the de-ring statistics pass binds and saves `STATSMAX`).
struct TexturePool {
    private var textures: [String: MTLTexture] = [:]
    private var inputSize: (Int, Int)?

    /// What the pool holds, by the textures' own allocated size.
    var allocatedBytes: Int { textures.values.reduce(0) { $0 + $1.allocatedSize } }

    /// Whether frames of this size are already running through the pool.
    func isPrepared(forInput size: (Int, Int)) -> Bool {
        guard let inputSize, !textures.isEmpty else { return false }
        return inputSize == size
    }

    /// A new source size invalidates every intermediate.
    mutating func prepare(forInput size: (Int, Int)) {
        if let inputSize, inputSize == size { return }
        textures.removeAll()
        inputSize = size
    }

    struct AllocationFailed: Error, CustomStringConvertible {
        let key: String
        var description: String { "could not allocate the \(key) texture" }
    }

    mutating func texture(device: MTLDevice, key: String, width: Int, height: Int,
                          avoiding bound: [MTLTexture]) throws -> MTLTexture {
        var variant = 0
        while true {
            let fullKey = "\(key)|\(width)x\(height)|\(variant)"
            if let existing = textures[fullKey] {
                if bound.contains(where: { $0 === existing }) {
                    variant += 1
                    continue
                }
                return existing
            }
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
            descriptor.usage = [.shaderRead, .shaderWrite]
            descriptor.storageMode = .private
            guard let made = device.makeTexture(descriptor: descriptor) else {
                throw AllocationFailed(key: key)
            }
            textures[fullKey] = made
            return made
        }
    }
}

/// A chain of mpv hook passes compiled for Metal, run in mpv's order: every `MAIN` pass in file
/// order, then every `PREKERNEL` pass.
final class MPVHookPipeline {
    struct Pass {
        let shader: MPVHookShader
        let state: MTLComputePipelineState
    }

    let passes: [Pass]

    init(device: MTLDevice, shaders: [MPVHookShader], halfPrecision: Bool = true) throws {
        let ordered = shaders.filter { $0.hook == "MAIN" } + shaders.filter { $0.hook == "PREKERNEL" }
        passes = try ordered.map { shader in
            // Half precision where the pass's GLSL allows it; the de-ring clamp mixes scalar widths
            // in a way Metal calls ambiguous in half, and it is cheap, so it compiles in full.
            let library: MTLLibrary
            if halfPrecision, let halfLibrary = try? device.makeLibrary(
                source: shader.metalSource(halfPrecision: true), options: nil) {
                library = halfLibrary
            } else {
                library = try device.makeLibrary(source: shader.metalSource(halfPrecision: false), options: nil)
            }
            guard let function = library.makeFunction(name: MPVHookShader.kernelName) else {
                throw Anime4KUpscaler.PreparationError.missingFunction(shader.description)
            }
            return Pass(shader: shader, state: try device.makeComputePipelineState(function: function))
        }
    }

    enum EncodeError: Error {
        case unboundTexture(String)
        case badSize(String)
    }

    /// The two passes `upscaler_fused_tail` replaces: Anime4K's depth-to-space and its de-ring clamp.
    static func isFusedTailPass(_ shader: MPVHookShader) -> Bool {
        shader.description.contains("Depth-to-Space") || shader.description.contains("De-Ring-Clamp")
    }

    /// Whether this chain ends the way the fused kernel assumes: the depth-to-space is the last
    /// `MAIN` pass (reading `MAIN` and `conv2d_last_tf`), and the de-ring clamp is the only
    /// `PREKERNEL` pass (reading `STATSMAX`).
    var hasFusableTail: Bool {
        let main = passes.filter { $0.shader.hook == "MAIN" }
        let prekernel = passes.filter { $0.shader.hook == "PREKERNEL" }
        guard let last = main.last?.shader, last.description.contains("Depth-to-Space"),
              last.binds == ["MAIN", "conv2d_last_tf"],
              prekernel.count == 1, let clamp = prekernel.first?.shader,
              clamp.description.contains("De-Ring-Clamp"), clamp.binds == ["HOOKED", "STATSMAX"]
        else { return false }
        return true
    }

    /// Encode every pass into `commandBuffer`; returns the final `MAIN`.
    func encode(commandBuffer: MTLCommandBuffer, input: MTLTexture, output: (width: Int, height: Int),
                sampler: MTLSamplerState, pool: inout TexturePool, device: MTLDevice) throws -> MTLTexture {
        let textures = try encodeTextures(commandBuffer: commandBuffer, input: input, output: output,
                                          sampler: sampler, pool: &pool, device: device, skipping: { _ in false })
        guard let result = textures["MAIN"] else { throw EncodeError.unboundTexture("MAIN") }
        return result
    }

    /// Encode every pass `skipping` does not exclude; returns every texture by the name it was saved under.
    func encodeTextures(commandBuffer: MTLCommandBuffer, input: MTLTexture, output: (width: Int, height: Int),
                        sampler: MTLSamplerState, pool: inout TexturePool, device: MTLDevice,
                        skipping: (MPVHookShader) -> Bool) throws -> [String: MTLTexture] {
        var textures: [String: MTLTexture] = ["MAIN": input, "NATIVE": input]
        for pass in passes where !skipping(pass.shader) {
            let shader = pass.shader
            guard let hooked = textures["MAIN"] else { throw EncodeError.unboundTexture("MAIN") }
            var sizes: [String: (Double, Double)] = [
                "HOOKED": (Double(hooked.width), Double(hooked.height)),
                "OUTPUT": (Double(output.width), Double(output.height)),
                "NATIVE": (Double(input.width), Double(input.height)),
            ]
            for (name, texture) in textures { sizes[name] = (Double(texture.width), Double(texture.height)) }

            if let when = shader.when {
                guard let value = MPVHookShader.evaluate(when, sizes: sizes) else {
                    throw EncodeError.badSize("WHEN \(when.joined(separator: " "))")
                }
                if value == 0 { continue }
            }
            let width = try shader.width.map { tokens -> Int in
                guard let value = MPVHookShader.evaluate(tokens, sizes: sizes) else {
                    throw EncodeError.badSize("WIDTH \(tokens.joined(separator: " "))")
                }
                return Int(value.rounded())
            } ?? hooked.width
            let height = try shader.height.map { tokens -> Int in
                guard let value = MPVHookShader.evaluate(tokens, sizes: sizes) else {
                    throw EncodeError.badSize("HEIGHT \(tokens.joined(separator: " "))")
                }
                return Int(value.rounded())
            } ?? hooked.height

            let bound = try shader.binds.map { name -> MTLTexture in
                if name == "HOOKED" { return hooked }
                guard let texture = textures[name] else { throw EncodeError.unboundTexture(name) }
                return texture
            }
            let destination = try pool.texture(device: device, key: shader.destination,
                                           width: width, height: height, avoiding: bound)

            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw Anime4KUpscaler.PreparationError.missingFunction("compute encoder")
            }
            encoder.setComputePipelineState(pass.state)
            for (index, texture) in bound.enumerated() { encoder.setTexture(texture, index: index) }
            encoder.setTexture(destination, index: bound.count)
            encoder.setSamplerState(sampler, index: 0)
            Self.dispatchCovering(encoder: encoder, pipeline: pass.state, width: width, height: height)
            encoder.endEncoding()
            textures[shader.destination] = destination
        }
        return textures
    }

    /// Whole threadgroups over the grid; every kernel bounds-checks, so the overhang writes nothing.
    static func dispatchCovering(encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState,
                                 width: Int, height: Int) {
        let w = pipeline.threadExecutionWidth
        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
        encoder.dispatchThreadgroups(
            MTLSize(width: (width + w - 1) / w, height: (height + h - 1) / h, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
    }
}
