import CoreML
import CoreVideo
import Foundation

/// Anime4K's networks on the Neural Engine, through Core ML: Mode A's restore and upscale, (S) or
/// (M), or C (S)'s single upscale-and-denoise network.
///
/// On an Apple TV 4K (3rd gen) A (S) on the GPU fits its 24 fps budget only until the GPU throttles,
/// 15–20 s in. The networks' arithmetic is the whole cost, so they move here and the GPU keeps only
/// the conversion to RGB before and the fused tail after (depth-to-space, de-ring clamp, 8-bit 4:2:0).
///
/// The models (`Anime4K_A_S`, `Anime4K_A_M`, `Anime4K_C_S` `.mlmodelc`, built from the bundled
/// shaders by `Scripts/anime4k_to_coreml.py`)
/// takes the frame as planar RGB, `[1, 3, H, W]` float16, and returns `[1, 8, H, W]` float16: the
/// restored RGB, the four sub-pixel residuals (`conv2d_last_tf`) and the de-ring statistics
/// (`STATSMAX`). Both sides are IOSurface-backed one-component half-float pixel buffers W wide and
/// C × H tall, so the GPU writes the input and reads the output in place with no copies.
final class NeuralEngineUpscaler {
    static let inputChannels = 3
    static let outputChannels = 8

    let model: MLModel
    /// Source sizes the model was built for; any other runs on the GPU.
    let sizes: Set<Size>
    private let options = MLPredictionOptions()

    struct Size: Hashable {
        let width: Int
        let height: Int
    }

    enum LoadError: Error, CustomStringConvertible {
        case noNeuralEngine
        case missingModel
        case noShapes

        var description: String {
            switch self {
            case .noNeuralEngine: return "no Neural Engine"
            case .missingModel: return "the Neural Engine model is missing from the bundle"
            case .noShapes: return "the Neural Engine model declares no sizes"
            }
        }
    }

    static func modelURL(_ name: String) -> URL? {
        Bundle.module.url(forResource: name, withExtension: "mlmodelc", subdirectory: "Anime4K")
    }

    /// Whether this device has a Neural Engine at all. Without one Core ML would run the model on the
    /// CPU, far slower than the GPU path (the Simulator, an Intel Mac).
    static var deviceHasNeuralEngine: Bool {
        MLComputeDevice.allComputeDevices.contains {
            if case .neuralEngine = $0 { return true }
            return false
        }
    }

    /// Loads the model for the Neural Engine. The first load on a device compiles it for that
    /// device's Neural Engine, a few seconds; the system caches that compilation.
    init(modelName: String) throws {
        guard Self.deviceHasNeuralEngine else { throw LoadError.noNeuralEngine }
        guard let url = Self.modelURL(modelName) else { throw LoadError.missingModel }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: url, configuration: configuration)
        let shapes = model.modelDescription.inputDescriptionsByName["rgb"]?
            .multiArrayConstraint?.shapeConstraint.enumeratedShapes ?? []
        let declared = shapes.compactMap { shape -> Size? in
            guard shape.count == 4 else { return nil }
            return Size(width: shape[3].intValue, height: shape[2].intValue)
        }
        guard !declared.isEmpty else { throw LoadError.noShapes }
        sizes = Set(declared)
    }

    func supports(width: Int, height: Int) -> Bool {
        sizes.contains(Size(width: width, height: height))
    }

    /// A pixel buffer holding a `[1, channels, height, width]` float16 tensor, plane after plane.
    static func makeTensorBuffer(width: Int, height: Int, channels: Int) -> CVPixelBuffer? {
        let attributes: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height * channels,
                                         kCVPixelFormatType_OneComponent16Half, attributes as CFDictionary, &buffer)
        return status == kCVReturnSuccess ? buffer : nil
    }

    /// Runs the networks: reads `input` (3 planes), writes `output` (8 planes) in place. Synchronous;
    /// called on the host's upscale queue, never the decode callback.
    func predict(input: CVPixelBuffer, output: CVPixelBuffer, width: Int, height: Int) throws {
        let inputArray = MLMultiArray(pixelBuffer: input, shape: [1, Self.inputChannels, height, width] as [NSNumber])
        let outputArray = MLMultiArray(pixelBuffer: output, shape: [1, Self.outputChannels, height, width] as [NSNumber])
        options.outputBackings = ["planes": outputArray]
        let features = try MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: inputArray)])
        let result = try model.prediction(from: features, options: options)
        // Core ML writes into the backing when it can; when it hands back its own array instead, the
        // result is copied so the GPU still finds it where it reads.
        guard let produced = result.featureValue(for: "planes")?.multiArrayValue else {
            throw PredictionError.noOutput
        }
        if produced.pixelBuffer != output {
            try Self.copy(produced, into: output, width: width, height: height)
        }
    }

    enum PredictionError: Error, CustomStringConvertible {
        case noOutput
        case unexpectedOutput

        var description: String {
            switch self {
            case .noOutput: return "the Neural Engine model returned nothing"
            case .unexpectedOutput: return "the Neural Engine model returned an unexpected tensor"
            }
        }
    }

    private static func copy(_ array: MLMultiArray, into buffer: CVPixelBuffer, width: Int, height: Int) throws {
        guard array.dataType == .float16, array.shape.map(\.intValue) == [1, outputChannels, height, width] else {
            throw PredictionError.unexpectedOutput
        }
        let strides = array.strides.map(\.intValue)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw PredictionError.unexpectedOutput }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        array.withUnsafeBytes { raw in
            let source = raw.bindMemory(to: UInt16.self)
            for channel in 0..<outputChannels {
                for y in 0..<height {
                    let destination = base.advanced(by: (channel * height + y) * rowBytes)
                        .assumingMemoryBound(to: UInt16.self)
                    let rowStart = channel * strides[1] + y * strides[2]
                    for x in 0..<width { destination[x] = source[rowStart + x * strides[3]] }
                }
            }
        }
    }
}
