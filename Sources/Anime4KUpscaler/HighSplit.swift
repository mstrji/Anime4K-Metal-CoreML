import CoreML
import CoreVideo
import Foundation
import Metal

/// `.upscaleEnhanceHigh` (Anime4K A (M)) on a 1920x1080 frame, **split between the Neural Engine and the
/// GPU**. On an Apple TV 4K (3rd gen) A (M) takes ~45-50 ms a frame whole on the Neural Engine, against a
/// 39.6 ms budget at 23.976 fps; the GPU runs the same networks as Metal shaders, so the two can share a
/// frame. Split, that box settles at row 720 with each side at ~25-33 ms and ~30-38 ms a frame.
///
/// The Neural Engine runs the model on the top band (rows `0 ..< row + halo`) and the GPU the shaders on
/// the bottom band (`row - halo ..< 1080`) at the same time. The model writes its eight planes (restored
/// RGB, four residuals, de-ring statistics) straight into the frame's output tensor; the GPU's textures
/// carry exactly those quantities. The rows below the model's band take the fused tail inside the GPU's
/// half, straight from its textures (`upscaler_fused_tail_rows`), while the Neural Engine still runs.
/// After both finish, `upscaler_textures_to_planes` writes the GPU's rows just below the split into the
/// tensor (the model writes its halo rows there too, computed with less context, and they must be
/// overwritten), and the planar tail runs on the rows above.
/// The halo (24 rows) covers both networks' 14-pixel reach and the 5x5 de-ring statistics. Every 2 s the
/// split moves to the row where both sides' measured cost per row says they would take the same time.
/// Bands must be sizes the model was built for (`Scripts/anime4k_to_coreml.py`, SPLIT_ROWS).
extension Anime4KUpscaler {
    /// The rows an Apple TV 4K (3rd gen) moves between (600 to 720); each is a shape the model compiles
    /// on first load.
    static let highSplitRows = [540, 600, 660, 720]
    static let highSplitHalo = 24

    /// Where to split this frame, or nil to run it whole: A (M) only, 1920x1080 only, with the fused
    /// tail's shaders and every kernel the split needs.
    func highSplitRow(preset: Preset, width: Int, height: Int, model: NeuralEngineUpscaler,
                         pipeline: MPVHookPipeline, resources: Resources) -> Int? {
        guard preset == Self.mediumPreset, width == 1920, height == 1080, fusesTail, pipeline.hasFusableTail,
              resources.texturesToPlanes != nil, resources.yuvToRGBRows != nil, resources.fusedTail != nil,
              resources.fusedTailRows != nil
        else { return nil }
        stateLock.lock()
        let enabled = splitsHighStorage
        let row = Self.highSplitRows[highSplitIndex]
        stateLock.unlock()
        guard enabled, model.supports(width: width, height: row + Self.highSplitHalo) else { return nil }
        return row
    }

    // swiftlint:disable:next function_parameter_count
    func encodeHighSplit(
        source: CVPixelBuffer, width: Int, height: Int, format: PlaneFormat,
        inputParams: inout ConversionParams, outputParams: inout ConversionParams,
        resources: Resources, model: NeuralEngineUpscaler, pipeline: MPVHookPipeline, row: Int
    ) throws -> FrameResult? {
        defer { CVMetalTextureCacheFlush(resources.textureCache, 0) }
        guard let toPlanar = resources.yuvToPlanar, let toRGBRows = resources.yuvToRGBRows else { return nil }
        let halo = Self.highSplitHalo
        let topRows = row + halo, bandStart = row - halo, bandRows = height - bandStart
        let device = resources.device

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
        if splitTextures?.bandRows != bandRows || splitTextures?.band.width != width {
            func texture(_ h: Int) throws -> MTLTexture {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: h,
                                                                 mipmapped: false)
                d.usage = [.shaderRead, .shaderWrite]
                d.storageMode = .private
                guard let made = device.makeTexture(descriptor: d) else {
                    throw TexturePool.AllocationFailed(key: "split RGB")
                }
                return made
            }
            splitTextures = (try texture(bandRows), bandRows)
            splitPool = TexturePool()
        }
        guard let rgb = splitTextures else { return nil }
        splitPool.prepare(forInput: (width, bandRows))

        guard let destination = makeOutputBuffer(pixelFormat: Self.eightBitFormat(fullRange: format.fullRange),
                                                 width: width * 2, height: height * 2) else { return nil }
        let workStarted = CFAbsoluteTimeGetCurrent()
        guard let sourceLuma = Self.planeTexture(source, plane: 0, format: format.luma, cache: resources.textureCache),
              let sourceChroma = Self.planeTexture(source, plane: 1, format: format.chroma, cache: resources.textureCache),
              let inputPlanes = Self.wholeTexture(tensors.input, format: .r16Float, cache: resources.textureCache),
              let outputPlanes = Self.wholeTexture(tensors.output, format: .r16Float, cache: resources.textureCache),
              let destLuma = Self.planeTexture(destination, plane: 0, format: .r8Unorm, cache: resources.textureCache),
              let destChroma = Self.planeTexture(destination, plane: 1, format: .rg8Unorm, cache: resources.textureCache),
              let before = resources.queue.makeCommandBuffer() else { return nil }

        // All three command buffers are encoded before any is committed, so the GPU goes straight from
        // one step to the next without waiting for the CPU, and the CPU only waits where it must.
        // 1. The model's band of the frame as its planes: the one step the Neural Engine waits for. Rows
        //    below the band are never read, so they are not converted.
        try Self.dispatch(before, toPlanar, textures: [sourceLuma.texture, sourceChroma.texture, inputPlanes.texture],
                          params: &inputParams, grid: (width, topRows))

        // 2. The GPU's half: its own rows of the frame as RGB, then the chain's shaders on them. Committed
        //    right behind step 1, so it runs while the Neural Engine runs the model on the top band.
        guard let gpuBand = resources.queue.makeCommandBuffer(),
              let convert = gpuBand.makeComputeCommandEncoder() else { return nil }
        var rowOffset = UInt32(bandStart)
        convert.setComputePipelineState(toRGBRows)
        convert.setTexture(sourceLuma.texture, index: 0)
        convert.setTexture(sourceChroma.texture, index: 1)
        convert.setTexture(rgb.band, index: 2)
        convert.setBytes(&inputParams, length: MemoryLayout<ConversionParams>.stride, index: 0)
        convert.setBytes(&rowOffset, length: MemoryLayout<UInt32>.stride, index: 1)
        MPVHookPipeline.dispatchCovering(encoder: convert, pipeline: toRGBRows, width: width, height: bandRows)
        convert.endEncoding()
        let bandTextures = try pipeline.encodeTextures(
            commandBuffer: gpuBand, input: rgb.band, output: (width * 2, bandRows * 2), sampler: resources.sampler,
            pool: &splitPool, device: device, skipping: MPVHookPipeline.isFusedTailPass)
        guard let restored = bandTextures["MAIN"], let residual = bandTextures["conv2d_last_tf"],
              let stats = bandTextures["STATSMAX"],
              restored.width == width, residual.width == width, stats.width == width,
              restored.height == bandRows, residual.height == bandRows, stats.height == bandRows else {
            throw MPVHookPipeline.EncodeError.unboundTexture("split band textures")
        }
        // The tail on the GPU's own rows, inside its half, so it runs while the Neural Engine is still
        // working instead of after both. It starts two rows below the last row the model writes
        // (`topRows`): the tail reads one row either side, and nothing it reads can be the model's.
        let tailStart = topRows + 2
        guard let tailRows = resources.fusedTailRows, let bandTail = gpuBand.makeComputeCommandEncoder() else {
            return nil
        }
        var tailParams = TailRowsParams(srcRow: UInt32(tailStart - bandStart), dstRow: UInt32(tailStart),
                                        rows: UInt32(height - tailStart))
        bandTail.setComputePipelineState(tailRows)
        for (index, texture) in [restored, residual, stats, destLuma.texture, destChroma.texture].enumerated() {
            bandTail.setTexture(texture, index: index)
        }
        bandTail.setBytes(&outputParams, length: MemoryLayout<ConversionParams>.stride, index: 0)
        bandTail.setBytes(&tailParams, length: MemoryLayout<TailRowsParams>.stride, index: 1)
        MPVHookPipeline.dispatchCovering(encoder: bandTail, pipeline: tailRows, width: width, height: height - tailStart)
        bandTail.endEncoding()

        // 3. The GPU's rows the rest of the tail reads into the planes (after the model's, which they
        //    overwrite), then the tail on the rows above `tailStart`. One pass reading both halves
        //    directly was tried: no faster on a Mac (27.2 against 26.0-26.9 ms), its hand-made bilinear
        //    costing what the copy saved.
        guard let after = resources.queue.makeCommandBuffer(), let encoder = after.makeComputeCommandEncoder() else {
            return nil
        }
        guard let toPlanes = resources.texturesToPlanes, let planarTail = resources.planarTail else { return nil }
        let planesEnd = min(height, tailStart + 2)
        var params = PlanesParams(planeRows: UInt32(height), srcRow: UInt32(halo), dstRow: UInt32(row),
                                  rows: UInt32(planesEnd - row))
        encoder.setComputePipelineState(toPlanes)
        for (index, texture) in [restored, residual, stats, outputPlanes.texture].enumerated() {
            encoder.setTexture(texture, index: index)
        }
        encoder.setBytes(&params, length: MemoryLayout<PlanesParams>.stride, index: 0)
        MPVHookPipeline.dispatchCovering(encoder: encoder, pipeline: toPlanes, width: width, height: planesEnd - row)
        encoder.endEncoding()
        // Exactly `tailStart` rows: rounded up to whole threadgroups it would also draw rows below, from
        // planes those rows no longer have (a black band across the frame on device), over the GPU's own.
        guard let tailEncoder = after.makeComputeCommandEncoder() else { return nil }
        tailEncoder.setComputePipelineState(planarTail)
        for (index, texture) in [outputPlanes.texture, destLuma.texture, destChroma.texture].enumerated() {
            tailEncoder.setTexture(texture, index: index)
        }
        tailEncoder.setBytes(&outputParams, length: MemoryLayout<ConversionParams>.stride, index: 0)
        let tailWidth = planarTail.threadExecutionWidth
        tailEncoder.dispatchThreads(
            MTLSize(width: width, height: tailStart, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tailWidth,
                                           height: max(1, planarTail.maxTotalThreadsPerThreadgroup / tailWidth), depth: 1))
        tailEncoder.endEncoding()

        let outcome = HighSplitOutcome()
        let gpuStarted = CFAbsoluteTimeGetCurrent()
        gpuBand.addCompletedHandler { _ in outcome.setGPU((CFAbsoluteTimeGetCurrent() - gpuStarted) * 1000) }
        before.commit()
        gpuBand.commit()
        before.waitUntilCompleted()
        if let error = before.error {
            gpuBand.waitUntilCompleted()
            throw error
        }
        let prepare = (CFAbsoluteTimeGetCurrent() - workStarted) * 1000
        let predictionStarted = CFAbsoluteTimeGetCurrent()
        do {
            try predictTopBand(model: model, input: tensors.input, output: tensors.output,
                               width: width, height: height, rows: topRows)
        } catch {
            gpuBand.waitUntilCompleted()
            throw NeuralEnginePredictionFailed(underlying: error)
        }
        let prediction = (CFAbsoluteTimeGetCurrent() - predictionStarted) * 1000
        // The tail goes behind the GPU half on the same queue, which orders it after that half's writes.
        let tailCommitted = CFAbsoluteTimeGetCurrent()
        after.commit()
        after.waitUntilCompleted()
        withExtendedLifetime((sourceLuma, sourceChroma, inputPlanes, outputPlanes, destLuma, destChroma)) {}
        if let error = gpuBand.error { throw error }
        if let error = after.error { throw error }
        let gpuBandMilliseconds = outcome.gpu ?? prediction
        // The tail's own share: from the later of the two halves to the end.
        let tail = (CFAbsoluteTimeGetCurrent() - max(tailCommitted, gpuStarted + gpuBandMilliseconds / 1000)) * 1000

        stateLock.lock()
        // Every 48 frames (2 s), each side's median time per row decides where the split should sit: the
        // row where both halves would take the same time, snapped to the rows the model was built for.
        // It moves there only when that row is predicted at least 1.5 ms faster per frame (the slower
        // half is what a frame waits for) and at least 10 s after the last move: each move costs a slow
        // frame, since the GPU band is reallocated and the model switches shape. Three gap-based rules
        // were tried on an Apple TV 4K (3rd gen) and each either hopped between rows (gaps of 8-10% are
        // the halves' ordinary noise) or stayed put on a worse row because the gap never crossed 15%.
        let total = (CFAbsoluteTimeGetCurrent() - workStarted) * 1000
        highSplitWindow.append((prediction, gpuBandMilliseconds, total))
        var moved: String?
        if highSplitWindow.count >= 48 {
            func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
            let ne = median(highSplitWindow.map(\.0)), gpu = median(highSplitWindow.map(\.1))
            highSplitWindow.removeAll()
            let neRate = ne / Double(topRows), gpuRate = gpu / Double(bandRows)
            func predicted(_ r: Int) -> Double {
                max(neRate * Double(r + halo), gpuRate * Double(height - r + halo))
            }
            let best = Self.highSplitRows.indices.min { predicted(Self.highSplitRows[$0])
                < predicted(Self.highSplitRows[$1]) } ?? highSplitIndex
            let now = Date()
            if best != highSplitIndex, now.timeIntervalSince(highSplitArrivedAt) >= 10,
               predicted(row) - predicted(Self.highSplitRows[best]) >= 1.5 {
                moved = best < highSplitIndex ? "up" : "down"
                highSplitIndex = best
                highSplitArrivedAt = now
            }
        }
        let newRow = Self.highSplitRows[highSplitIndex]
        highSplitNote = String(format: "split at row %d: Neural Engine %d rows %.0f ms, GPU %d rows %.0f ms, "
                                  + "prepare %.1f ms, tail %.1f ms",
                                  row, row, prediction, height - row, gpuBandMilliseconds, prepare, tail)
        let log = Date().timeIntervalSince(lastHighSplitLog) >= 2
        if log { lastHighSplitLog = Date() }
        let line = highSplitNote ?? ""
        stateLock.unlock()
        // Every 2 s, so a device log shows where the split settled and what each side took.
        if log { Anime4KLog.emit(String(format: "[Upscaler] A (M) %@, frame %.0f ms", line, total)) }
        if let moved {
            Anime4KLog.emit("[Upscaler] A (M) split moves \(moved) to row \(newRow)")
        }

        let gpu = (max(0, before.gpuEndTime - before.gpuStartTime) + max(0, gpuBand.gpuEndTime - gpuBand.gpuStartTime)
                   + max(0, after.gpuEndTime - after.gpuStartTime)) * 1000
        return FrameResult(buffer: destination, workMilliseconds: (CFAbsoluteTimeGetCurrent() - workStarted) * 1000,
                           gpuMilliseconds: gpu, neuralEngineMilliseconds: prediction)
    }

    /// The model on rows `0 ..< rows` of the frame: strided views of the frame's input and output
    /// tensors (planes `height` tall), so nothing is copied unless Core ML declines the backing.
    private func predictTopBand(model: NeuralEngineUpscaler, input: CVPixelBuffer, output: CVPixelBuffer,
                                width: Int, height: Int, rows: Int) throws {
        CVPixelBufferLockBaseAddress(input, [])
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(output, [])
            CVPixelBufferUnlockBaseAddress(input, [])
        }
        guard let inBase = CVPixelBufferGetBaseAddress(input), let outBase = CVPixelBufferGetBaseAddress(output) else {
            throw NeuralEngineUpscaler.PredictionError.unexpectedOutput
        }
        let inRow = CVPixelBufferGetBytesPerRow(input) / 2, outRow = CVPixelBufferGetBytesPerRow(output) / 2
        let inChannels = NeuralEngineUpscaler.inputChannels, outChannels = NeuralEngineUpscaler.outputChannels
        let inputArray = try MLMultiArray(dataPointer: inBase, shape: [1, inChannels, rows, width] as [NSNumber],
                                          dataType: .float16,
                                          strides: [inChannels * height * inRow, height * inRow, inRow, 1] as [NSNumber])
        let outputArray = try MLMultiArray(dataPointer: outBase, shape: [1, outChannels, rows, width] as [NSNumber],
                                           dataType: .float16,
                                           strides: [outChannels * height * outRow, height * outRow, outRow, 1] as [NSNumber])
        let options = MLPredictionOptions()
        options.outputBackings = ["planes": outputArray]
        let features = try MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: inputArray)])
        let result = try model.model.prediction(from: features, options: options)
        guard let produced = result.featureValue(for: "planes")?.multiArrayValue else {
            throw NeuralEngineUpscaler.PredictionError.noOutput
        }
        if produced.dataPointer != outputArray.dataPointer {
            guard produced.dataType == .float16, produced.shape == outputArray.shape else {
                throw NeuralEngineUpscaler.PredictionError.unexpectedOutput
            }
            let s = produced.strides.map(\.intValue), d = outputArray.strides.map(\.intValue)
            let from = produced.dataPointer.assumingMemoryBound(to: UInt16.self)
            let to = outputArray.dataPointer.assumingMemoryBound(to: UInt16.self)
            for c in 0..<outChannels {
                for y in 0..<rows {
                    let a = c * s[1] + y * s[2], b = c * d[1] + y * d[2]
                    for x in 0..<width { to[b + x * d[3]] = from[a + x * s[3]] }
                }
            }
        }
    }
}

/// `upscaler_fused_tail_rows`' parameters, laid out as the kernel declares them.
struct TailRowsParams {
    var srcRow: UInt32
    var dstRow: UInt32
    var rows: UInt32
    var padding: UInt32 = 0
}

/// `upscaler_textures_to_planes`' parameters, laid out as the kernel declares them.
struct PlanesParams {
    var planeRows: UInt32
    var srcRow: UInt32
    var dstRow: UInt32
    var rows: UInt32
}

private final class HighSplitOutcome: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Double?
    func setGPU(_ ms: Double) { lock.lock(); stored = ms; lock.unlock() }
    var gpu: Double? { lock.lock(); defer { lock.unlock() }; return stored }
}
