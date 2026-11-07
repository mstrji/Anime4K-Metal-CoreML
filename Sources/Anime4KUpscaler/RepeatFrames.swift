import CoreVideo
import Foundation
import Metal

/// A frame that repeats the drawing the upscaler last upscaled is returned as that same
/// upscaled frame instead of being upscaled again. Anime is mostly drawn on twos and threes: on a 1080p
/// Crunchyroll episode (A Tale of the Secret Saint, E06, the first 4:21) 64% of frames carried no 8x8
/// block whose mean luma moved more than 2-4 codes from the frame last upscaled, and the share was the
/// same at 2, 3 and 4 codes while compression noise alone moved everything at 1. Skipping those frames
/// cuts the Neural Engine and GPU work (and the heat: on an Apple TV 4K, 3rd gen, A (M) runs both
/// flat out) roughly in proportion, with the picture unchanged.
///
/// A frame counts as a repeat when no 8x8 block of its luma, or of its chroma, differs on average by
/// more than `repeatThreshold` from the frame the reused output was made from. Comparing against that
/// frame, not the previous one, means a slow fade cannot creep past unnoticed one small step at a time.
/// The comparison is one small GPU pass (~2 MB of luma, ~1 MB of chroma).
extension Anime4KUpscaler {
    static let repeatThreshold: Float = 3.0 / 255

    /// The last upscaled output, if `source` repeats the frame it was made from. Caller holds `workLock`.
    func repeatedOutput(for source: CVPixelBuffer, format: PlaneFormat, resources: Resources) -> CVPixelBuffer? {
        guard let compare = resources.repeatCompare, let last = repeatState,
              last.width == CVPixelBufferGetWidth(source), last.height == CVPixelBufferGetHeight(source),
              last.lumaFormat == format.luma, last.chromaFormat == format.chroma,
              let luma = Self.planeTexture(source, plane: 0, format: format.luma, cache: resources.textureCache),
              let chroma = Self.planeTexture(source, plane: 1, format: format.chroma, cache: resources.textureCache),
              let commandBuffer = resources.queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
        let flag = last.flag.contents().assumingMemoryBound(to: UInt32.self)
        flag.pointee = 0
        var threshold = Self.repeatThreshold
        encoder.setComputePipelineState(compare)
        encoder.setBuffer(last.flag, offset: 0, index: 0)
        encoder.setBytes(&threshold, length: MemoryLayout<Float>.stride, index: 1)
        for (current, reference) in [(luma.texture, last.luma), (chroma.texture, last.chroma)] {
            encoder.setTexture(current, index: 0)
            encoder.setTexture(reference, index: 1)
            MPVHookPipeline.dispatchCovering(encoder: encoder, pipeline: compare,
                                             width: (current.width + 7) / 8, height: (current.height + 7) / 8)
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        withExtendedLifetime((luma, chroma)) {}
        let repeated = commandBuffer.error == nil && flag.pointee == 0
        repeatFrames += 1
        if repeated { repeatReused += 1 }
        logRepeatsIfDue()
        return repeated ? last.output : nil
    }

    /// Keeps `output` and a copy of `source`'s planes as the reference for the frames that follow. Caller
    /// holds `workLock`.
    func rememberUpscaled(source: CVPixelBuffer, output: CVPixelBuffer, format: PlaneFormat, resources: Resources) {
        guard resources.repeatCompare != nil else { return }
        let width = CVPixelBufferGetWidth(source), height = CVPixelBufferGetHeight(source)
        if repeatState.map({ $0.width != width || $0.height != height || $0.lumaFormat != format.luma
            || $0.chromaFormat != format.chroma }) ?? true {
            func texture(_ pixelFormat: MTLPixelFormat, _ w: Int, _ h: Int) -> MTLTexture? {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: pixelFormat, width: w, height: h,
                                                                 mipmapped: false)
                d.usage = [.shaderRead]
                d.storageMode = .private
                return resources.device.makeTexture(descriptor: d)
            }
            guard let luma = texture(format.luma, width, height),
                  let chroma = texture(format.chroma, (width + 1) / 2, (height + 1) / 2),
                  let flag = resources.device.makeBuffer(length: 4, options: .storageModeShared) else {
                repeatState = nil
                return
            }
            repeatState = RepeatState(width: width, height: height, lumaFormat: format.luma,
                                      chromaFormat: format.chroma, luma: luma, chroma: chroma, flag: flag,
                                      output: output)
        }
        guard var state = repeatState,
              let luma = Self.planeTexture(source, plane: 0, format: format.luma, cache: resources.textureCache),
              let chroma = Self.planeTexture(source, plane: 1, format: format.chroma, cache: resources.textureCache),
              let commandBuffer = resources.queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            repeatState = nil
            return
        }
        blit.copy(from: luma.texture, to: state.luma)
        blit.copy(from: chroma.texture, to: state.chroma)
        blit.endEncoding()
        // Queue order puts the copy ahead of the next frame's comparison; the textures only need to
        // outlive it.
        commandBuffer.addCompletedHandler { _ in withExtendedLifetime((luma, chroma)) {} }
        commandBuffer.commit()
        state.output = output
        repeatState = state
    }

    func forgetRepeats() {
        repeatState = nil
    }

    private func logRepeatsIfDue() {
        let now = Date()
        guard now.timeIntervalSince(lastRepeatLog) >= 10, repeatFrames > 0 else { return }
        lastRepeatLog = now
        Anime4KLog.emit(String(format: "[Upscaler] repeated frames reused: %d of %d compared (%.0f%%)",
                              repeatReused, repeatFrames, 100 * Double(repeatReused) / Double(repeatFrames)))
    }
}

/// The reference a repeat is judged against: the planes of the frame last upscaled, and its output.
struct RepeatState {
    let width: Int
    let height: Int
    let lumaFormat: MTLPixelFormat
    let chromaFormat: MTLPixelFormat
    let luma: MTLTexture
    let chroma: MTLTexture
    let flag: MTLBuffer
    var output: CVPixelBuffer
}
