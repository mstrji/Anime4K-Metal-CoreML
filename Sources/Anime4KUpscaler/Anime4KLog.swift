import Foundation

/// Where the upscaler's diagnostic lines go: preparation, the first frame, slow frames, step-downs
/// and why it switched itself off. Prints nothing until the host sets `handler`.
public enum Anime4KLog {
    nonisolated(unsafe) public static var handler: (@Sendable (String) -> Void)?

    static func emit(_ line: String) {
        handler?(line)
    }
}
