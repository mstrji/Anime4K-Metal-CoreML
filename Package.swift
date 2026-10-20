// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "Anime4KUpscaler",
    platforms: [
        .iOS(.v17),
        .tvOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "Anime4KUpscaler", targets: ["Anime4KUpscaler"]),
    ],
    targets: [
        .target(
            name: "Anime4KUpscaler",
            resources: [.copy("Resources/Anime4K")]
        ),
        .testTarget(
            name: "Anime4KUpscalerTests",
            dependencies: ["Anime4KUpscaler"]
        ),
    ]
)
