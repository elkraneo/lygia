// swift-tools-version: 5.9
// LYGIA's Metal files as a Swift package. See swift/README.md.
import PackageDescription

let package = Package(
    name: "Lygia",
    platforms: [.macOS(.v14), .iOS(.v17), .tvOS(.v17), .visionOS(.v1)],
    products: [
        .library(name: "Lygia", targets: ["Lygia"]),
    ],
    targets: [
        // The .msl tree is copied into the bundle as-is, so `#include "lygia/..."`
        // resolves against Bundle.module's `lygia` folder (see Lygia.includePath).
        .target(
            name: "Lygia",
            path: "swift/Sources/Lygia",
            resources: [.copy("Resources/lygia")]
        ),
        .testTarget(
            name: "LygiaTests",
            dependencies: ["Lygia"],
            path: "swift/Tests/LygiaTests"
        ),
    ]
)
