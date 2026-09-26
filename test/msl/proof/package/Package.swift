// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "LygiaKit",
    platforms: [.macOS(.v14), .iOS(.v17), .visionOS(.v1)],
    products: [.library(name: "LygiaKit", targets: ["LygiaKit"])],
    targets: [.target(name: "LygiaKit", resources: [.process("Shaders")])]
)
