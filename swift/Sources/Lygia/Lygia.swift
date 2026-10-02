import Foundation
import Metal

/// LYGIA's Metal shader library, as a Swift package.
///
/// There are two ways to use it:
///
/// 1. **At runtime**, compile Metal source that includes LYGIA with
///    ``makeLibrary(source:device:options:defines:)``. The `#include "lygia/..."`
///    lines are inlined from the bundled files before the Metal compiler sees
///    them (it can't read files itself). Works on every platform with no build
///    settings; the first compile of a source takes tens of milliseconds.
///
/// 2. **At build time**, for `.metal` files in your own target, point the Metal
///    compiler at the bundled headers in your target's build settings
///    (SwiftPM can't do this for you):
///
///        MTL_HEADER_SEARCH_PATHS = $(BUILT_PRODUCTS_DIR)/Lygia_Lygia.bundle/Contents/Resources
///
///    (without `Contents/Resources` on iOS, tvOS and visionOS). Or copy the
///    `lygia` folder from ``includePath`` into your project.
///
/// Every function is `static inline` by default, so several `.metal` files can
/// include the same LYGIA file. Define `LYGIA_FNC` as empty before the includes
/// to export the functions instead, to build LYGIA once as a helper library.
public enum Lygia {
    /// The folder that contains `lygia/`, for `-I` or `MTL_HEADER_SEARCH_PATHS`.
    public static var includePath: URL {
        Bundle.module.resourceURL!
    }

    /// The bundled `lygia/` folder itself.
    public static var root: URL {
        includePath.appendingPathComponent("lygia")
    }

    /// LYGIA's version, from the bundled `lygia/VERSION` file when present.
    public static var version: String? {
        try? String(contentsOf: root.appendingPathComponent("VERSION"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Relative paths of every bundled `.msl` file, sorted (e.g. `generative/fbm.msl`).
    public static var files: [String] {
        guard let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) else { return [] }
        let prefix = root.standardizedFileURL.path + "/"
        return enumerator.compactMap { ($0 as? URL)?.standardizedFileURL.path }
            .filter { $0.hasSuffix(".msl") }
            .map { String($0.dropFirst(prefix.count)) }
            .sorted()
    }

    /// `source` with its `#include "lygia/..."` (and nested) includes inlined,
    /// ready for `MTLDevice.makeLibrary(source:)`. `#line` directives keep
    /// compiler errors pointing at the original files.
    public static func flatten(_ source: String, name: String = "shader.metal") throws -> String {
        var flattener = IncludeFlattener(searchRoots: [includePath])
        return try flattener.flatten(source, name: name)
    }

    /// Compiles Metal source that includes LYGIA. `<metal_stdlib>` is included
    /// and `using namespace metal;` added unless the source already has them.
    /// `defines` become `#define` lines in front (e.g. `["FBM_OCTAVES": "6"]`).
    public static func makeLibrary(source: String, device: MTLDevice,
                                   options: MTLCompileOptions? = nil,
                                   defines: [String: String] = [:]) async throws -> MTLLibrary {
        try await device.makeLibrary(source: prepare(source, defines: defines), options: options)
    }

    /// The full source ``makeLibrary(source:device:options:defines:)`` compiles.
    public static func prepare(_ source: String, defines: [String: String] = [:]) throws -> String {
        var head = defines.keys.sorted().map { "#define \($0) \(defines[$0]!)" }
        if !source.contains("<metal_stdlib>") { head.append("#include <metal_stdlib>") }
        if !source.contains("using namespace metal") { head.append("using namespace metal;") }
        let body = try flatten(source)
        return (head + [body]).joined(separator: "\n")
    }
}
