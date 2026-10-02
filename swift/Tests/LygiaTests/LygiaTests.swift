import Metal
import XCTest
@testable import Lygia

final class LygiaTests: XCTestCase {
    func testBundleHasTheFiles() {
        let files = Lygia.files
        XCTAssertGreaterThan(files.count, 600)
        XCTAssertTrue(files.contains("generative/fbm.msl"))
        XCTAssertTrue(files.contains("lygia.msl"), "lygia.msl defines LYGIA_FNC")
        XCTAssertNotNil(Lygia.version)
    }

    func testFlattenInlinesIncludes() throws {
        let flat = try Lygia.flatten(#"#include "lygia/generative/fbm.msl""#)
        // fbm includes snoise, which includes random: all of it ends up inline.
        XCTAssertTrue(flat.contains("LYGIA_FNC FBM_NOISE_TYPE fbm("), "fbm itself")
        XCTAssertTrue(flat.contains("snoise("), "its noise")
        XCTAssertTrue(flat.contains("#define LYGIA_FNC static inline"), "lygia.msl was inlined first")
        // Only the doc comment's example include and "(already included)" notes may remain.
        let active = flat.split(separator: "\n").filter { $0.trimmingCharacters(in: .whitespaces).hasPrefix("#include \"") }
        XCTAssertEqual(active.count, 1, "only lygia.msl's documented example include remains: \(active)")
    }

    func testMissingIncludeIsAnError() {
        XCTAssertThrowsError(try Lygia.flatten(#"#include "lygia/nope.msl""#))
    }

    func testCompilesOnTheGPU() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let source = """
        #include "lygia/generative/fbm.msl"
        kernel void k(device float* out [[buffer(0)]]) { out[0] = fbm(float2(0.3, 0.7)); }
        """
        let library = try await Lygia.makeLibrary(source: source, device: device, defines: ["FBM_OCTAVES": "2"])
        XCTAssertNotNil(library.makeFunction(name: "k"))
    }

    func testPreparedSourceIsPlainMetal() throws {
        let prepared = try Lygia.prepare("float f() { return 1.0; }", defines: ["A": "1"])
        XCTAssertTrue(prepared.hasPrefix("#define A 1\n#include <metal_stdlib>\nusing namespace metal;\n"))
    }
}
