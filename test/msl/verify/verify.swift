// Runs the Metal verification cases in test/msl/verify/cases on the GPU.
//
// Each case file is a .metal file with kernels that write float4 results, plus
// annotations with the expected values (taken from the WGSL tests in test/wesl,
// so Metal is checked against the same numbers as the other languages):
//
//   // @test name [count]            kernel `name` writes `count` float4s (default 1)
//   // @expect name[i] x [y [z [w]]]  result i matches (only the given components)
//   // @same name[i] name[j]          results i and j match
//   // @differ name[i] name[j]        results i and j differ (in the given component, .x)
//   // @range name lo hi              every component of every result is in [lo, hi]
//   // @eps 0.001                     tolerance for this file (default 0.0001, as in testUtil.ts)
//   // @xfail name[i] reason          checks on result i are expected to fail (a known, documented
//                                     divergence); they're reported, and it's an error if they pass
//
// Kernels have the signature `kernel void name(device float4* results [[buffer(0)]])`
// and run as a single thread. LYGIA is included as "lygia/...".
//
// usage: swiftc -O verify.swift -o verify && ./verify <lygia root> [case.metal ...]

import Foundation
import Metal

struct Failure: Error { let message: String }

func run(_ args: [String]) -> Int32 {
    guard args.count >= 2 else {
        print("usage: verify <lygia root> [case.metal ...]")
        return 2
    }
    let root = URL(fileURLWithPath: args[1]).standardizedFileURL
    var cases = args.dropFirst(2).map { URL(fileURLWithPath: $0) }
    if cases.isEmpty {
        let dir = root.appending(path: "test/msl/verify/cases")
        cases = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "metal" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        print("no Metal device")
        return 2
    }

    // "lygia/..." includes resolve through a temporary folder with a symlink to the root.
    let tmp = FileManager.default.temporaryDirectory.appending(path: "lygia-verify-\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    try? FileManager.default.createSymbolicLink(at: tmp.appending(path: "lygia"), withDestinationURL: root)
    defer { try? FileManager.default.removeItem(at: tmp) }

    var passed = 0, failed = 0
    for file in cases {
        let name = file.lastPathComponent
        do {
            let (checks, notes) = try verify(file: file, includeDir: tmp, device: device, queue: queue)
            passed += checks
            print("PASS \(name) (\(checks) checks)")
            notes.forEach { print($0) }
        } catch let failure as Failure {
            failed += 1
            print("FAIL \(name)\n\(failure.message)")
        } catch {
            failed += 1
            print("FAIL \(name)\n     \(error)")
        }
    }
    print("MSL verify: \(cases.count - failed)/\(cases.count) files passed, \(passed) checks")
    return failed == 0 ? 0 : 1
}

func verify(file: URL, includeDir: URL, device: MTLDevice, queue: MTLCommandQueue) throws -> (Int, [String]) {
    let source = try String(contentsOf: file, encoding: .utf8)
    var eps: Float = 0.0001
    var counts: [String: Int] = [:]
    var order: [String] = []
    var directives: [[String]] = []
    for line in source.split(separator: "\n") {
        let t = line.trimmingCharacters(in: .whitespaces)
        guard t.hasPrefix("// @") else { continue }
        let parts = t.dropFirst(4).split(separator: " ").map(String.init)
        switch parts.first {
        case "test":
            counts[parts[1]] = parts.count > 2 ? Int(parts[2]) ?? 1 : 1
            order.append(parts[1])
        case "eps":
            eps = Float(parts[1]) ?? eps
        default:
            directives.append(parts)
        }
    }
    if order.isEmpty { throw Failure(message: "     no @test kernels") }

    // Compile offline: the runtime compiler can't resolve #include paths.
    let lib = includeDir.appending(path: file.deletingPathExtension().lastPathComponent + ".metallib")
    let compile = Process()
    compile.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
    compile.arguments = ["-sdk", "macosx", "metal", "-std=metal3.1", "-I", includeDir.path, file.path, "-o", lib.path]
    let errPipe = Pipe()
    compile.standardError = errPipe
    try compile.run()
    compile.waitUntilExit()
    if compile.terminationStatus != 0 {
        let err = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let firstErrors = err.split(separator: "\n").filter { $0.contains("error") }.prefix(3).joined(separator: "\n     ")
        throw Failure(message: "     compile failed:\n     \(firstErrors)")
    }
    let library = try device.makeLibrary(URL: lib)

    var results: [String: [SIMD4<Float>]] = [:]
    for kernel in order {
        guard let fn = library.makeFunction(name: kernel) else { throw Failure(message: "     no kernel \(kernel)") }
        let pso = try device.makeComputePipelineState(function: fn)
        let count = counts[kernel] ?? 1
        let buffer = device.makeBuffer(length: count * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)!
        memset(buffer.contents(), 0, buffer.length)
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(buffer, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw Failure(message: "     \(kernel): \(error)") }
        let p = buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        results[kernel] = (0..<count).map { p[$0] }
    }

    func value(_ ref: String) throws -> SIMD4<Float> {
        // name[i]
        guard let open = ref.firstIndex(of: "["), let close = ref.firstIndex(of: "]"),
              let i = Int(ref[ref.index(after: open)..<close]) else { throw Failure(message: "     bad reference \(ref)") }
        let name = String(ref[..<open])
        guard let r = results[name], i < r.count else { throw Failure(message: "     unknown result \(ref)") }
        return r[i]
    }
    func show(_ v: SIMD4<Float>) -> String { "(\(v.x), \(v.y), \(v.z), \(v.w))" }

    var messages: [String] = []
    var checks = 0
    var xfails: [String: String] = [:]          // ref -> reason
    for d in directives where d.first == "xfail" {
        xfails[d[1]] = d.dropFirst(2).joined(separator: " ")
    }
    var xfailed: Set<String> = []
    for d in directives where d.first != "xfail" {
        checks += 1
        let before = messages.count
        defer {
            // A failing check on an @xfail result is expected: record it instead of failing.
            if d.count > 1, xfails[d[1]] != nil, messages.count > before {
                messages.removeLast(messages.count - before)
                xfailed.insert(d[1])
            }
        }
        switch d.first {
        case "expect":
            let got = try value(d[1])
            let want = d.dropFirst(2).compactMap(Float.init)
            for (c, w) in want.enumerated() where !(abs(got[c] - w) < eps) {
                messages.append("     \(d[1]): got \(show(got)), expected \(want) (component \(c), eps \(eps))")
                break
            }
        case "same":
            let a = try value(d[1]), b = try value(d[2])
            if (0..<4).contains(where: { !(abs(a[$0] - b[$0]) < eps) }) { messages.append("     \(d[1]) \(show(a)) != \(d[2]) \(show(b))") }
        case "differ":
            let a = try value(d[1]), b = try value(d[2])
            if abs(a.x - b.x) < 0.05 { messages.append("     \(d[1]) \(show(a)) should differ from \(d[2]) \(show(b))") }
        case "range":
            guard let r = results[d[1]], let lo = Float(d[2]), let hi = Float(d[3]) else { break }
            if let bad = r.first(where: { v in (0..<4).contains(where: { !(v[$0] >= lo && v[$0] <= hi) }) }) {
                messages.append("     \(d[1]): \(show(bad)) outside [\(lo), \(hi)]")
            }
        default:
            messages.append("     unknown directive @\(d.joined(separator: " "))")
        }
    }
    for (ref, reason) in xfails.sorted(by: { $0.key < $1.key }) where !xfailed.contains(ref) {
        messages.append("     \(ref) passes but is marked @xfail (\(reason)); remove the @xfail")
    }
    let notes = xfailed.sorted().map { "     xfail \($0): \(xfails[$0]!)" }
    // Any NaN or infinity in the results is a failure, even without an expectation.
    for (name, r) in results {
        if let i = r.firstIndex(where: { v in (0..<4).contains(where: { !v[$0].isFinite }) }) {
            messages.append("     \(name)[\(i)] is not finite: \(show(r[i]))")
        }
    }
    if !messages.isEmpty { throw Failure(message: (messages + notes).joined(separator: "\n")) }
    return (checks, notes)
}

exit(run(CommandLine.arguments))
