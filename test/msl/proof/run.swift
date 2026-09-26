// Runs one compute kernel from a metallib as a single thread and prints the
// float4 results it writes to buffer 0, one per line. With --size, prints the
// size of the kernel's GPU machine code instead (the compiled pipeline, stored
// in a binary archive). Used by proof.py and linking.sh.
//
// usage: swiftc -O run.swift -o run
//        ./run <file.metallib> <kernel> <count>
//        ./run --size <file.metallib> <kernel>

import Foundation
import Metal

var args = CommandLine.arguments
let size = args.count > 1 && args[1] == "--size"
if size { args.remove(at: 1) }
guard (size && args.count == 3) || (args.count == 4 && Int(args[3]) != nil) else {
    print("usage: run <file.metallib> <kernel> <count>\n       run --size <file.metallib> <kernel>")
    exit(2)
}
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    print("no Metal device")
    exit(2)
}
do {
    let library = try device.makeLibrary(URL: URL(fileURLWithPath: args[1]))
    guard let function = library.makeFunction(name: args[2]) else {
        print("no kernel named \(args[2])")
        exit(2)
    }
    if size {
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        let archive = try device.makeBinaryArchive(descriptor: MTLBinaryArchiveDescriptor())
        try archive.addComputePipelineFunctions(descriptor: descriptor)
        let url = FileManager.default.temporaryDirectory.appending(path: "lygia-\(UUID().uuidString).binarchive")
        try archive.serialize(to: url)
        print(try FileManager.default.attributesOfItem(atPath: url.path)[.size] as! Int)
        try? FileManager.default.removeItem(at: url)
        exit(0)
    }
    let count = Int(args[3])!
    let pipeline = try device.makeComputePipelineState(function: function)
    let length = count * MemoryLayout<SIMD4<Float>>.stride
    guard let buffer = device.makeBuffer(length: length, options: .storageModeShared),
          let commands = queue.makeCommandBuffer(),
          let encoder = commands.makeComputeCommandEncoder() else {
        print("couldn't create the Metal objects")
        exit(2)
    }
    memset(buffer.contents(), 0, length)
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(buffer, offset: 0, index: 0)
    encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error {
        print("GPU error: \(error.localizedDescription)")
        exit(2)
    }
    let results = buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
    for i in 0..<count {
        let v = results[i]
        print(v.x, v.y, v.z, v.w)
    }
} catch {
    print(error.localizedDescription)
    exit(2)
}
