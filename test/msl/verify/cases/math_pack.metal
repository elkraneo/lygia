// Ported from test/wesl/math-pack.test.ts (same inputs and expected values).
// WGSL names encode the argument type; in MSL they're overloads:
//   WGSL unpack4(vec4f)            ==  MSL unpack(float4)   (ThreeJS UnpackFactors)
//   WGSL unpackBase(vec3f, base)   ==  MSL unpack(float3, float)
//   WGSL unpack(vec3f)             ==  MSL unpack(float3)   (UNPACK_FNC defaults to unpack256 in both)
// Note: WGSL's pack() tidies the overflow sequentially (r.y -= r.x/256, then r.z -= r.y/256
// with the updated r.y, ...), while GLSL and MSL do it in one step (r.yzw -= r.xyz/256).
#include <metal_stdlib>
using namespace metal;
#include "lygia/math/pack.msl"
#include "lygia/math/unpack.msl"

// "pack/unpack roundtrip"
// @test roundtrip
// @expect roundtrip[0] 0.12346 0.12346
kernel void roundtrip(device float4* results [[buffer(0)]]) {
    float original = 0.12346;
    results[0] = float4(original, unpack(pack(original)), 0.0, 0.0);
}

// "pack/unpack roundtrip - multiple values"
// @test roundtripMultiple
// @expect roundtripMultiple[0] 0.0 0.25 0.5 0.75
kernel void roundtripMultiple(device float4* results [[buffer(0)]]) {
    results[0] = float4(unpack(pack(0.0)), unpack(pack(0.25)), unpack(pack(0.5)), unpack(pack(0.75)));
}

// "unpack256 - default base 256" (only x, y, z are checked, as in the WGSL test)
// @test unpack256
// @expect unpack256[0] 0.000015 0.50787 1.01578
kernel void unpack256(device float4* results [[buffer(0)]]) {
    results[0] = float4(unpack256(float3(1.0, 0.0, 0.0)), unpack256(float3(0.5, 0.5, 0.5)), unpack256(float3(1.0, 1.0, 1.0)), 0.0);
}

// "unpack - alias for unpack256": results[0].x == results[0].y
// @test unpackAlias 2
// @same unpackAlias[0] unpackAlias[1]
kernel void unpackAlias(device float4* results [[buffer(0)]]) {
    float3 v = float3(0.5, 0.5, 0.5);
    results[0] = float4(unpack(v), 0.0, 0.0, 0.0);
    results[1] = float4(unpack256(v), 0.0, 0.0, 0.0);
}

// @test unpack8
// @expect unpack8[0] 0.015625 0.125 1.0 0.5703125
kernel void unpack8(device float4* results [[buffer(0)]]) {
    results[0] = float4(unpack8(float3(1.0, 0.0, 0.0)), unpack8(float3(0.0, 1.0, 0.0)), unpack8(float3(0.0, 0.0, 1.0)), unpack8(float3(0.5, 0.5, 0.5)));
}

// @test unpack16
// @expect unpack16[0] 0.00390625 0.0625 1.0 0.533203125
kernel void unpack16(device float4* results [[buffer(0)]]) {
    results[0] = float4(unpack16(float3(1.0, 0.0, 0.0)), unpack16(float3(0.0, 1.0, 0.0)), unpack16(float3(0.0, 0.0, 1.0)), unpack16(float3(0.5, 0.5, 0.5)));
}

// @test unpack32
// @expect unpack32[0] 0.00098 0.03125 1.0 0.5161
kernel void unpack32(device float4* results [[buffer(0)]]) {
    results[0] = float4(unpack32(float3(1.0, 0.0, 0.0)), unpack32(float3(0.0, 1.0, 0.0)), unpack32(float3(0.0, 0.0, 1.0)), unpack32(float3(0.5, 0.5, 0.5)));
}

// @test unpack64
// @expect unpack64[0] 0.000244 0.015625 1.0 0.507935
kernel void unpack64(device float4* results [[buffer(0)]]) {
    results[0] = float4(unpack64(float3(1.0, 0.0, 0.0)), unpack64(float3(0.0, 1.0, 0.0)), unpack64(float3(0.0, 0.0, 1.0)), unpack64(float3(0.5, 0.5, 0.5)));
}

// @test unpack128
// @expect unpack128[0] 0.000061 0.0078125 1.0 0.503967
kernel void unpack128(device float4* results [[buffer(0)]]) {
    results[0] = float4(unpack128(float3(1.0, 0.0, 0.0)), unpack128(float3(0.0, 1.0, 0.0)), unpack128(float3(0.0, 0.0, 1.0)), unpack128(float3(0.5, 0.5, 0.5)));
}

// "unpackBase - custom base"
// @test unpackBase
// @expect unpackBase[0] 0.01 0.1 1.0 0.555
kernel void unpackBase(device float4* results [[buffer(0)]]) {
    float base = 10.0;
    results[0] = float4(unpack(float3(1.0, 0.0, 0.0), base), unpack(float3(0.0, 1.0, 0.0), base),
                        unpack(float3(0.0, 0.0, 1.0), base), unpack(float3(0.5, 0.5, 0.5), base));
}

// "unpack4 - vec4 unpacking (ThreeJS style)"
// @test unpack4
// @expect unpack4[0] 5.960464e-8 0.99609375 0.5 1.0
kernel void unpack4(device float4* results [[buffer(0)]]) {
    results[0] = float4(unpack(float4(1.0, 0.0, 0.0, 0.0)), unpack(float4(0.0, 0.0, 0.0, 1.0)),
                        unpack(float4(0.5, 0.5, 0.5, 0.5)), unpack(float4(1.0, 1.0, 1.0, 1.0)));
}
