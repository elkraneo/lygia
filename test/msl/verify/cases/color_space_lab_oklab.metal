// Ported from test/wesl/color-space-lab-oklab.test.ts (same inputs and expected values).
// WGSL names encode the argument type; in MSL they're overloads:
//   WGSL xxx(vec3f) / xxx4(vec4f)  ==  MSL xxx(float3) / xxx(float4)
// vec3f results are written as float4(result, 0) and only x y z are checked.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/space/oklab2srgb.msl"
#include "lygia/color/space/srgb2oklab.msl"
#include "lygia/color/space/oklab2rgb.msl"
#include "lygia/color/space/rgb2oklab.msl"

// @test oklab2srgb_test
// @expect oklab2srgb_test[0] 1.0 0.0 0.0
kernel void oklab2srgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(oklab2srgb(float3(0.628, 0.225, 0.126)), 0.0);
}

// @test srgb2oklab_test
// @expect srgb2oklab_test[0] 0.628 0.2249 0.1258
kernel void srgb2oklab_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(srgb2oklab(float3(1.0, 0.0, 0.0)), 0.0);
}

// @test oklab2rgb_test
// @expect oklab2rgb_test[0] 1.0008 -0.0002 -0.0002
kernel void oklab2rgb_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(oklab2rgb(float3(0.628, 0.225, 0.126)), 0.0);
}

// @test rgb2oklab_test
// @expect rgb2oklab_test[0] 0.628 0.2249 0.1258
kernel void rgb2oklab_test(device float4* results [[buffer(0)]]) {
    results[0] = float4(rgb2oklab(float3(1.0, 0.0, 0.0)), 0.0);
}

// WGSL oklab2rgb4
// @test oklab2rgb4_test
// @expect oklab2rgb4_test[0] 1.0008 -0.0002 -0.0002 0.45
kernel void oklab2rgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = oklab2rgb(float4(0.628, 0.225, 0.126, 0.45));
}

// WGSL oklab2srgb4
// @test oklab2srgb4_test
// @expect oklab2srgb4_test[0] 1.0 0.0 0.0 0.35
kernel void oklab2srgb4_test(device float4* results [[buffer(0)]]) {
    results[0] = oklab2srgb(float4(0.628, 0.225, 0.126, 0.35));
}

// WGSL rgb2oklab4
// @test rgb2oklab4_test
// @expect rgb2oklab4_test[0] 0.628 0.2249 0.1258 0.6
kernel void rgb2oklab4_test(device float4* results [[buffer(0)]]) {
    results[0] = rgb2oklab(float4(1.0, 0.0, 0.0, 0.6));
}

// WGSL srgb2oklab4
// @test srgb2oklab4_test
// @expect srgb2oklab4_test[0] 0.62796 0.22486 0.12585 0.85
kernel void srgb2oklab4_test(device float4* results [[buffer(0)]]) {
    results[0] = srgb2oklab(float4(1.0, 0.0, 0.0, 0.85));
}
