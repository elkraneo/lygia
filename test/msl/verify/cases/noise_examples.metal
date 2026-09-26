// Ported from test/wesl/noise-examples.test.ts.
//
// All five tests render a shader from test/wesl/shaders/ and compare it with an image
// snapshot, which this harness can't do: "Perlin noise FBM pattern" (perlin-noise-fbm),
// "Simplex noise FBM pattern" (snoise-fbm), "Worley cellular pattern" (worley-cellular),
// "Periodic noise tiling pattern" (pnoise-tiling), "Wavelet vorticity pattern"
// (wavelet-vorticity). Approximation: each kernel evaluates the same per-pixel color
// expression as the shader on a 16x16 grid of pixel centers and checks the colors are
// finite and in range (the harness also fails on NaN/inf). No pixel values are compared.
//
// WGSL's random defaults to the sin-less hash with the .1031 scale (used by worley and
// wavelet), so use WGSL's configuration.
#include <metal_stdlib>
using namespace metal;
#define RANDOM_SINLESS
#define RANDOM_HIGHER_RANGE
#include "lygia/generative/cnoise.msl"
#include "lygia/generative/snoise.msl"
#include "lygia/generative/worley.msl"
#include "lygia/generative/pnoise.msl"
#include "lygia/generative/wavelet.msl"

// (perlin fbm, simplex fbm, worley, pnoise fbm) colors; simplex noise can overshoot
// [-1, 1] slightly (the WGSL noise tests allow 1.1), hence the margin.
// @test patterns 256
// @range patterns -0.05 1.05
kernel void patterns(device float4* results [[buffer(0)]]) {
    for (uint i = 0; i < 256; i++) {
        float2 st = (float2(i % 16u, i / 16u) + 0.5) / 16.0;
        float perlin = 0.0, simplex = 0.0, periodic = 0.0;
        float amplitude = 0.5, frequency = 3.0;
        for (int o = 0; o < 5; o++) {
            perlin += cnoise(st * frequency) * amplitude;
            frequency *= 2.0;
            amplitude *= 0.5;
        }
        amplitude = 0.5; frequency = 5.0;
        for (int o = 0; o < 5; o++) {
            simplex += amplitude * snoise(st * frequency);
            periodic += amplitude * pnoise(st * frequency, float2(frequency));
            frequency *= 2.0;
            amplitude *= 0.5;
        }
        results[i] = float4(perlin * 0.5 + 0.5, simplex * 0.5 + 0.5, worley(st * 8.0), periodic * 0.5 + 0.5);
    }
}

// wavelet-vorticity color (WAVELET_VORTICITY unset, as the WGSL const is 0.0)
// @test wavelet 64
// @range wavelet -0.05 1.05
kernel void wavelet(device float4* results [[buffer(0)]]) {
    for (uint i = 0; i < 256; i++) {
        float2 st = (float2(i % 16u, i / 16u) + 0.5) / 16.0;
        float value = 0.0, amplitude = 0.5, frequency = 5.0;
        for (int o = 0; o < 5; o++) {
            value += amplitude * wavelet(st * frequency);
            frequency *= 2.0;
            amplitude *= 0.5;
        }
        results[i / 4][i % 4] = value * 0.5 + 0.5;
    }
}
