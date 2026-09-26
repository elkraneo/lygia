// Ported from test/wesl/generative-noise.test.ts (same inputs and expected values).
// Name mapping (WGSL -> MSL): cnoise2/3/4 -> cnoise(float2|3|4); snoise2/3/4 -> snoise(float2|3|4);
// snoise22 -> snoise2(float2); snoise33 -> snoise3(float3); snoise34 -> snoise3(float4);
// pnoise2/3/4(p, period) -> pnoise(float2|3|4, float2|3|4). None of these use random,
// so no configuration defines are needed.
//
// The WGSL distribution tests write 256 pairs (p, p + 0.01) to the results and check in
// JS: the max |pair difference| (continuity), the min/max over all 512 values, and the
// value at the first point (regression). Here each kernel evaluates the same 256 pairs
// and writes [0] = (first value, max diff, min, max) and [1] = 1/0 flags for
// (max diff < limit, min >= lo, max <= hi). Other inequality checks are flags too.
#include <metal_stdlib>
using namespace metal;
#include "lygia/generative/cnoise.msl"
#include "lygia/generative/snoise.msl"
#include "lygia/generative/pnoise.msl"

static inline float flag(bool b) { return b ? 1.0 : 0.0; }

// Accumulates (first value, max pair diff, min, max) over the pairs.
static inline void accumulate(thread float4& s, uint i, float a, float b) {
    if (i == 0) s = float4(a, 0.0, min(a, b), max(a, b));
    s.y = max(s.y, abs(b - a));
    s.z = min(s.z, min(a, b));
    s.w = max(s.w, max(a, b));
}

// WGSL cnoise2: max diff < 0.1, values in [-1.0, 1.0], regression at the first point
// @test cnoise2 2
// @expect cnoise2[0] -0.4915
// @expect cnoise2[1] 1 1 1
kernel void cnoise2(device float4* results [[buffer(0)]]) {
    float4 s = float4(0.0);
    for (uint i = 0; i < 256; i++) {
        float2 p = float2(float(i % 16u) * 0.2 + 0.5, float(i / 16u) * 0.2 + 0.5);
        accumulate(s, i, cnoise(p), cnoise(p + 0.01));
    }
    results[0] = s;
    results[1] = float4(flag(s.y < 0.1), flag(s.z >= -1.0), flag(s.w <= 1.0), 0.0);
}

// WGSL cnoise3: max diff < 0.1, values in [-1.0, 1.0], regression at the first point
// @test cnoise3 2
// @expect cnoise3[0] -0.3962
// @expect cnoise3[1] 1 1 1
kernel void cnoise3(device float4* results [[buffer(0)]]) {
    float4 s = float4(0.0);
    for (uint i = 0; i < 256; i++) {
        float3 p = float3(float(i % 8u) * 0.2 + 0.5, float((i / 8u) % 8u) * 0.2 + 0.5, float(i / 64u) * 0.2 + 0.5);
        accumulate(s, i, cnoise(p), cnoise(p + 0.01));
    }
    results[0] = s;
    results[1] = float4(flag(s.y < 0.1), flag(s.z >= -1.0), flag(s.w <= 1.0), 0.0);
}

// WGSL cnoise4: max diff < 0.1, values in [-1.0, 1.0], regression at the first point
// @test cnoise4 2
// @expect cnoise4[0] 0.0203
// @expect cnoise4[1] 1 1 1
kernel void cnoise4(device float4* results [[buffer(0)]]) {
    float4 s = float4(0.0);
    for (uint i = 0; i < 256; i++) {
        float4 p = float4(float(i % 4u) * 0.2 + 0.5, float((i / 4u) % 4u) * 0.2 + 0.5, float((i / 16u) % 4u) * 0.2 + 0.5, float(i / 64u) * 0.2 + 0.5);
        accumulate(s, i, cnoise(p), cnoise(p + 0.01));
    }
    results[0] = s;
    results[1] = float4(flag(s.y < 0.1), flag(s.z >= -1.0), flag(s.w <= 1.0), 0.0);
}

// WGSL snoise2: max diff < 0.2, values in [-1.1, 1.1], regression at the first point
// @test snoise2 2
// @expect snoise2[0] 0.3683
// @expect snoise2[1] 1 1 1
kernel void snoise2(device float4* results [[buffer(0)]]) {
    float4 s = float4(0.0);
    for (uint i = 0; i < 256; i++) {
        float2 p = float2(float(i % 16u) * 0.2 + 1.0, float(i / 16u) * 0.2 + 2.0);
        accumulate(s, i, snoise(p), snoise(p + 0.01));
    }
    results[0] = s;
    results[1] = float4(flag(s.y < 0.2), flag(s.z >= -1.1), flag(s.w <= 1.1), 0.0);
}

// WGSL snoise3: max diff < 0.35, values in [-1.1, 1.1], regression at the first point
// @test snoise3 2
// @expect snoise3[0] 0.7335
// @expect snoise3[1] 1 1 1
kernel void snoise3(device float4* results [[buffer(0)]]) {
    float4 s = float4(0.0);
    for (uint i = 0; i < 256; i++) {
        float3 p = float3(float(i % 8u) * 0.2 + 1.0, float((i / 8u) % 8u) * 0.2 + 2.0, float(i / 64u) * 0.2 + 3.0);
        accumulate(s, i, snoise(p), snoise(p + 0.01));
    }
    results[0] = s;
    results[1] = float4(flag(s.y < 0.35), flag(s.z >= -1.1), flag(s.w <= 1.1), 0.0);
}

// WGSL snoise4: max diff < 0.2, values in [-1.1, 1.1], regression at the first point
// @test snoise4 2
// @expect snoise4[0] -0.3748
// @expect snoise4[1] 1 1 1
kernel void snoise4(device float4* results [[buffer(0)]]) {
    float4 s = float4(0.0);
    for (uint i = 0; i < 256; i++) {
        float4 p = float4(float(i % 4u) * 0.2 + 1.0, float((i / 4u) % 4u) * 0.2 + 2.0, float((i / 16u) % 4u) * 0.2 + 3.0, float(i / 64u) * 0.2 + 4.0);
        accumulate(s, i, snoise(p), snoise(p + 0.01));
    }
    results[0] = s;
    results[1] = float4(flag(s.y < 0.2), flag(s.z >= -1.1), flag(s.w <= 1.1), 0.0);
}

// WGSL snoise22: determinism
// @test snoise22 2
// @same snoise22[0] snoise22[1]
kernel void snoise22(device float4* results [[buffer(0)]]) {
    float2 n1 = snoise2(float2(1.0, 2.0)), n2 = snoise2(float2(1.0, 2.0));
    results[0] = float4(n1, n2);
    results[1] = float4(n2, n1);
}

// WGSL snoise33: length(n3 - n1) < 0.2, regression
// @test snoise33 2
// @expect snoise33[0] 0.7335
// @expect snoise33[1] 1
kernel void snoise33(device float4* results [[buffer(0)]]) {
    float3 n1 = snoise3(float3(1.0, 2.0, 3.0));
    float3 n3 = snoise3(float3(1.01, 2.01, 3.01));
    results[0] = float4(n1, length(n3 - n1));
    results[1] = float4(flag(length(n3 - n1) < 0.2), 0.0, 0.0, 0.0);
}

// WGSL snoise34: length(n3 - n1) < 0.2, regression
// @test snoise34 2
// @expect snoise34[0] -0.3748
// @expect snoise34[1] 1
kernel void snoise34(device float4* results [[buffer(0)]]) {
    float3 n1 = snoise3(float4(1.0, 2.0, 3.0, 4.0));
    float3 n3 = snoise3(float4(1.01, 2.01, 3.01, 4.01));
    results[0] = float4(n1, length(n3 - n1));
    results[1] = float4(flag(length(n3 - n1) < 0.2), 0.0, 0.0, 0.0);
}

// WGSL pnoise2, period 4: max diff < 0.1, values in [-1.0, 1.0], regression at the first point
// @test pnoise2 2
// @expect pnoise2[0] -0.4915
// @expect pnoise2[1] 1 1 1
kernel void pnoise2(device float4* results [[buffer(0)]]) {
    float4 s = float4(0.0);
    for (uint i = 0; i < 256; i++) {
        float2 p = float2(float(i % 16u) * 0.2 + 0.5, float(i / 16u) * 0.2 + 0.5);
        accumulate(s, i, pnoise(p, float2(4.0)), pnoise(p + 0.01, float2(4.0)));
    }
    results[0] = s;
    results[1] = float4(flag(s.y < 0.1), flag(s.z >= -1.0), flag(s.w <= 1.0), 0.0);
}

// WGSL pnoise2 periodicity: n(p) == n(p + period) == n(p + 2 period)
// @test pnoise2Period 2
// @same pnoise2Period[0] pnoise2Period[1]
kernel void pnoise2Period(device float4* results [[buffer(0)]]) {
    float2 period = float2(4.0), p = float2(0.5);
    float n1 = pnoise(p, period), n2 = pnoise(p + period, period), n3 = pnoise(p + period * 2.0, period);
    results[0] = float4(n1, n2, n3, 0.0);
    results[1] = float4(n1, n1, n1, 0.0);
}

// WGSL pnoise3, period 4: max diff < 0.1, values in [-1.0, 1.0], regression at the first point
// @test pnoise3 2
// @expect pnoise3[0] -0.3962
// @expect pnoise3[1] 1 1 1
kernel void pnoise3(device float4* results [[buffer(0)]]) {
    float4 s = float4(0.0);
    for (uint i = 0; i < 256; i++) {
        float3 p = float3(float(i % 8u) * 0.2 + 0.5, float((i / 8u) % 8u) * 0.2 + 0.5, float(i / 64u) * 0.2 + 0.5);
        accumulate(s, i, pnoise(p, float3(4.0)), pnoise(p + 0.01, float3(4.0)));
    }
    results[0] = s;
    results[1] = float4(flag(s.y < 0.1), flag(s.z >= -1.0), flag(s.w <= 1.0), 0.0);
}

// WGSL pnoise3 periodicity
// @test pnoise3Period 2
// @same pnoise3Period[0] pnoise3Period[1]
kernel void pnoise3Period(device float4* results [[buffer(0)]]) {
    float3 period = float3(4.0), p = float3(0.5);
    float n1 = pnoise(p, period), n2 = pnoise(p + period, period), n3 = pnoise(p + period * 2.0, period);
    results[0] = float4(n1, n2, n3, 0.0);
    results[1] = float4(n1, n1, n1, 0.0);
}

// WGSL pnoise4, period 4: max diff < 0.1, values in [-1.0, 1.0], regression at the first point
// @test pnoise4 2
// @expect pnoise4[0] 0.0203
// @expect pnoise4[1] 1 1 1
kernel void pnoise4(device float4* results [[buffer(0)]]) {
    float4 s = float4(0.0);
    for (uint i = 0; i < 256; i++) {
        float4 p = float4(float(i % 4u) * 0.2 + 0.5, float((i / 4u) % 4u) * 0.2 + 0.5, float((i / 16u) % 4u) * 0.2 + 0.5, float(i / 64u) * 0.2 + 0.5);
        accumulate(s, i, pnoise(p, float4(4.0)), pnoise(p + 0.01, float4(4.0)));
    }
    results[0] = s;
    results[1] = float4(flag(s.y < 0.1), flag(s.z >= -1.0), flag(s.w <= 1.0), 0.0);
}

// WGSL pnoise4 periodicity (three separate shaders in WGSL, three kernels here)
// @test pnoise4Period0
// @test pnoise4Period1
// @test pnoise4Period2
// @same pnoise4Period0[0] pnoise4Period1[0]
// @same pnoise4Period0[0] pnoise4Period2[0]
kernel void pnoise4Period0(device float4* results [[buffer(0)]]) {
    results[0] = float4(pnoise(float4(0.5), float4(4.0)), 0.0, 0.0, 0.0);
}
kernel void pnoise4Period1(device float4* results [[buffer(0)]]) {
    results[0] = float4(pnoise(float4(0.5) + float4(4.0), float4(4.0)), 0.0, 0.0, 0.0);
}
kernel void pnoise4Period2(device float4* results [[buffer(0)]]) {
    results[0] = float4(pnoise(float4(0.5) + float4(4.0) * 2.0, float4(4.0)), 0.0, 0.0, 0.0);
}
