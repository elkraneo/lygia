// Ported from test/wesl/shaders/color_palette_spectral.test.wesl (same inputs and expected values).
// The WESL test uses expectNearVec3 (relative 1e-3, absolute 1e-6); here the default
// absolute 0.0001 is used. That is stricter for most values; for the out-of-gamut geoffrey
// values (up to -7.13) WESL allows ~0.007, and the expected values are given to 4 decimals.
// WGSL spectralExpand(x, l) == MSL spectral(float x, float l).
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/palette/spectral.msl"

// @test geoffreyMid
// @expect geoffreyMid[0] 0.4275 0.9819 0.4275
kernel void geoffreyMid(device float4* results [[buffer(0)]]) { results[0] = float4(spectral_geoffrey(0.5), 0.0); }

// @test geoffreyViolet
// @expect geoffreyViolet[0] -7.1325 -3.8061 -0.8325
kernel void geoffreyViolet(device float4* results [[buffer(0)]]) { results[0] = float4(spectral_geoffrey(0.0), 0.0); }

// @test geoffreyRed
// @expect geoffreyRed[0] -0.8325 -3.0501 -7.1325
kernel void geoffreyRed(device float4* results [[buffer(0)]]) { results[0] = float4(spectral_geoffrey(1.0), 0.0); }

// @test softMid
// @expect softMid[0] 0.5931 1.0 0.5931
kernel void softMid(device float4* results [[buffer(0)]]) { results[0] = float4(spectral_soft(0.5), 0.0); }

// @test softStart
// @expect softStart[0] 0.0528 0.0 0.0528
kernel void softStart(device float4* results [[buffer(0)]]) { results[0] = float4(spectral_soft(0.0), 0.0); }

// @test gemsMid
// @expect gemsMid[0] 0.0 1.0 0.0
kernel void gemsMid(device float4* results [[buffer(0)]]) { results[0] = float4(spectral_gems(0.5), 0.0); }

// @test gemsQuarter
// @expect gemsQuarter[0] 0.0 0.0 1.0
kernel void gemsQuarter(device float4* results [[buffer(0)]]) { results[0] = float4(spectral_gems(0.25), 0.0); }

// @test gemsThreeQuarter
// @expect gemsThreeQuarter[0] 1.0 0.0 0.0
kernel void gemsThreeQuarter(device float4* results [[buffer(0)]]) { results[0] = float4(spectral_gems(0.75), 0.0); }

// @test zucconiMid
// @expect zucconiMid[0] 0.4964 0.8404 0.2163
kernel void zucconiMid(device float4* results [[buffer(0)]]) { results[0] = float4(spectral_zucconi(0.5), 0.0); }

// @test zucconiRed
// @expect zucconiRed[0] 0.9765 0.4925 0.0
kernel void zucconiRed(device float4* results [[buffer(0)]]) { results[0] = float4(spectral_zucconi(0.7), 0.0); }

// @test zucconi6Mid
// @expect zucconi6Mid[0] 0.4964 0.8472 0.1837
kernel void zucconi6Mid(device float4* results [[buffer(0)]]) { results[0] = float4(spectral_zucconi6(0.5), 0.0); }

// @test zucconi6Blue
// @expect zucconi6Blue[0] 0.0 0.5299 0.4708
kernel void zucconi6Blue(device float4* results [[buffer(0)]]) { results[0] = float4(spectral_zucconi6(0.3), 0.0); }

// @test spectralMid
// @expect spectralMid[0] 0.6734 0.9020 0.2002
kernel void spectralMid(device float4* results [[buffer(0)]]) { results[0] = float4(spectral(0.5), 0.0); }

// @test spectralExpandMid
// @expect spectralExpandMid[0] 0.8044 1.0 0.8044
kernel void spectralExpandMid(device float4* results [[buffer(0)]]) { results[0] = float4(spectral(0.5, 0.5), 0.0); }
