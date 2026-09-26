// Ported from test/wesl/color-layer-visual.test.ts.
// Every WGSL test in this file is an image snapshot of a layer*SourceOver4 blend over two textures
// (expectBlend + blendInputs sampling src/dst textures), so none can run here. Skipped:
//   hardLight blend mode (layerHardLightSourceOver4; fragment render + image snapshot)
//   softLight blend mode (layerSoftLightSourceOver4; fragment render + image snapshot)
//   vividLight blend mode (layerVividLightSourceOver4; fragment render + image snapshot)
//   linearLight blend mode (layerLinearLightSourceOver4; fragment render + image snapshot)
//   pinLight blend mode (layerPinLightSourceOver4; fragment render + image snapshot)
//   hardMix blend mode (layerHardMixSourceOver4; fragment render + image snapshot)
//   colorBurn blend mode (layerColorBurnSourceOver4; fragment render + image snapshot)
//   linearBurn blend mode (layerLinearBurnSourceOver4; fragment render + image snapshot)
//   colorDodge blend mode (layerColorDodgeSourceOver4; fragment render + image snapshot)
//   linearDodge blend mode (layerLinearDodgeSourceOver4; fragment render + image snapshot)
//   color blend mode (layerColorSourceOver4; fragment render + image snapshot)
//   hue blend mode (layerHueSourceOver4; fragment render + image snapshot)
//   saturation blend mode (layerSaturationSourceOver4; fragment render + image snapshot)
//   luminosity blend mode (layerLuminositySourceOver4; fragment render + image snapshot)
//   average blend mode (layerAverageSourceOver4; fragment render + image snapshot)
//   negation blend mode (layerNegationSourceOver4; fragment render + image snapshot)
//   reflect blend mode (layerReflectSourceOver4; fragment render + image snapshot)
//   glow blend mode (layerGlowSourceOver4; fragment render + image snapshot)
//
// Not from the WGSL test (it has no numeric expectations): a smoke kernel that compiles every MSL
// layerXSourceOver(float4, float4) used above (WGSL layerXSourceOver4) and evaluates it on a 5x5 grid of
// interior src/dst values; the harness fails on any NaN/inf. No @expect, so it reports 0 checks.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/layer/hardLightSourceOver.msl"
#include "lygia/color/layer/softLightSourceOver.msl"
#include "lygia/color/layer/vividLightSourceOver.msl"
#include "lygia/color/layer/linearLightSourceOver.msl"
#include "lygia/color/layer/pinLightSourceOver.msl"
#include "lygia/color/layer/hardMixSourceOver.msl"
#include "lygia/color/layer/colorBurnSourceOver.msl"
#include "lygia/color/layer/linearBurnSourceOver.msl"
#include "lygia/color/layer/colorDodgeSourceOver.msl"
#include "lygia/color/layer/linearDodgeSourceOver.msl"
#include "lygia/color/layer/colorSourceOver.msl"
#include "lygia/color/layer/hueSourceOver.msl"
#include "lygia/color/layer/saturationSourceOver.msl"
#include "lygia/color/layer/luminositySourceOver.msl"
#include "lygia/color/layer/averageSourceOver.msl"
#include "lygia/color/layer/negationSourceOver.msl"
#include "lygia/color/layer/reflectSourceOver.msl"
#include "lygia/color/layer/glowSourceOver.msl"

// @test layer_smoke 450
kernel void layer_smoke(device float4* results [[buffer(0)]]) {
    uint k = 0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            float a = 0.1 + 0.2 * float(i), b = 0.1 + 0.2 * float(j);
            float4 src = float4(a, b, 1.0 - a, 0.3 + 0.1 * float(j));
            float4 dst = float4(1.0 - b, a, b, 0.4 + 0.1 * float(i));
            results[k++] = layerHardLightSourceOver(src, dst);
            results[k++] = layerSoftLightSourceOver(src, dst);
            results[k++] = layerVividLightSourceOver(src, dst);
            results[k++] = layerLinearLightSourceOver(src, dst);
            results[k++] = layerPinLightSourceOver(src, dst);
            results[k++] = layerHardMixSourceOver(src, dst);
            results[k++] = layerColorBurnSourceOver(src, dst);
            results[k++] = layerLinearBurnSourceOver(src, dst);
            results[k++] = layerColorDodgeSourceOver(src, dst);
            results[k++] = layerLinearDodgeSourceOver(src, dst);
            results[k++] = layerColorSourceOver(src, dst);
            results[k++] = layerHueSourceOver(src, dst);
            results[k++] = layerSaturationSourceOver(src, dst);
            results[k++] = layerLuminositySourceOver(src, dst);
            results[k++] = layerAverageSourceOver(src, dst);
            results[k++] = layerNegationSourceOver(src, dst);
            results[k++] = layerReflectSourceOver(src, dst);
            results[k++] = layerGlowSourceOver(src, dst);
        }
    }
}
