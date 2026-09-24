// Calls every lighting overload (no env, cubemap, equirect) so the templated
// lighting API gets instantiated. Included by the config files in this folder,
// which set KNAME and the options to test.
// program-scope arrays used through defines must be declared before the includes
constant float3 SH9[9] = { float3(1.0), float3(0.1), float3(0.1), float3(0.1), float3(0.0), float3(0.0), float3(0.0), float3(0.0), float3(0.0) };
// instantiates every template/overload with no env, a cubemap and an equirect
#include "../../../lighting/material/new.msl"
inline Material raymarchMap(float3 p) {
    Material m = materialNew(float3(1.0, 0.2, 0.2), 0.3, 0.0, length(p) - 1.0);
    return m;
}
#if defined(RAYMARCH_VOLUME)
#include "../../../lighting/medium/new.msl"
inline Medium raymarchVolumeMap(float3 p) {
    Medium m = mediumNew();
    m.sdf = length(p) - 1.0;
    return m;
}
#endif
#include "../../../lighting/pbr.msl"
#include "../../../lighting/pbrClearCoat.msl"
#include "../../../lighting/pbrGlass.msl"
#include "../../../lighting/pbrLittle.msl"
#include "../../../lighting/gooch.msl"
#include "../../../lighting/raymarch.msl"
#include "../../../lighting/raymarch/glass.msl"
#include "../../../lighting/transparent.msl"
#include "../../../lighting/atmosphere.msl"
#include "../../../lighting/ssao.msl"
#include "../../../lighting/ssr.msl"
#include "../../../lighting/volumetricLightScattering.msl"
#include "../../../lighting/light/spot.msl"
#include "../../../lighting/iridescence.msl"
#include "../../../lighting/wavelength.msl"
#include "../../../lighting/blackbody.msl"
#include "../../../lighting/sphereMap.msl"
#include "../../../lighting/debugCube.msl"
#include "../../../lighting/exposure.msl"
#include "../../../lighting/specular/ward.msl"
#include "../../../lighting/specular/importanceSampling.msl"
#include "../../../lighting/common/gtaoMultiBounce.msl"
#include "../../../lighting/common/penner.msl"
#include "../../../lighting/common/charlie.msl"
#include "../../../lighting/common/ashikhmin.msl"
#include "../../../lighting/sphericalHarmonics.msl"
#include "../../../lighting/ray/new.msl"
#include "../../../lighting/ray/cast.msl"
#include "../../../lighting/ray/direction.msl"
#include "../../../lighting/camera.msl"
#include "../../../sample/shadowPCF.msl"
#include "../../../sample/triplanar.msl"


kernel void KNAME(texturecube<float> cube [[texture(0)]], texture2d<float> equi [[texture(1)]],
                  texture2d<float> depth [[texture(2)]], constant float3* samples [[buffer(1)]],
                  device float4* o [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    Material mat = materialNew();
    mat.position = float3(0.0, 0.0, 1.0);
    mat.normal = float3(0.0, 0.0, -1.0);
    ShadingData sd = shadingDataNew();
    sd.V = float3(0.0, 0.0, -1.0);
    float4 c = 0.0;
    c += pbr(mat, cube) + pbr(mat, sd, cube);
#ifndef IBL_IMPORTANCE_SAMPLING
    c += pbr(mat) + pbr(mat, sd) + pbr(mat, equi) + pbr(mat, sd, equi);
#endif
#ifndef IBL_IMPORTANCE_SAMPLING
    c += pbrClearCoat(mat) + pbrClearCoat(mat, sd, cube) + pbrClearCoat(mat, equi);
    c += pbrGlass(mat) + pbrGlass(mat, sd, cube) + pbrGlass(mat, equi);
    c += pbrLittle(mat) + pbrLittle(mat, sd, cube) + pbrLittle(mat, equi);
    c += raymarch(float3(0.0, 0.0, -3.0), float3(0.0), float2(0.5));
    c += raymarch(float3(0.0, 0.0, -3.0), float3(0.0), float2(0.5), cube);
    float eyeDepth = 0.0;
    Material res;
    c += raymarch(float3(0.0, 0.0, -3.0), float3(0.0), float2(0.5), eyeDepth, res, equi);
    c += raymarch(lookAtView(float3(0.0, 0.0, -3.0), float3(0.0)), float2(0.5), eyeDepth);
    c += raymarchGlass(float3(0.0, 0.0, 1.0), float3(0.0, 0.0, -3.0), 1.5, 0.1);
    c += raymarchGlass(float3(0.0, 0.0, 1.0), float3(0.0, 0.0, -3.0), 1.5, 0.1, cube);
    c.rgb += transparent(mat.normal, -sd.V, 0.04, float3(0.66), 0.1, cube);
    c.rgb += fresnelIridescentReflection(mat.normal, sd.V, 1.0, 1.5, 300.0, 0.2, cube);
    c.rgb += fresnelIridescentReflection(mat.normal, sd.V, float3(1.0), float3(1.5), 300.0, 0.2);
    c += gooch(mat);
#endif
    c.rgb += atmosphere(normalize(float3(0.0, 0.2, 1.0)), normalize(float3(0.0, 0.1, 1.0)));
    c.rgb += sphericalHarmonics(SH9, mat.normal) + sphericalHarmonics(samples, mat.normal);
    c.r += ssao(depth, float2(0.5), float2(1.0/512.0), 1.0, 0.1, 100.0) + ssao(depth, float2(0.5));
    c.r += ssao(depth, depth, float2(0.5), 1.0, samples, 8, float4x4(1.0));
    float op = 0.5, dist = 0.0;
    c.rg += ssr(depth, depth, float2(0.5), float2(1.0/512.0), op, dist, 0.1, float4x4(1.0));
    c.rgb += ssr(depth, depth, depth, float2(0.5), float2(1.0/512.0), 0.1, float4x4(1.0));
    c.r += volumetricLightScattering(depth, float2(0.5), 0.1, 100.0, float4x4(1.0), float4x4(1.0), float3(0.0), depth, float4x4(1.0), float3(0.0, 1.0, 0.0));
    float3 d = 0.0, s = 0.0;
    lightSpot(float3(1.0), float3(0.04), mat.normal, sd.V, 0.5, 0.3, 0.04, float3(0.0, -1.0, 0.0), 30.0, 5.0, 100.0, d, s);
    c.rgb += d + s + iridescence(0.5, 0.5) + wavelength(550.0) + blackbody(6500.0) + debugCube(mat.normal, 64.0, 1.0);
    c += sphereMap(equi, mat.normal, sd.V) + sampleShadowPCF(depth, float2(512.0), float2(0.5), 0.5) + sampleTriplanar(equi, mat.normal);
    c.r += exposure(16.0, 1.0/125.0, 100.0);
    c.r += specularWard(float3(0,1,0), float3(0,1,0), float3(0,0,1), float3(1,0,0), 0.2, 0.3);
    c.rgb += gtaoMultiBounce(0.5, float3(0.5)) + penner(0.5, 0.5);
    c.r += charlie(0.5, 0.5) + ashikhmin(0.5, 0.5);
    Ray r = rayNew(float3(0.0), float3(0.0, 0.0, 1.0));
    c.rgb += rayCast(r, 2.0) + rayDirection(float3(0.0), float2(0.5), float2(1.0), 1.0).direction;
#ifdef CAMERA_NEAR_CLIP
    c.r += ssao(depth, float2(0.5), float2(1.0/512.0), 1.0) + ssao(depth, depth, float2(0.5), 1.0);
    c.rg += ssr(depth, depth, float2(0.5), float2(1.0/512.0), op) + ssr(depth, depth, float2(0.5), float2(1.0/512.0), op, dist);
    c.rgb += ssr(depth, depth, depth, float2(0.5), float2(1.0/512.0)) + ssr(depth, depth, depth, float2(0.5), float2(1.0/512.0), 0.3);
    c.r += volumetricLightScattering(depth, float2(0.5), depth);
#endif
    o[id] = c;
}
