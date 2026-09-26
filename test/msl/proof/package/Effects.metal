#include <metal_stdlib>
#include <SwiftUI/SwiftUI_Metal.h>
using namespace metal;
#include "../lygia/generative/fbm.msl"
[[ stitchable ]] half4 lygiaFbm(float2 position, half4 color, float2 size, float time) {
    float n = fbm(float3(position / size * 4.0, time));
    return half4(half3(n * 0.5 + 0.5), 1.0);
}
