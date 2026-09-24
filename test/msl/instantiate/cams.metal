#include <metal_stdlib>
using namespace metal;
#define KNAME k_cams
#define CAMERA_NEAR_CLIP 0.1
#define CAMERA_FAR_CLIP 100.0
#define CAMERA_PROJECTION_MATRIX float4x4(1.0)
#define CAMERA_VIEW_MATRIX float4x4(1.0)
#define CAMERA_POSITION float3(0.0, 0.0, -3.0)
#define LIGHT_MATRIX float4x4(1.0)
#define SSAO_SAMPLES_ARRAY SH9
#define SSAO_NOISE_ARRAY SH9
#define SSAO_NOISE_NUM 4
#define ATMOSPHERE_STARS_LAYERS 3
#define ATMOSPHERE_GROUND float3(0.3)
#include "body.h"
