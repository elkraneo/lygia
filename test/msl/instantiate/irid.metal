#include <metal_stdlib>
using namespace metal;
#define KNAME k_irid
#define SHADING_MODEL_IRIDESCENCE
#define SHADING_MODEL_SUBSURFACE
#define LIGHT_POSITION float3(0.0, 10.0, -10.0)
#define SCENE_BACK_SURFACE
#define TRANSPARENT_DISPERSION 0.05
#define MATERIAL_ANISOTROPY 0.5
#define MATERIAL_ANISOTROPY_DIRECTION float3(1.0, 0.0, 0.0)
#include "body.h"
