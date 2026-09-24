#include <metal_stdlib>
using namespace metal;
#define KNAME k_lightdir
#define LIGHT_DIRECTION float3(0.0, 1.0, -1.0)
#define RAYMARCH_SHADING_FNC pbr
#define RAYMARCH_AOV
#include "body.h"
