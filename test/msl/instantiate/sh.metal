#include <metal_stdlib>
using namespace metal;
#define KNAME k_sh
#define SCENE_SH_ARRAY SH9
#define RAYMARCH_MULTISAMPLE 4
#define RESOLUTION float2(512.0)
#define RAYMARCH_VOLUME
#define RAYMARCH_AOV
#include "body.h"
