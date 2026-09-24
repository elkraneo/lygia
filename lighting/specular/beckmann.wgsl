#include "../common/beckmann.wgsl"
#include "../shadingData/shadingData.wgsl"

fn specularBeckmann(shadingData: ShadingData) -> f32 {
    return beckmann(shadingData.NoH, shadingData.roughness);
}
