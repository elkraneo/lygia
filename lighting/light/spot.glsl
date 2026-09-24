/*
contributors: Patricio Gonzalez Vivo
description: Calculate spot light
use: lightSpot(<vec3> _diffuseColor, <vec3> _specularColor, <vec3> _N, <vec3> _V, <float> _NoV, <float> _roughness, <float> _f0, out <vec3> _diffuse, out <vec3> _specular)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SURFACE_POSITION: in glslViewer is v_position
    - LIGHT_POSITION: in glslViewer is u_light
    - LIGHT_COLOR: in glslViewer is u_lightColor
    - LIGHT_INTENSITY: in glslViewer is  u_lightIntensity
    - LIGHT_SPOT_DIRECTION: direction the spot light points to. Default -LIGHT_POSITION (towards the origin)
    - LIGHT_SPOT_CUTOFF: angle of the spot light cone in degrees. Default 45.0
    - LIGHT_SPOT_FACTOR: angle in degrees of the soft edge inside the cone. Default 10.0
    - LIGHT_FALLOFF: distance falloff radius. If not defined there is no falloff
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#include "../specular.glsl"
#include "../diffuse.glsl"
#include "falloff.glsl"
#include "../../math/saturate.glsl"

#ifndef SURFACE_POSITION
#define SURFACE_POSITION vec3(0.0, 0.0, 0.0)
#endif

#ifndef LIGHT_POSITION
#define LIGHT_POSITION vec3(0.0, 10.0, -50.0)
#endif

#ifndef LIGHT_COLOR
#define LIGHT_COLOR vec3(0.5)
#endif

#ifndef LIGHT_INTENSITY
#define LIGHT_INTENSITY 1.0
#endif

#ifndef LIGHT_SPOT_DIRECTION
#define LIGHT_SPOT_DIRECTION (-(LIGHT_POSITION))
#endif

#ifndef LIGHT_SPOT_CUTOFF
#define LIGHT_SPOT_CUTOFF 45.0
#endif

#ifndef LIGHT_SPOT_FACTOR
#define LIGHT_SPOT_FACTOR 10.0
#endif

#ifndef FNC_LIGHT_SPOT
#define FNC_LIGHT_SPOT

void lightSpot(const in vec3 _diffuseColor, const in vec3 _specularColor, const in vec3 _N, const in vec3 _V, const in float _NoV, const in float _roughness, const in float _f0, out vec3 _diffuse, out vec3 _specular) {
    vec3 toLight = LIGHT_POSITION - (SURFACE_POSITION).xyz;
    float toLightLength = length(toLight);
    vec3 s = toLight/toLightLength;

    float angle = acos(dot(-s, normalize(LIGHT_SPOT_DIRECTION)));
    float cutoff1 = radians(clamp(LIGHT_SPOT_CUTOFF - max(LIGHT_SPOT_FACTOR, 0.01), 0.0, 89.9));
    float cutoff2 = radians(clamp(LIGHT_SPOT_CUTOFF, 0.0, 90.0));
    if (angle < cutoff2) {
        ShadingData shadingData = shadingDataNew();
        shadingData.L = s;
        shadingData.N = _N;
        shadingData.V = _V;
        shadingData.H = normalize(s + _V);
        shadingData.NoV = _NoV;
        shadingData.NoL = saturate(dot(_N, s));
        shadingData.NoH = saturate(dot(_N, shadingData.H));
        shadingData.roughness = _roughness;
        shadingData.linearRoughness = perceptual2linearRoughness(_roughness);
        shadingData.specularColor = vec3(_f0);

        float dif = diffuse(shadingData);
        #ifdef LIGHT_FALLOFF
        float fall = falloff(toLightLength, LIGHT_FALLOFF);
        #else
        float fall = 1.0;
        #endif
        vec3 spec = specular(shadingData);
        _diffuse = LIGHT_INTENSITY * (_diffuseColor * LIGHT_COLOR * dif * fall) * smoothstep(cutoff2, cutoff1, angle);
        _specular = LIGHT_INTENSITY * (_specularColor * LIGHT_COLOR * spec * fall) * smoothstep(cutoff2, cutoff1, angle);
    }
    else {
        _diffuse = vec3(0.0);
        _specular = vec3(0.0);
    }
}

#endif