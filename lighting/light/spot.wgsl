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

#include "../specular.wgsl"
#include "../diffuse.wgsl"
#include "falloff.wgsl"

// #define SURFACE_POSITION vec3(0.0, 0.0, 0.0)

// #define LIGHT_POSITION vec3(0.0, 10.0, -50.0)

// #define LIGHT_COLOR vec3(0.5)

const LIGHT_INTENSITY: f32 = 1.0;

// #define LIGHT_SPOT_DIRECTION (-(LIGHT_POSITION))

const LIGHT_SPOT_CUTOFF: f32 = 45.0;

const LIGHT_SPOT_FACTOR: f32 = 10.0;

fn lightSpot(_diffuseColor: vec3f, _specularColor: vec3f, _N: vec3f, _V: vec3f, _NoV: f32, _roughness: f32, _f0: f32, _diffuse: vec3f, _specular: vec3f) {
    let toLight = LIGHT_POSITION - (SURFACE_POSITION).xyz;
    let toLightLength = length(toLight);
    let s = toLight/toLightLength;

    let angle = acos(dot(-s, normalize(LIGHT_SPOT_DIRECTION)));
    let cutoff1 = radians(clamp(LIGHT_SPOT_CUTOFF - max(LIGHT_SPOT_FACTOR, 0.01), 0.0, 89.9));
    let cutoff2 = radians(clamp(LIGHT_SPOT_CUTOFF, 0.0, 90.0));
    if (angle < cutoff2) {
        var shadingData = shadingDataNew();
        shadingData.L = s;
        shadingData.N = _N;
        shadingData.V = _V;
        shadingData.H = normalize(s + _V);
        shadingData.NoV = _NoV;
        shadingData.NoL = saturate(dot(_N, s));
        shadingData.NoH = saturate(dot(_N, shadingData.H));
        shadingData.roughness = _roughness;
        shadingData.linearRoughness = perceptual2linearRoughness(_roughness);
        shadingData.specularColor = vec3f(_f0);

        let dif = diffuse(shadingData);
        let fall = falloff(toLightLength, LIGHT_FALLOFF);
        let spec = specular(shadingData);
        _diffuse = LIGHT_INTENSITY * (_diffuseColor * LIGHT_COLOR * dif * fall) * smoothstep(cutoff2, cutoff1, angle);
        _specular = LIGHT_INTENSITY * (_specularColor * LIGHT_COLOR * spec * fall) * smoothstep(cutoff2, cutoff1, angle);
    }
    else {
        _diffuse = vec3f(0.0);
        _specular = vec3f(0.0);
    }
}
