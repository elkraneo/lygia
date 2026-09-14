#include "random.glsl"
#include "srandom.glsl"
#include "../math/cubic.glsl"
#include "../math/quintic.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Gradient Noise
use: gnoise(<float> x)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef GNOISE_NOISE_FNC
#define GNOISE_NOISE_FNC(UV) random(UV)
#endif

#ifndef GNOISE_NOISE2_FNC
#define GNOISE_NOISE2_FNC(UV) GNOISE_NOISE_FNC(UV)
#endif

#ifndef GNOISE_NOISE3_FNC
#define GNOISE_NOISE3_FNC(UV) GNOISE_NOISE_FNC(UV)
#endif

#ifndef GNOISE_NOISE_TILABLE_FNC
#define GNOISE_NOISE_TILABLE_FNC(UV, TILE) srandom3(UV, TILE)
#endif

#ifndef FNC_GNOISE
#define FNC_GNOISE

float gnoise(float x) {
    float i = floor(x);  // integer
    float f = fract(x);  // fraction
    return mix(GNOISE_NOISE_FNC(i), GNOISE_NOISE_FNC(i + 1.0), smoothstep(0.,1.,f)); 
}

float gnoise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    float a = GNOISE_NOISE2_FNC(i);
    float b = GNOISE_NOISE2_FNC(i + vec2(1.0, 0.0));
    float c = GNOISE_NOISE2_FNC(i + vec2(0.0, 1.0));
    float d = GNOISE_NOISE2_FNC(i + vec2(1.0, 1.0));
    vec2 u = cubic(f);
    return mix( a, b, u.x) +
                (c - a)* u.y * (1.0 - u.x) +
                (d - b) * u.x * u.y;
}

float gnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = quintic(f);
    return -1.0 + 2.0 * mix( mix( mix( GNOISE_NOISE3_FNC(i + vec3(0.0,0.0,0.0)), 
                                        GNOISE_NOISE3_FNC(i + vec3(1.0,0.0,0.0)), u.x),
                                mix( GNOISE_NOISE3_FNC(i + vec3(0.0,1.0,0.0)), 
                                        GNOISE_NOISE3_FNC(i + vec3(1.0,1.0,0.0)), u.x), u.y),
                            mix( mix( GNOISE_NOISE3_FNC(i + vec3(0.0,0.0,1.0)), 
                                        GNOISE_NOISE3_FNC(i + vec3(1.0,0.0,1.0)), u.x),
                                mix( GNOISE_NOISE3_FNC(i + vec3(0.0,1.0,1.0)), 
                                        GNOISE_NOISE3_FNC(i + vec3(1.0,1.0,1.0)), u.x), u.y), u.z );
}

float gnoise(vec3 p, float tileLength) {
    vec3 i = floor(p);
    vec3 f = fract(p);
            
    vec3 u = quintic(f);
        
    return mix( mix( mix( dot( GNOISE_NOISE_TILABLE_FNC(i + vec3(0.0,0.0,0.0), tileLength), f - vec3(0.0,0.0,0.0)), 
                            dot( GNOISE_NOISE_TILABLE_FNC(i + vec3(1.0,0.0,0.0), tileLength), f - vec3(1.0,0.0,0.0)), u.x),
                    mix( dot( GNOISE_NOISE_TILABLE_FNC(i + vec3(0.0,1.0,0.0), tileLength), f - vec3(0.0,1.0,0.0)), 
                            dot( GNOISE_NOISE_TILABLE_FNC(i + vec3(1.0,1.0,0.0), tileLength), f - vec3(1.0,1.0,0.0)), u.x), u.y),
                mix( mix( dot( GNOISE_NOISE_TILABLE_FNC(i + vec3(0.0,0.0,1.0), tileLength), f - vec3(0.0,0.0,1.0)), 
                            dot( GNOISE_NOISE_TILABLE_FNC(i + vec3(1.0,0.0,1.0), tileLength), f - vec3(1.0,0.0,1.0)), u.x),
                    mix( dot( GNOISE_NOISE_TILABLE_FNC(i + vec3(0.0,1.0,1.0), tileLength), f - vec3(0.0,1.0,1.0)), 
                            dot( GNOISE_NOISE_TILABLE_FNC(i + vec3(1.0,1.0,1.0), tileLength), f - vec3(1.0,1.0,1.0)), u.x), u.y), u.z );
}

vec3 gnoise3(vec3 x) {
    return vec3(gnoise(x+vec3(123.456, 0.567, 0.37)),
                gnoise(x+vec3(0.11, 47.43, 19.17)),
                gnoise(x) );
}

#endif