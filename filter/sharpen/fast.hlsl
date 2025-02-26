/*
original_author: Johan Ismael
description: sharpening convolutional operation
use: sharpen(<sampler2D> texture, <float2> st, <float2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SHARPENFAST_KERNELSIZE: Defaults 2
    - SHARPENFAST_TYPE: defaults to vec3
    - SHARPENFAST_SAMPLER_FNC(POS_UV): defaults to texture2D(tex, POS_UV).rgb
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) tex2D(TEX, UV)
#endif

#ifndef SHARPENFAST_KERNELSIZE
#ifdef SHARPEN_KERNELSIZE
#define SHARPENFAST_KERNELSIZE SHARPEN_KERNELSIZE
#else
#define SHARPENFAST_KERNELSIZE 2
#endif
#endif

#ifndef SHARPENFAST_TYPE
#ifdef SHARPEN_TYPE
#define SHARPENFAST_TYPE SHARPEN_TYPE
#else
#define SHARPENFAST_TYPE float3
#endif
#endif

#ifndef SHARPENFAST_SAMPLER_FNC
#ifdef SHARPEN_SAMPLER_FNC
#define SHARPENFAST_SAMPLER_FNC(POS_UV) SHARPEN_SAMPLER_FNC(POS_UV)
#else
#define SHARPENFAST_SAMPLER_FNC(POS_UV) SAMPLER_FNC(tex, POS_UV).rgb
#endif
#endif

#ifndef FNC_SHARPENFAST
#define FNC_SHARPENFAST
SHARPENFAST_TYPE sharpenFast(in sampler2D tex, in float2 coords, in float2 pixel, float strenght) {
    SHARPENFAST_TYPE sum = float4(0.0,0.0,0.0,0.0);
    for (int i = 0; i < SHARPENFAST_KERNELSIZE; i++) {
        float f_size = float(i) + 1.;
        f_size *= strenght;
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + float2( -1., 0.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + float2( 0., -1.) * pixel * f_size);
        sum +=  5. * SHARPENFAST_SAMPLER_FNC(coords + float2( 0., 0.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + float2( 0., 1.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + float2( 1., 0.) * pixel * f_size);
    }
    return sum / float(SHARPENFAST_KERNELSIZE);
}

SHARPENFAST_TYPE sharpenFast(in sampler2D tex, in float2 coords, in float2 pixel) {
    SHARPENFAST_TYPE sum = float4(0.0,0.0,0.0,0.0);
    for (int i = 0; i < SHARPENFAST_KERNELSIZE; i++) {
        float f_size = float(i) + 1.;
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + float2( -1., 0.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + float2( 0., -1.) * pixel * f_size);
        sum +=  5. * SHARPENFAST_SAMPLER_FNC(coords + float2( 0., 0.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + float2( 0., 1.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + float2( 1., 0.) * pixel * f_size);
    }
    return sum / float(SHARPENFAST_KERNELSIZE);
}
#endif