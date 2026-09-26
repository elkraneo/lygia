# METAL version

Metal support is currently highly experimental and very work in progress.

Every `*.msl` file is compiled by CI, both on its own and all together in a single translation unit, and two such units are linked together to catch duplicate symbols. The lighting test configurations are also checked for functions with external linkage. Run the same check locally on macOS with:

```sh
test/msl/compile.sh               # every *.msl file
test/msl/compile.sh sdf/*.msl     # just some
```

`test/msl/verify.sh` runs the cases in `test/msl/verify/cases` on the GPU. They're ported from the WGSL tests in `test/wesl`, with the same inputs and expected values, so Metal is checked against the same numbers as WGSL. Where WGSL itself differs from GLSL, the check is marked `@xfail` with the reason. Tests that need textures or image snapshots aren't ported yet.

Where upstream fixed a bug only in the WESL files, Metal follows the fix: `rgb2xyz` returns XYZ in 0-100 (#271), `hueShiftRYB` uses its angle, `rgb2lms(float4)` converts, and `fisheye2xyz` handles the center. GLSL still has these bugs.

LYGIA files don't include the Metal standard library themselves, so include it before any of them:

```cpp
#include <metal_stdlib>
using namespace metal;

#include "lygia/sdf/starSDF.msl"
#include "lygia/space/kaleidoscope.msl"
```

## Porting Progress

- [x] Animation
- [x] Blend
- [x] Color
  - [x] Blend
  - [x] Dither (not fully vetted / just spot checked)
  - [x] Palette
  - [x] Levels
  - [x] Space
  - [x] Tonemap
- [x] Distort
- [x] Draw - (not fully vetted / just spot checked)
- [x] Filters
- [x] Generative (not fully vetted / just spot checked)
- [x] Geometry
- [x] Lighting (not fully vetted / spot checked and render tested)
- [x] Math - (not fully vetted / just spot checked)
- [x] Morphological
- [x] Sample
- [x] Sampler
- [x] SDF - (not fully vetted / just spot checked)
- [x] Simulate
- [x] Space

## Porting Methodology

`test/msl/glsl2msl.py path/to/file.glsl` does a first pass of the steps below. Always review its output and run `test/msl/compile.sh` on it.

- dupe `*.glsl` files -> and rename them to `*.msl`
- find replace `.glsl` -> `.msl` and ensure you repeat the above for imports
- find replace `vec2` -> `float2`
- find replace `vec3` -> `float3`
- find replace `vec4` -> `float4`
- find replace `matN` -> `floatNxN`. Avoid spelling it `matrix<float, N, N>`: `draw/matrix.msl` defines a `matrix()` function, which makes that spelling ambiguous
- find replace two-argument `atan(y, x)` -> `atan2(y, x)`
- find replace `in ` function argument keyword -> `` as metal doesn't have the in function keyword
- find `inout` and determine which thread local memory keyword should replace it, and make it a reference
- ensure `const` is only used within functions, `constant` must be used for global scoped constants
- make sure every `#ifndef FNC_*` include guard is followed by its `#define`
- mark every function `static inline` (`test/msl/add_inline.py` does it). Without it, an app with two `.metal` files that include the same LYGIA file fails to link with duplicate symbols. Plain `inline` links, but if the files include a function with different options (e.g. `FBM_OCTAVES`), both end up using one definition; `static` gives each file its own copy, with no extra GPU code
- rename anything that collides with a Metal reserved word or built-in: the `char()` function is `drawChar()`, and local variables named `kernel` are `kern`. Don't redefine functions Metal already has, like `atan2` or `transpose`
- `dFdx`/`dFdy`/`fwidth` are `dfdx`/`dfdy`/`fwidth`, and `discard` is `discard_fragment()`. These only work in fragment functions, so say so in the description of files that use them

## Things to look out for

- Metal does not have the same basic math functions signatures as GLSL. We are adding all the polyfill functions in the `math/` folder.
- Texture precision and filtering.
  - Added `SAMPLER_TYPE` which specifies the texture precisions. Defaults to `texture2d<float>`
  - This means your texture definition must match the default `float` precision, or you will need to override `SAMPLER_TYPE`
  - Added `SAMPLER` which specifies the Metal sampler object. Defaults to `sampler( min_filter::linear, mag_filter::linear )`

- Metal already has a native `atan2(y, x)`, equivalent to GLSL's `atan(y, x)`. LYGIA's GLSL `atan2` (with a 0 to TAU range) can't be redefined without making every call ambiguous, so `math/atan2.msl` only documents it.
- Metal has no global uniforms, textures or `gl_FragCoord`, and library functions can't see the entry point's arguments. So:
  - functions that default to `gl_FragCoord` in GLSL (e.g. dithering) only have overloads taking the coordinate explicitly. Pass the `[[position]]` coords from your main shader.
  - functions that default to a uniform in GLSL (e.g. `u_projectionMatrix`) take it as an explicit argument. The overload without it only exists when you `#define` the option (e.g. `CAMERA_PROJECTION_MATRIX`) to something in scope. See `space/view2screenPosition.msl`.

## Lighting

- Light options like `LIGHT_POSITION` or `LIGHT_COLOR` still work: define them to literals, `constant` values or function constants before including.
- Scene textures can't be globals, so the functions that sample the environment take it as an optional last argument, either a `texturecube<float>` or an equirect `SAMPLER_TYPE`. Without one they fall back to `fakeCube`, as GLSL does without `SCENE_CUBEMAP`:

```cpp
float4 color = pbr(mat);                 // no environment texture
float4 color = pbr(mat, shadingData, cubemap);
float4 color = raymarch(camera, target, st, cubemap);
```

- `sphericalHarmonics` takes the coefficients (`constant float3* sh`). Shadow maps are passed to `shadow(...)`, and you multiply a light's intensity by the result yourself.
- `ssao`, `ssr` and `volumetricLightScattering` take the camera values (near/far, matrices, sample arrays) as arguments. The overloads without them need the matching `CAMERA_*` options.
- For raymarching, define `static inline Material raymarchMap(float3 pos)` (and `static inline Medium raymarchVolumeMap(float3 pos)` with `RAYMARCH_VOLUME`). They're `static` so each `.metal` file can have its own scene.
- `test/msl/instantiate/` calls every lighting overload under several option sets, since templates are only checked when used. `compile.sh` builds and links them.
