# LYGIA for Metal, as a Swift package

```swift
.package(url: "https://github.com/elkraneo/lygia.git", branch: "metal/lighting")
```

The `Lygia` library bundles every `.msl` file and gives you two ways to use them.

## At runtime (any platform, no build settings)

Compile Metal source that includes LYGIA with `Lygia.makeLibrary`. The `#include "lygia/..."` lines are inlined from the bundle before the Metal compiler sees them, since it can't read files itself.

```swift
import Lygia

let source = """
#include "lygia/generative/fbm.msl"

kernel void clouds(texture2d<float, access::write> out [[texture(0)]],
                   uint2 gid [[thread_position_in_grid]]) {
    float2 st = float2(gid) / float2(out.get_width(), out.get_height());
    out.write(float4(float3(fbm(st * 4.0) * 0.5 + 0.5), 1.0), gid);
}
"""
let library = try await Lygia.makeLibrary(source: source, device: device, defines: ["FBM_OCTAVES": "6"])
let clouds = library.makeFunction(name: "clouds")
```

`Lygia.flatten(source)` gives you the inlined source if you'd rather compile it yourself (for a SwiftUI `ShaderLibrary`, say), and `Lygia.files` lists the bundled modules.

## At build time (`.metal` files in your own target)

SwiftPM can't pass include paths to the Metal compiler, so point it at the bundled headers yourself, in your app target's build settings:

```
MTL_HEADER_SEARCH_PATHS = $(BUILT_PRODUCTS_DIR)/Lygia_Lygia.bundle/Contents/Resources
```

On iOS, tvOS and visionOS the bundle has no `Contents/Resources`, so use `$(BUILT_PRODUCTS_DIR)/Lygia_Lygia.bundle`. Then `#include "lygia/sdf/circleSDF.msl"` works in any `.metal` file of the target, and the functions end up in your `default.metallib`.

Or skip the package for shaders and copy the `lygia` folder (`Lygia.includePath`) into your project.

## Several `.metal` files, one library

Every LYGIA function is `static inline` by default (`LYGIA_FNC`, in `lygia.msl`), so several `.metal` files in one target can include the same file, and each keeps its own options such as `FBM_OCTAVES`. To build LYGIA once as a helper library that other files call into, define `LYGIA_FNC` as empty before including it; the functions are exported, and the callers declare the ones they use.

## Keeping the copy in sync

The package's resources are a plain copy of the repository's `.msl` files in `swift/Sources/Lygia/Resources/lygia`, because SwiftPM copies a resource folder as-is. After changing a `.msl` file, run `swift/sync.sh`; CI runs `swift/sync.sh --check` and `swift test`.
