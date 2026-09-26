// Ported from test/wesl/shaders/color_palette_named.test.wesl (same inputs and expected values).
// The WESL test uses expectNearVec3 (relative 1e-3, absolute 1e-6); here the default
// absolute 0.0001 is used, which is at least as strict for these values (all <= 1).
// WESL exports the named colors as `const`s; in MSL they are #defines (same names).
// Tests comparing two calls (zornWraps, spyderAMatchesSpyder, ...) use @same.
#include <metal_stdlib>
using namespace metal;
#include "lygia/color/palette/zorn.msl"
#include "lygia/color/palette/macbeth.msl"
#include "lygia/color/palette/spyder.msl"
#include "lygia/color/palette/pigments.msl"

// ============ zorn ============

// @test zornWhite
// @expect zornWhite[0] 0.973 0.961 0.929
kernel void zornWhite(device float4* results [[buffer(0)]]) { results[0] = float4(zorn(0), 0.0); }

// @test zornYellowOchre
// @expect zornYellowOchre[0] 0.706 0.486 0.188
kernel void zornYellowOchre(device float4* results [[buffer(0)]]) { results[0] = float4(zorn(1), 0.0); }

// @test zornCadmiumRed
// @expect zornCadmiumRed[0] 1.000 0.153 0.008
kernel void zornCadmiumRed(device float4* results [[buffer(0)]]) { results[0] = float4(zorn(2), 0.0); }

// @test zornIvoryBlack
// @expect zornIvoryBlack[0] 0.180 0.180 0.180
kernel void zornIvoryBlack(device float4* results [[buffer(0)]]) { results[0] = float4(zorn(3), 0.0); }

// [0] = zorn(4), [1] = zorn(0)
// @test zornWraps 2
// @same zornWraps[0] zornWraps[1]
kernel void zornWraps(device float4* results [[buffer(0)]]) {
    results[0] = float4(zorn(4), 0.0);
    results[1] = float4(zorn(0), 0.0);
}

// ============ macbeth ============

// @test macbethDarkSkin
// @expect macbethDarkSkin[0] 0.46017 0.33059 0.27477
kernel void macbethDarkSkin(device float4* results [[buffer(0)]]) { results[0] = float4(macbeth(0), 0.0); }

// @test macbethBlueSky
// @expect macbethBlueSky[0] 0.356 0.472 0.609
kernel void macbethBlueSky(device float4* results [[buffer(0)]]) { results[0] = float4(macbeth(2), 0.0); }

// @test macbethOrange
// @expect macbethOrange[0] 0.867 0.487 0.184
kernel void macbethOrange(device float4* results [[buffer(0)]]) { results[0] = float4(macbeth(6), 0.0); }

// @test macbethYellow
// @expect macbethYellow[0] 0.925 0.784 0.094
kernel void macbethYellow(device float4* results [[buffer(0)]]) { results[0] = float4(macbeth(15), 0.0); }

// @test macbethWhite
// @expect macbethWhite[0] 0.956 0.956 0.945
kernel void macbethWhite(device float4* results [[buffer(0)]]) { results[0] = float4(macbeth(18), 0.0); }

// @test macbethBlack
// @expect macbethBlack[0] 0.200 0.200 0.204
kernel void macbethBlack(device float4* results [[buffer(0)]]) { results[0] = float4(macbeth(23), 0.0); }

// ============ spyder ============

// @test spyderLowSatRed
// @expect spyderLowSatRed[0] 0.824 0.475 0.459
kernel void spyderLowSatRed(device float4* results [[buffer(0)]]) { results[0] = float4(spyder(0), 0.0); }

// @test spyderPrimaryYellow
// @expect spyderPrimaryYellow[0] 0.961 0.804 0.000
kernel void spyderPrimaryYellow(device float4* results [[buffer(0)]]) { results[0] = float4(spyder(38), 0.0); }

// @test spyderCardWhite
// @expect spyderCardWhite[0] 0.976 0.949 0.933
kernel void spyderCardWhite(device float4* results [[buffer(0)]]) { results[0] = float4(spyder(42), 0.0); }

// @test spyderCardBlack
// @expect spyderCardBlack[0] 0.169 0.161 0.169
kernel void spyderCardBlack(device float4* results [[buffer(0)]]) { results[0] = float4(spyder(47), 0.0); }

// [0] = spyderA(5), [1] = spyder(5)
// @test spyderAMatchesSpyder 2
// @same spyderAMatchesSpyder[0] spyderAMatchesSpyder[1]
kernel void spyderAMatchesSpyder(device float4* results [[buffer(0)]]) {
    results[0] = float4(spyderA(5), 0.0);
    results[1] = float4(spyder(5), 0.0);
}

// [0] = spyderB(0), [1] = spyder(24)
// @test spyderBMatchesSpyderOffset 2
// @same spyderBMatchesSpyderOffset[0] spyderBMatchesSpyderOffset[1]
kernel void spyderBMatchesSpyderOffset(device float4* results [[buffer(0)]]) {
    results[0] = float4(spyderB(0), 0.0);
    results[1] = float4(spyder(24), 0.0);
}

// ============ exported constants ============

// @test pigmentsCadmiumRed
// @expect pigmentsCadmiumRed[0] 1.000 0.153 0.008
kernel void pigmentsCadmiumRed(device float4* results [[buffer(0)]]) { results[0] = float4(CADMIUM_RED, 0.0); }

// @test pigmentsTitaniumWhite
// @expect pigmentsTitaniumWhite[0] 0.973 0.961 0.929
kernel void pigmentsTitaniumWhite(device float4* results [[buffer(0)]]) { results[0] = float4(TITANIUM_WHITE, 0.0); }

// [0] = DARK_SKIN, [1] = macbeth(0)
// @test macbethDarkSkinConst 2
// @same macbethDarkSkinConst[0] macbethDarkSkinConst[1]
kernel void macbethDarkSkinConst(device float4* results [[buffer(0)]]) {
    results[0] = float4(DARK_SKIN, 0.0);
    results[1] = float4(macbeth(0), 0.0);
}

// [0] = BLUE_SKY, [1] = macbeth(2)
// @test macbethBlueSkyConst 2
// @same macbethBlueSkyConst[0] macbethBlueSkyConst[1]
kernel void macbethBlueSkyConst(device float4* results [[buffer(0)]]) {
    results[0] = float4(BLUE_SKY, 0.0);
    results[1] = float4(macbeth(2), 0.0);
}

// [0] = LOW_SAT_RED, [1] = spyder(0)
// @test spyderLowSatRedConst 2
// @same spyderLowSatRedConst[0] spyderLowSatRedConst[1]
kernel void spyderLowSatRedConst(device float4* results [[buffer(0)]]) {
    results[0] = float4(LOW_SAT_RED, 0.0);
    results[1] = float4(spyder(0), 0.0);
}

// [0] = PRIMARY_YELLOW, [1] = spyder(38)
// @test spyderPrimaryYellowConst 2
// @same spyderPrimaryYellowConst[0] spyderPrimaryYellowConst[1]
kernel void spyderPrimaryYellowConst(device float4* results [[buffer(0)]]) {
    results[0] = float4(PRIMARY_YELLOW, 0.0);
    results[1] = float4(spyder(38), 0.0);
}
