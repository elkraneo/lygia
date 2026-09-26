import SwiftUI
public enum Lygia {
    public static let library = ShaderLibrary.bundle(.module)
    public static func fbm(size: CGSize, time: Float) -> Shader {
        library.lygiaFbm(.float2(size), .float(time))
    }
}
