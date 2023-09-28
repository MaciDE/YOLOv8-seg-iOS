//
//  MaskPrediction.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 25.06.23.
//

import Foundation

// MARK: MaskPrediction
struct MaskPrediction: Identifiable {
    let id = UUID()
    
    let classIndex: Int
    
    let mask: [UInt8]
    let maskSize: (width: Int, height: Int)
    
    func getMaskImage() -> UIImage? {
        guard !mask.isEmpty else { return nil }
        
        let coloredMask = colorizeMask(mask, color: colors[classIndex])
        
        let numComponents = 4
        let numBytes = maskSize.width * maskSize.height * numComponents
        
        let colorspace = CGColorSpaceCreateDeviceRGB()
        let rgbData = CFDataCreate(nil, coloredMask, numBytes)!
        let provider = CGDataProvider(data: rgbData)!
        guard let rgbImageRef = CGImage(
            width: maskSize.width,
            height: maskSize.height,
            bitsPerComponent: 8,
            bitsPerPixel: 8 * numComponents,
            bytesPerRow: maskSize.width * numComponents,
            space: colorspace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: CGColorRenderingIntent(rawValue: 0)!
        ) else { return nil }
        
        return UIImage(cgImage: rgbImageRef)
    }
}

let colors: [UIColor] = {
    var colors = [
        UIColor(hex: "#FF3838FF")!,
        UIColor(hex: "#FF9D97FF")!,
        UIColor(hex: "#FF701FFF")!,
        UIColor(hex: "#FFB21DFF")!,
        UIColor(hex: "#CFD231FF")!,
        UIColor(hex: "#48F90AFF")!,
        UIColor(hex: "#92CC17FF")!,
        UIColor(hex: "#3DDB86FF")!,
        UIColor(hex: "#1A9334FF")!,
        UIColor(hex: "#00D4BBFF")!,
        UIColor(hex: "#2C99A8FF")!,
        UIColor(hex: "#00C2FFFF")!,
        UIColor(hex: "#344593FF")!,
        UIColor(hex: "#6473FFFF")!,
        UIColor(hex: "#0018ECFF")!,
    ]
    colors.append(contentsOf: (0..<65).map { _ in UIColor.random })
    return colors
}()

func colorizeMask(_ mask: [UInt8], color: UIColor) -> [UInt8] {
    var red: CGFloat = 0
    var green: CGFloat = 0
    var blue: CGFloat = 0
    var alpha: CGFloat = 0
    
    color.getRed(&red, green: &green, blue: &blue, alpha: &alpha)
    
    var coloredMask: [UInt8] = []
    for value in mask {
        if value == 0 {
            coloredMask.append(contentsOf: [0, 0, 0, 0])
        } else {
            coloredMask.append(
                contentsOf: [
                    UInt8(truncating: (red * 255) as NSNumber),
                    UInt8(truncating: (green * 255) as NSNumber),
                    UInt8(truncating: (blue * 255) as NSNumber),
                    255,
                ]
            )
        }
    }
    
    return coloredMask
}

extension Collection where Element == MaskPrediction {
    func combineToSingleImage() -> UIImage? {
        guard let firstMask = self.first else { return nil }
        
        let size = CGSize(width: firstMask.maskSize.width, height: firstMask.maskSize.height)
        UIGraphicsBeginImageContext(size)
        
        let areaSize = CGRect(x: 0, y: 0, width: size.width, height: size.height)
        firstMask.getMaskImage()?.draw(in: areaSize)
        
        for mask in self.dropFirst() {
            mask.getMaskImage()?.draw(in: areaSize, blendMode: .normal, alpha: 1.0)
        }
        
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return newImage
    }
}
