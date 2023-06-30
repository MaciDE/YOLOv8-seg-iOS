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
    
    let mask: [UInt8] // (64, 64)
    let maskSize: (width: Int, height: Int)
   
    let originalImgSize: CGSize
    
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
            .resized(to: originalImgSize)
    }
}

let colors = (0..<80).map { _ in UIColor.random }

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
                    UInt8(truncating: (0.7 * 255) as NSNumber),
                ]
            )
        }
    }
    
    return coloredMask
}
