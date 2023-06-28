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
        
        let numComponents = 1
        let numBytes = maskSize.width * maskSize.height * numComponents
        
        let colorspace = CGColorSpaceCreateDeviceGray()
        let rgbData = CFDataCreate(nil, mask, numBytes)!
        let provider = CGDataProvider(data: rgbData)!
        guard let rgbImageRef = CGImage(
            width: maskSize.width,
            height: maskSize.height,
            bitsPerComponent: 8,
            bitsPerPixel: 8 * numComponents,
            bytesPerRow: maskSize.width * numComponents,
            space: colorspace,
            bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: CGColorRenderingIntent(rawValue: 0)!
        ) else { return nil }
                
        return UIImage(cgImage: rgbImageRef)
            .resized(to: originalImgSize)
    }
}
