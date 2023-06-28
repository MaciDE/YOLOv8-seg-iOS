//
//  CGImage+pixelBufferFromCGImage.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 29.05.23.
//

import Foundation

extension CGImage {
    func pixelBufferFromCGImage(pixelFormatType: OSType) -> CVPixelBuffer? {
        var pxbuffer: CVPixelBuffer? = nil
        let options: NSDictionary = [:]

        let width =  width
        let height = height
        let bytesPerRow = bytesPerRow

        guard let imageData = dataProvider?.data else {
            return nil
        }
        
        let dataFromImageDataProvider = CFDataCreateMutableCopy(kCFAllocatorDefault, 0, imageData)
        
        guard let data = CFDataGetMutableBytePtr(dataFromImageDataProvider) else {
            return nil
        }

        CVPixelBufferCreateWithBytes(
            kCFAllocatorDefault,
            width,
            height,
            pixelFormatType,
            data,
            bytesPerRow,
            nil,
            nil,
            options,
            &pxbuffer
        )
        return pxbuffer
    }
}
