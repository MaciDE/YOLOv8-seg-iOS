//
//  UIImage+Helpers.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 18.05.23.
//

import UIKit

extension UIImage {
    func resized(to newSize: CGSize, scale: CGFloat = 1) -> UIImage {
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = scale
        let renderer = UIGraphicsImageRenderer(size: newSize, format: format)
        let image = renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: newSize))
        }
        return image
    }
    
    func normalized() -> [Float32]? {
        guard let cgImage = self.cgImage else {
            return nil
        }
        let w = cgImage.width
        let h = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * w
        let bitsPerComponent = 8
        var rawBytes: [UInt8] = [UInt8](repeating: 0, count: w * h * 4)
        rawBytes.withUnsafeMutableBytes { ptr in
            if let cgImage = self.cgImage,
               let context = CGContext(
                data: ptr.baseAddress,
                width: w,
                height: h,
                bitsPerComponent: bitsPerComponent,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) {
                let rect = CGRect(x: 0, y: 0, width: w, height: h)
                context.draw(cgImage, in: rect)
            }
        }
        var normalizedBuffer: [Float32] = [Float32](repeating: 0, count: w * h * 3)
        for i in 0 ..< w * h {
            normalizedBuffer[i] = Float32(rawBytes[i * 4 + 0]) / 255.0
            normalizedBuffer[w * h + i] = Float32(rawBytes[i * 4 + 1]) / 255.0
            normalizedBuffer[w * h * 2 + i] = Float32(rawBytes[i * 4 + 2]) / 255.0
        }
        return normalizedBuffer
    }
    
    func transformToUpOrientation() -> UIImage {
        UIGraphicsBeginImageContext(size)
        draw(at: .zero)
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return newImage ?? self
    }
    
    /// Converts UIImage to Data
    ///
    /// Ignores alpha channel and normalizes pixel values
    ///
    /// - Parameter image: Image that will be converted to data
    func normalizedDataFromImage() -> Data? {
        let imageSize = size
        
        guard let cgImage: CGImage = cgImage else {
            return nil
        }
        guard let context = CGContext(
          data: nil,
          width: cgImage.width, height: cgImage.height,
          bitsPerComponent: 8, bytesPerRow: cgImage.width * 4,
          space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
          return nil
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
        guard let imageData = context.data else { return nil }

        var inputData = Data()
        for row in 0 ..< Int(imageSize.width) {
            for col in 0 ..< Int(imageSize.height) {
                let offset = 4 * (row * context.width + col)
                // (Ignore offset 0, the unused alpha channel)
                let red = imageData.load(fromByteOffset: offset+1, as: UInt8.self)
                let green = imageData.load(fromByteOffset: offset+2, as: UInt8.self)
                let blue = imageData.load(fromByteOffset: offset+3, as: UInt8.self)

                // Normalize channel values to [0.0, 1.0]. This requirement varies
                // by model. For example, some models might require values to be
                // normalized to the range [-1.0, 1.0] instead, and others might
                // require fixed-point values or the original bytes.
                var normalizedRed = Float32(red) / 255.0
                var normalizedGreen = Float32(green) / 255.0
                var normalizedBlue = Float32(blue) / 255.0

                // Append normalized values to Data object in RGB order.
                let elementSize = MemoryLayout.size(ofValue: normalizedRed)
                var bytes = [UInt8](repeating: 0, count: elementSize)
                memcpy(&bytes, &normalizedRed, elementSize)
                inputData.append(&bytes, count: elementSize)
                memcpy(&bytes, &normalizedGreen, elementSize)
                inputData.append(&bytes, count: elementSize)
                memcpy(&bytes, &normalizedBlue, elementSize)
                inputData.append(&bytes, count: elementSize)
            }
        }
        return inputData
    }
    
    func applyFilter(_ filter: CIFilter) -> UIImage {
        let ciImage = (ciImage ?? CIImage(cgImage: cgImage!))
        filter.setValue(ciImage, forKey: kCIInputImageKey)
        guard let outputImage = filter.outputImage else {
            print("Can not apply filter, outputImage is nil")
            return self
        }
        let context = CIContext(options: nil)
        guard let cgImage = context.createCGImage(outputImage, from: outputImage.extent) else {
            print("Can not create CGImage from outputImage created by filter")
            return self
        }
        return UIImage(cgImage: cgImage, scale: self.scale, orientation: self.imageOrientation)
    }
}

extension UIImage {
    func imageWithInsets(insets: UIEdgeInsets) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(
            CGSize(width: self.size.width + insets.left + insets.right,
                   height: self.size.height + insets.top + insets.bottom), false, self.scale)
        let _ = UIGraphicsGetCurrentContext()
        let origin = CGPoint(x: insets.left, y: insets.top)
        self.draw(at: origin)
        let imageWithInsets = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return imageWithInsets
    }
}

extension UIImage.Orientation {
    func toCGImagePropertyOrientation() -> CGImagePropertyOrientation? {
        switch self {
        case .up:
            return .up
        case .down:
            return .down
        case .left:
            return .left
        case .right:
            return .right
        case .upMirrored:
            return .upMirrored
        case .downMirrored:
            return .downMirrored
        case .leftMirrored:
            return .leftMirrored
        case .rightMirrored:
            return .rightMirrored
        @unknown default:
            return nil
        }
    }
}
