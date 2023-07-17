//
//  Array+Upsample.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 09.07.23.
//

import Foundation

extension Array where Element == UInt8 {
    func upsample(
        initialSize: (width: Int, height: Int),
        targetSize: (width: Int, height: Int)? = nil,
        scale: Int? = nil
    ) -> [UInt8] {
        let initialWidth = initialSize.width
        let initialHeight = initialSize.height
        
        let newSize: (width: Int, height: Int)? = {
            if let targetSize {
                return targetSize
            } else if let scale {
                return (width: initialWidth * scale, height: initialHeight * scale)
            }
            return nil
        }()
        
        guard let newWidth = newSize?.width,
              let newHeight = newSize?.width
        else { return self }
        
        let scaleX = Double(scale ?? (newWidth / initialWidth))
        let scaleY = Double(scale ?? (newHeight / initialHeight))
        
        var upsampledArr = [UInt8](repeating: 0, count: newWidth * newHeight)
        
        for y in 0..<newHeight {
            for x in 0..<newWidth {
             
                let sourceX = Double(x) / scaleX
                let sourceY = Double(y) / scaleY
                
                let x1 = Int(sourceX)
                let y1 = Int(sourceY)
                let x2 = Swift.min(x1 + 1, initialWidth - 1)
                let y2 = Swift.min(y1 + 1, initialHeight - 1)
                
                let q11 = self[y1 * initialWidth + x1]
                let q12 = self[y2 * initialWidth + x1]
                let q21 = self[y1 * initialWidth + x2]
                let q22 = self[y2 * initialWidth + x2]
                
                let xFraction = sourceX - Double(x1)
                let yFraction = sourceY - Double(y1)
                
                let interpolatedValue = Double(q11) * (1 - xFraction) * (1 - yFraction) +
                                        Double(q21) * (xFraction) * (1 - yFraction) +
                                        Double(q12) * (1 - xFraction) * yFraction +
                                        Double(q22) * xFraction * yFraction
                
                upsampledArr[y * newWidth + x] = UInt8(interpolatedValue)
            }
        }
        return upsampledArr
    }
}

extension Array where Element == Float {
    func upsample(
        initialSize: (width: Int, height: Int),
        targetSize: (width: Int, height: Int)? = nil,
        scale: Int? = nil
    ) -> [Float] {
        let initialWidth = initialSize.width
        let initialHeight = initialSize.height
        
        let newSize: (width: Int, height: Int)? = {
            if let targetSize {
                return targetSize
            } else if let scale {
                return (width: initialWidth * scale, height: initialHeight * scale)
            }
            return nil
        }()
        
        guard let newWidth = newSize?.width,
              let newHeight = newSize?.width
        else { return self }
        
        guard !(initialWidth == newWidth && initialHeight == newHeight) else {
            return self
        }
        
        let scaleX = Double(scale ?? (newWidth / initialWidth))
        let scaleY = Double(scale ?? (newHeight / initialHeight))
        
        var upsampledArr = [Float](repeating: 0, count: newWidth * newHeight)
        
        for y in 0..<newHeight {
            for x in 0..<newWidth {
             
                let sourceX = Double(x) / scaleX
                let sourceY = Double(y) / scaleY
                
                let x1 = Int(sourceX)
                let y1 = Int(sourceY)
                let x2 = Swift.min(x1 + 1, initialWidth - 1)
                let y2 = Swift.min(y1 + 1, initialHeight - 1)
                
                let q11 = self[y1 * initialWidth + x1]
                let q12 = self[y2 * initialWidth + x1]
                let q21 = self[y1 * initialWidth + x2]
                let q22 = self[y2 * initialWidth + x2]
                
                let xFraction = sourceX - Double(x1)
                let yFraction = sourceY - Double(y1)
                
                let interpolatedValue = Double(q11) * (1 - xFraction) * (1 - yFraction) +
                                        Double(q21) * (xFraction) * (1 - yFraction) +
                                        Double(q12) * (1 - xFraction) * yFraction +
                                        Double(q22) * xFraction * yFraction
                
                upsampledArr[y * newWidth + x] = Float(interpolatedValue)
            }
        }
        return upsampledArr
    }
}
