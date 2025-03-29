//
//  UIImage+Helpers.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 18.05.23.
//

import UIKit

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
