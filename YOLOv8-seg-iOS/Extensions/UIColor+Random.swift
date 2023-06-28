//
//  UIColor+Random.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 28.06.23.
//

import Foundation

extension UIColor {
    static var random: UIColor {
        return UIColor(
            red: .random(in: 0...1),
            green: .random(in: 0...1),
            blue: .random(in: 0...1),
            alpha: 1.0
        )
    }
}
