//
//  Prediction.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 25.06.23.
//

import Foundation

typealias XYXY = (x1: Float, y1: Float, x2: Float, y2: Float)

// MARK: Prediction
struct Prediction {
    let id = UUID()
    
    let classIndex: Int
    let score: Float
    let xyxy: XYXY
    let maskCoefficients: [Float]
    
    let inputImgSize: CGSize
}
