//
//  DetectionLayer.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 05.06.23.
//

import UIKit

class DetectionLayer: CALayer {
    
    private let rotationAngle: CGFloat = CGFloat(0)
    
    func addDetection(
        objectBounds: CGRect,
        className: String? = nil,
        confidence: Float
    ) {
        let shapeLayer = createRoundedRectLayerWithBounds(objectBounds)
        
        let annotationLayer = createAnnotationLayer(
            objectBounds,
            identifier: className ?? "",
            confidence: confidence)
        
        shapeLayer.addSublayer(annotationLayer)
        
        addSublayer(shapeLayer)
        
        updateLayerGeometry()
    }
    
    func updateLayerGeometry() {
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)

        // rotate the layer into screen orientation and scale and mirror
        setAffineTransform(
            CGAffineTransform(rotationAngle: rotationAngle)
                .scaledBy(x: 1, y: -1))

        CATransaction.commit()
    }
    
    private func createAnnotationLayer(
        _ bounds: CGRect,
        identifier: String,
        confidence: Float
    ) -> AnnotationLayer {
        let text = String(format: "%.2f", confidence)
        
        let layer = AnnotationLayer(text: text)
        layer.name = "AnnotationLayer"
        layer.position = CGPoint(x: bounds.midX, y: bounds.minY - layer.bounds.height / 2)
        // rotate the layer into screen orientation and scale and mirror
        layer.setAffineTransform(
            CGAffineTransform(rotationAngle: rotationAngle)
                .scaledBy(x: 1.0, y: -1.0))
        return layer
    }
    
    private func createRoundedRectLayerWithBounds(_ bounds: CGRect) -> CALayer {
        let shapeLayer = CALayer()
        shapeLayer.borderColor = UIColor.yellow.cgColor
        shapeLayer.borderWidth = 2
        shapeLayer.bounds = bounds
        shapeLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        shapeLayer.name = "Found Object"
        shapeLayer.cornerRadius = 8
        return shapeLayer
    }
    
}
