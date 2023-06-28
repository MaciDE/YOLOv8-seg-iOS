//
//  AnnotationLayer.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 05.06.23.
//

import UIKit

class AnnotationLayer: CALayer {
    
    private let directionTriangleSize: CGFloat = 7
    private let fontSize: CGFloat = 16
    private let contentHeight: CGFloat = 31
    private let paddingOffset: CGFloat = 4
    private let maxWidth: CGFloat = 200
    
    init(text: String) {
        super.init()
        
        let scalingFactor = 1.0
        
        let directionTriangleSize = directionTriangleSize * scalingFactor
        let contentHeight = contentHeight * scalingFactor
        let paddingOffset = paddingOffset * scalingFactor
        let maxWidth = maxWidth * scalingFactor
        let fontSize = fontSize * (0.5 * scalingFactor + 0.5)
        
        let maxTextWidth = maxWidth - paddingOffset * 3
        
        // Text
        let textLayer = getTextLayer(text: text, fontSize: fontSize, maxWidth: maxTextWidth)
        let textX = 2 * paddingOffset
        let textY = directionTriangleSize + (contentHeight - textLayer.frame.height) / 2
        textLayer.transform = CATransform3DMakeTranslation(textX, textY, 0)
        
        let contentWidth = textLayer.frame.width + paddingOffset * 4
        let contentSize = CGSize(width: contentWidth, height: contentHeight)
        
        // Backround
        let backgroundLayer = getAnnotationBackground(
            directionTriangleSize: directionTriangleSize,
            size: contentSize
        )
        
        addSublayer(backgroundLayer)
        addSublayer(textLayer)
        let size = CGSize(width: contentSize.width, height: contentSize.height + directionTriangleSize)
        bounds = CGRect(x: 0, y: 0, width: size.width, height: size.height)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func getTextLayer(text: String, fontSize: CGFloat, maxWidth: CGFloat) -> CATextLayer {
        let textLayer = CATextLayer()
        
        textLayer.rasterizationScale = UIScreen.main.scale
        textLayer.contentsScale = UIScreen.main.scale
        textLayer.alignmentMode = .center
        textLayer.foregroundColor = UIColor.black.cgColor
        textLayer.fontSize = fontSize
        textLayer.font = UIFont.systemFont(ofSize: fontSize, weight: .regular)
        textLayer.isWrapped = false
        textLayer.truncationMode = .end
        textLayer.string = text
        
        let preferredSize = textLayer.preferredFrameSize()
        textLayer.frame = CGRect(x: 0, y: 0, width: min(maxWidth, preferredSize.width), height: preferredSize.height)
        
        return textLayer
    }
    
    private func getAnnotationBackground(directionTriangleSize: CGFloat, size: CGSize) -> CALayer {
        let layer = CAShapeLayer()
        let path = UIBezierPath()
        
        let fullFrameRect = CGRect(
            x: 0,
            y: 0,
            width: size.width,
            height: size.height + directionTriangleSize
        )
        let contentRect = CGRect(
            x: fullFrameRect.minX,
            y: fullFrameRect.minY + directionTriangleSize,
            width: fullFrameRect.width,
            height: fullFrameRect.height - directionTriangleSize
        )
        let radius = contentRect.height / 2
        
        path.move(to: CGPoint(x: contentRect.minX + radius, y: contentRect.minY))
        
        path.addLine(to: CGPoint(x: contentRect.maxX - contentRect.width / 2 + directionTriangleSize, y: contentRect.minY))
        path.addLine(to: CGPoint(x: contentRect.maxX - contentRect.width / 2, y: fullFrameRect.minY))
        path.addLine(to: CGPoint(x: contentRect.maxX - contentRect.width / 2 - directionTriangleSize, y: contentRect.minY))
        path.addArc(
            withCenter: CGPoint(x: contentRect.maxX - radius, y: contentRect.minY + radius),
            radius: radius,
            startAngle: -.pi / 2,
            endAngle: 0,
            clockwise: true
        )
        path.addLine(to: CGPoint(x: contentRect.maxX, y: contentRect.maxY - radius))
        path.addArc(
            withCenter: CGPoint(x: contentRect.maxX - radius, y: contentRect.maxY - radius),
            radius: radius,
            startAngle: 0,
            endAngle: .pi / 2,
            clockwise: true
        )
        path.addLine(to: CGPoint(x: contentRect.minX + radius, y: contentRect.maxY))
        path.addArc(
            withCenter: CGPoint(x: contentRect.minX + radius, y: contentRect.maxY - radius),
            radius: radius,
            startAngle: .pi / 2,
            endAngle: .pi,
            clockwise: true
        )
        path.addLine(to: CGPoint(x: contentRect.minX, y: contentRect.minY + radius))
        path.addArc(
            withCenter: CGPoint(x: contentRect.minX + radius, y: contentRect.minY + radius),
            radius: radius,
            startAngle: .pi,
            endAngle: .pi * 3 / 2,
            clockwise: true
        )
        layer.path = path.cgPath
        layer.fillColor = UIColor.white.withAlphaComponent(0.8).cgColor
        
        return layer
    }
}
