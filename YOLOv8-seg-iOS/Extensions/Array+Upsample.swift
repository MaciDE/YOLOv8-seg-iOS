//
//  Array+Upsample.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 09.07.23.
//

import Metal

final class MetalHelper {
    static let shared = MetalHelper()

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    private init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            return nil
        }
        self.device = device
        self.commandQueue = commandQueue
    }
}

import Accelerate
import Metal
import MetalPerformanceShaders

extension Array where Element == Float {
    func upsample(
        initialSize: (width: Int, height: Int),
        targetSize: (width: Int, height: Int)? = nil,
        scale: Int? = nil,
        maskThreshold: Float
    ) -> [UInt8] {
        let initialWidth = initialSize.width
        let initialHeight = initialSize.height

        guard let metal = MetalHelper.shared else { return [] }

        let inputArray: [UInt8] = self.map { UInt8(clamping: Int($0 * 255)) }

        guard let newSize = targetSize ?? (scale.map { (initialWidth * $0, initialHeight * $0) }) else {
            return inputArray
        }

        let newWidth = newSize.0
        let newHeight = newSize.1

        guard initialWidth != newWidth || initialHeight != newHeight else {
            return inputArray
        }

        func createTexture(from array: [UInt8], width: Int, height: Int) -> MTLTexture? {
            let descriptor = MTLTextureDescriptor()
            descriptor.pixelFormat = .r8Unorm
            descriptor.width = width
            descriptor.height = height
            descriptor.usage = [.shaderRead, .shaderWrite]

            guard let texture = metal.device.makeTexture(descriptor: descriptor) else { return nil }

            let region = MTLRegionMake2D(0, 0, width, height)
            texture.replace(region: region, mipmapLevel: 0, withBytes: array, bytesPerRow: width)

            return texture
        }

        func readTextureData(texture: MTLTexture) -> [UInt8] {
            let byteCount = texture.width * texture.height
            var outputArray = [UInt8](repeating: 0, count: byteCount)
            let region = MTLRegionMake2D(0, 0, texture.width, texture.height)
            texture.getBytes(&outputArray, bytesPerRow: texture.width, from: region, mipmapLevel: 0)
            return outputArray
        }

        guard let inputTexture = createTexture(from: inputArray, width: initialWidth, height: initialHeight),
              let outputTexture = createTexture(from: [UInt8](repeating: 0, count: newWidth * newHeight), width: newWidth, height: newHeight),
              let commandBuffer = metal.commandQueue.makeCommandBuffer() else {
            return inputArray
        }

        let bilinear = MPSImageBilinearScale(device: metal.device)
        bilinear.encode(commandBuffer: commandBuffer, sourceTexture: inputTexture, destinationTexture: outputTexture)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let result = readTextureData(texture: outputTexture)
        let thresholdValue = UInt8(clamping: Int(maskThreshold * 255))

        let grayscaleArray: [UInt8] = result.map { $0 > thresholdValue ? 255 : 0 }
        return grayscaleArray
    }
}
