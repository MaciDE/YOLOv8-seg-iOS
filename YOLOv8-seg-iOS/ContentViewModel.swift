//
//  ContentViewModel.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 18.05.23.
//

import Combine
import PhotosUI
import SwiftUI
import Vision

enum Status {
    case preProcessing
    case postProcessing
    case inferencing
    case parsingBoxPredictions
    case performingNMS
    case parsingMaskProtos
    case generateMasksFromProtos
}

extension Status {
    var message: String {
        switch self {
        case .preProcessing:
            return "Preprocessing..."
        case .postProcessing:
            return "Postprocessing..."
        case .inferencing:
            return "Inferencing..."
        case .parsingBoxPredictions:
            return "Parsing box predictions..."
        case .performingNMS:
            return "Performing nms..."
        case .parsingMaskProtos:
            return "Parsing mask protos..."
        case .generateMasksFromProtos:
            return "Generate masks from protos..."
        }
    }
}

// MARK: ContentViewModel
class ContentViewModel: ObservableObject {
    
    var cancellables = Set<AnyCancellable>()
    
    @Published var imageSelection: PhotosPickerItem?
    @Published var uiImage: UIImage?
    
    @Published var confidenceThreshold: Float = 0.3
    @Published var iouThreshold: Float = 0.6
    @Published var maskThreshold: Float = 0.5
    
    @MainActor @Published var processing: Bool = false
    
    @MainActor @Published var predictions: [Prediction] = []
    @Published var maskPredictions: [MaskPrediction] = []
    
    @Published var combinedMaskImage: UIImage?
    
    @MainActor @Published var status: Status? = nil
    
    init() {
        setupBindings()
    }
        
    private func setupBindings() {
        $imageSelection.sink { [weak self] item in
            guard let item else { return }
            
            Task { [weak self] in
                
                if let data = try? await item.loadTransferable(type: Data.self) {
                    if let uiImage = UIImage(data: data) {
                        await MainActor.run { [weak self] in
                            self?.predictions = []
                            self?.maskPredictions = []
                            self?.combinedMaskImage = nil
                            self?.uiImage = uiImage
                        }
                        return
                    }
                }
            }
            
        }.store(in: &cancellables)
        
        $maskPredictions.sink { [weak self] predictions in
            guard !predictions.isEmpty else { return }
            
            self?.combinedMaskImage = predictions.combineToSingleImage()
        }.store(in: &cancellables)
    }
    
    func runInference() async {
        await MainActor.run { [weak self] in
            self?.processing = true
        }
        await runVisionInference()
    }
}

// MARK: Vision Inference
extension ContentViewModel {
    private func runVisionInference() async {
        @Sendable func handleResults(
            _ results: [VNObservation],
            inputSize: MLImageConstraint,
            originalImgSize: CGSize,
            processOnlyTopScoringBox: Bool? = nil
        ) async {
            NSLog(#function)
            defer {
                Task {
                    await MainActor.run { [weak self] in
                        self?.processing = false
                    }
                    await setStatus(to: nil)
                }
            }
            
            guard let boxesOutput = results[1, default: nil] as? VNCoreMLFeatureValueObservation,
                  let masksOutput = results[0, default: nil] as? VNCoreMLFeatureValueObservation
            else {
                return
            }
            
            guard let boxes = boxesOutput.featureValue.multiArrayValue else {
                return
            }
            
            let numSegmentationMasks = 32
            let numClasses = Int(truncating: boxes.shape[1]) - 4 - numSegmentationMasks
            
            NSLog("Model has \(numClasses) classes")
            
            await setStatus(to: .parsingBoxPredictions)
            let predictions = getPredictionsFromOutput(
                output: boxes,
                rows: Int(truncating: boxes.shape[1]),
                columns: Int(truncating: boxes.shape[2]),
                numberOfClasses: numClasses,
                inputImgSize: CGSize(width: inputSize.pixelsWide, height: inputSize.pixelsHigh),
                confidenceThreshold: confidenceThreshold
            )
            
            NSLog("Got \(predictions.count) predicted boxes")
            
            await setStatus(to: .performingNMS)
            
            guard !predictions.isEmpty else { return }
            
            NSLog("Perform non maximum suppression")
            
            let groupedPredictions = Dictionary(grouping: predictions) { prediction in
                prediction.classIndex
            }
            
            var nmsPredictions: [Prediction] = []
            let _ = groupedPredictions.mapValues { predictions in
                nmsPredictions.append(
                    contentsOf: nonMaximumSuppression(
                        predictions: predictions,
                        iouThreshold: iouThreshold,
                        limit: 100))
            }
            
            NSLog("\(nmsPredictions.count) boxes left after performing nms with iou threshold of 0.6")
            
            guard !nmsPredictions.isEmpty else {
                return
            }
            
            await MainActor.run { [weak self, nmsPredictions] in
                self?.predictions = nmsPredictions
            }
            
            guard let masks = masksOutput.featureValue.multiArrayValue else {
                print("No masks output")
                return
            }
            
            await setStatus(to: .parsingMaskProtos)
            let maskProtos = getMaskProtosFromOutput(
                output: masks,
                rows: Int(truncating: masks.shape[3]),
                columns: Int(truncating: masks.shape[2]),
                tubes: Int(truncating: masks.shape[1])
            )

            NSLog("Got \(maskProtos.count) mask protos")

            await setStatus(to: .generateMasksFromProtos)
            let maskPredictions = masksFromProtos(
                boxPredictions: nmsPredictions,
                maskProtos: maskProtos,
                maskSize: (
                    width: Int(truncating: masks.shape[3]),
                    height: Int(truncating: masks.shape[2])
                ),
                originalImgSize: originalImgSize
            )
          
            NSLog("Set maskpredictions")
            await MainActor.run { [weak self, maskPredictions] in
                self?.maskPredictions = maskPredictions
                self?.processing = false
            }
            await setStatus(to: nil)
        }
        
        guard let uiImage else {
            await MainActor.run { [weak self] in
                self?.processing = false
            }
            return
        }
        
        await setStatus(to: .preProcessing)
        
        NSLog("Start inference using Vision")
        
        var requests = [VNRequest]()
        do {
            let config = MLModelConfiguration()
            
            guard let model = try? coco128_yolo11n_seg(configuration: config) else {
                print("failed to init model")
                return
            }
            
            let inputDesc = model.model.modelDescription.inputDescriptionsByName
            guard let imgInputDesc = inputDesc["image"],
                  let imgsz = imgInputDesc.imageConstraint
            else { return }
            
            let visionModel = try VNCoreMLModel(for: model.model)
            let segmentationRequest = VNCoreMLRequest(
                model: visionModel,
                completionHandler: { (request, error) in
                    if let error = error {
                        print("VNCoreMLRequest complete with error: \(error)")
                    }
                    
                    if let results = request.results {
                        Task {
                            await self.setStatus(to: .postProcessing)
                            await handleResults(results, inputSize: imgsz, originalImgSize: uiImage.size)
                        }
                    }
                })
            segmentationRequest.preferBackgroundProcessing = false
            segmentationRequest.imageCropAndScaleOption = .scaleFill
            
            requests = [segmentationRequest]
        } catch let error as NSError {
            print("Model loading went wrong: \(error)")
        }
        
        guard let cgImage = uiImage.cgImage else { return }
        
        let imageRequestHandler = VNImageRequestHandler(
            cgImage: cgImage,
            orientation: uiImage.imageOrientation.toCGImagePropertyOrientation() ?? .up
        )
        do {
            await setStatus(to: .inferencing)
            NSLog("Perform inference")
            try imageRequestHandler.perform(requests)
        } catch {
            print(error)
        }
    }
}

// MARK: Outputs to predictions
extension ContentViewModel {
    func getPredictionsFromOutput(
        output: MLMultiArray,
        rows: Int,
        columns: Int,
        numberOfClasses: Int,
        inputImgSize: CGSize,
        confidenceThreshold: Float
    ) -> [Prediction] {
        NSLog(#function)
        guard output.count != 0 else { return [] }

        let strides = output.strides.map { $0.intValue }

        let pointer = output.dataPointer.assumingMemoryBound(to: Float.self)

        @inline(__always)
        func getIndex(_ channel: Int, _ i: Int) -> Int {
            return channel * strides[1] + i * strides[2]
        }
      
        let resultsQueue = DispatchQueue(label: "resultsQueue", attributes: .concurrent)

        var predictions = [Prediction]()
        DispatchQueue.concurrentPerform(iterations: columns) { i in
            let centerX = pointer[getIndex(0, i)]
            let centerY = pointer[getIndex(1, i)]
            let width   = pointer[getIndex(2, i)]
            let height  = pointer[getIndex(3, i)]

            var classScores = [Float](repeating: 0, count: numberOfClasses)
            for j in 0..<numberOfClasses {
                let classIdx = getIndex(4 + j, i)
                classScores[j] = pointer[classIdx]
            }

            var highestScore: Float = 0
            var classIndex: vDSP_Length = 0
            vDSP_maxvi(classScores, 1, &highestScore, &classIndex, vDSP_Length(numberOfClasses))

            if highestScore >= confidenceThreshold {
                var maskCoefficients = [Float](repeating: 0, count: 32)
                for k in 0..<32 {
                    let maskIdx = getIndex(4 + numberOfClasses + k, i)
                    if maskIdx >= output.count { break }
                    maskCoefficients[k] = pointer[maskIdx]
                }
              
                // Convert box from xywh to xyxy format
                let left = centerX - width * 0.5
                let top = centerY - height * 0.5
                let right = centerX + width * 0.5
                let bottom = centerY + height * 0.5

                let prediction = Prediction(
                    classIndex: Int(classIndex),
                    score: highestScore,
                    xyxy: .init(x1: left, y1: top, x2: right, y2: bottom),
                    maskCoefficients: maskCoefficients,
                    inputImgSize: inputImgSize
                )

                resultsQueue.async(flags: .barrier) {
                    predictions.append(prediction)
                }
            }
        }
        
        resultsQueue.sync(flags: .barrier) {}
        
        return predictions
    }
  
    func getMaskProtosFromOutput(
        output: MLMultiArray,
        rows: Int,
        columns: Int,
        tubes: Int
    ) -> [[Float]] {
        NSLog(#function)
        let maskSize = rows * columns
        let strideTube = output.strides[1].intValue

        let pointer = output.dataPointer.assumingMemoryBound(to: Float.self)

        var masks = Array(repeating: [Float](repeating: 0, count: maskSize), count: tubes)

        masks.withUnsafeMutableBufferPointer { maskBuffer in
            DispatchQueue.concurrentPerform(iterations: tubes) { tube in
                let srcPointer = pointer.advanced(by: tube * strideTube)

                let destPointer = maskBuffer[tube].withUnsafeMutableBufferPointer { $0.baseAddress! }
                memcpy(destPointer, srcPointer, maskSize * MemoryLayout<Float>.size)
            }
        }

        return masks
    }
}

import Accelerate

extension ContentViewModel {
    func masksFromProtos(
        boxPredictions: [Prediction],
        maskProtos: [[Float]],
        maskSize: (width: Int, height: Int),
        originalImgSize: CGSize
    ) -> [MaskPrediction] {
        NSLog(#function)
        var maskPredictions: [MaskPrediction] = []
        for prediction in boxPredictions {
            
            let maskCoefficients = prediction.maskCoefficients

            var finalMask = [Float](repeating: 0, count: maskSize.width * maskSize.height)
            NSLog("Perform matrix multiplication to create finalMask")
            finalMask.withUnsafeMutableBufferPointer { finalMaskBuffer in
                for (index, maskProto) in maskProtos.enumerated() {
                    var coeff = maskCoefficients[index]
                  
                    maskProto.withUnsafeBufferPointer { protoBuffer in
                        guard let protoBase = protoBuffer.baseAddress,
                              let finalBase = finalMaskBuffer.baseAddress
                        else { return }

                        vDSP_vsma(protoBase, 1, &coeff, finalBase, 1, finalBase, 1, vDSP_Length(maskSize.width * maskSize.height))
                    }
                }
            }

            NSLog("Apply sigmoid")
            let count = finalMask.count
            var negated = [Float](repeating: 0, count: count)
            var expResult = [Float](repeating: 0, count: count)
            var one = Float(1.0)

            vDSP_vneg(finalMask, 1, &negated, 1, vDSP_Length(count))
          
            vvexpf(&expResult, negated, [Int32(count)])

            vDSP_vsadd(expResult, 1, &one, &expResult, 1, vDSP_Length(count))
            vDSP_svdiv(&one, expResult, 1, &finalMask, 1, vDSP_Length(count))

            NSLog("Crop mask to bounding box")
            let croppedFinalMask = crop(
                mask: finalMask,
                maskSize: maskSize,
                box: .init(
                  x1: prediction.xyxy.x1 / 4,
                  y1: prediction.xyxy.y1 / 4,
                  x2: prediction.xyxy.x2 / 4,
                  y2: prediction.xyxy.y2 / 4
                ))
          
            let scale = min(
                max(
                    Int(originalImgSize.width) / maskSize.width,
                    Int(originalImgSize.height) / maskSize.height),
                6)
            let targetSize = (
                width: maskSize.width * scale,
                height: maskSize.height * scale)
            
            NSLog("Upsample mask with size \(maskSize) to \(targetSize)")
            let upsampledMask: [UInt8] = croppedFinalMask
                .map { Float(($0 > maskThreshold ? 1 : 0)) }
                .upsample(
                    initialSize: maskSize,
                    scale: scale,
                    maskThreshold: maskThreshold)
            
            NSLog("Crop mask to bounding box")
            let croppedUpsampledMaskSize = crop(
                mask: upsampledMask,
                maskSize: targetSize,
                box: .init(
                    x1: (prediction.xyxy.x1 / 4) * Float(scale),
                    y1: (prediction.xyxy.y1 / 4) * Float(scale),
                    x2: (prediction.xyxy.x2 / 4) * Float(scale),
                    y2: (prediction.xyxy.y2 / 4) * Float(scale)
                ))
            
            maskPredictions.append(
                MaskPrediction(
                    classIndex: prediction.classIndex,
                    mask: croppedUpsampledMaskSize,
                    maskSize: targetSize))
        }
        
        return maskPredictions
    }
    
    private func crop(
        mask: [Float],
        maskSize: (width: Int, height: Int),
        box: XYXY
    ) -> [Float] {
        let rows = maskSize.height
        let columns = maskSize.width
        
        let x1 = max(0, Int(box.x1))
        let y1 = max(0, Int(box.y1))
        let x2 = min(columns - 1, Int(box.x2))
        let y2 = min(rows - 1, Int(box.y2))

        var croppedArr = [Float](repeating: 0, count: rows * columns)

        croppedArr.withUnsafeMutableBufferPointer { buffer in
            mask.withUnsafeBufferPointer { sourceBuffer in
                for row in y1...y2 {
                    let srcStartIdx = row * columns + x1
                    let dstStartIdx = row * columns + x1
                    let count = x2 - x1 + 1
                    buffer.baseAddress!.advanced(by: dstStartIdx)
                      .update(from: sourceBuffer.baseAddress!.advanced(by: srcStartIdx), count: count)
                }
            }
        }

        return croppedArr
    }
  
    private func crop(
        mask: [UInt8],
        maskSize: (width: Int, height: Int),
        box: XYXY
    ) -> [UInt8] {
        let rows = maskSize.height
        let columns = maskSize.width
        
        let x1 = max(0, Int(box.x1))
        let y1 = max(0, Int(box.y1))
        let x2 = min(columns - 1, Int(box.x2))
        let y2 = min(rows - 1, Int(box.y2))

        var croppedArr = [UInt8](repeating: 0, count: rows * columns)

        croppedArr.withUnsafeMutableBufferPointer { buffer in
            mask.withUnsafeBufferPointer { sourceBuffer in
                for row in y1...y2 {
                    let srcStartIdx = row * columns + x1
                    let dstStartIdx = row * columns + x1
                    let count = x2 - x1 + 1
                    buffer.baseAddress!.advanced(by: dstStartIdx)
                      .update(from: sourceBuffer.baseAddress!.advanced(by: srcStartIdx), count: count)
                }
            }
        }

        return croppedArr
    }
}

// MARK: Non-Maximum-Suppression
extension ContentViewModel {
    func nonMaximumSuppression(
        predictions: [Prediction],
        iouThreshold: Float,
        limit: Int
    ) -> [Prediction] {
        guard !predictions.isEmpty else { return [] }

        // Sort by confidence score in descending order
        var sortedPredictions = predictions.sorted(by: { $0.score > $1.score })
        var selected: [Prediction] = []
        selected.reserveCapacity(limit)

        while !sortedPredictions.isEmpty {
            let best = sortedPredictions.removeFirst()
            selected.append(best)

            if selected.count >= limit { break }

            sortedPredictions.removeAll { IOU(a: best.xyxy, b: $0.xyxy) > iouThreshold }
        }
        
        return selected
    }
    
    private func IOU(a: XYXY, b: XYXY) -> Float {
        let x1 = max(a.x1, b.x1)
        let y1 = max(a.y1, b.y1)
        let x2 = min(a.x2, b.x2)
        let y2 = min(a.y2, b.y2)

        let intersection = max(x2 - x1, 0) * max(y2 - y1, 0)

        let area1 = (a.x2 - a.x1) * (a.y2 - a.y1)
        let area2 = (b.x2 - b.x1) * (b.y2 - b.y1)
        let union = area1 + area2 - intersection

        return intersection / union
    }
}

extension ContentViewModel {
    @MainActor
    fileprivate func setStatus(to status: Status?) {
        self.status = status
    }
}
