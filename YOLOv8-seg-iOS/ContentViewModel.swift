//
//  ContentViewModel.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 18.05.23.
//

import Combine
import CoreML
import PhotosUI
import SwiftUI
import TensorFlowLite
import Vision

// MARK: ContentViewModel
class ContentViewModel: ObservableObject {
    
    var cancellables = Set<AnyCancellable>()
    
    @Published var imageSelection: PhotosPickerItem?
    @Published var uiImage: UIImage?
    @Published var selectedDetector: Int = 0
    
    @Published var predictions: [Prediction] = []
    @Published var maskPredictions: [MaskPrediction] = []
    
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
                            self?.uiImage = uiImage
                        }
                        return
                    }
                }

                print("Failed")
            }
            
        }.store(in: &cancellables)
    }
    
    func runInference() {
        switch selectedDetector {
        case 0:
            runCoreMLInference()
        case 1:
            runPyTorchInference()
        case 2:
            runTFLiteInference()
        case 3:
            runVisionInference()
        default:
            break
        }
    }
}

// MARK: CoreML Inference
extension ContentViewModel {
    private func runCoreMLInference() {
        guard let uiImage else { return }
        
        NSLog("Start inference using CoreML")
        
        let config = MLModelConfiguration()
        
        guard let model = try? YOLOv8n_coco128_seg_06_05_23(configuration: config) else {
            NSLog("Failed to init model")
            return
        }
        
        let inputDesc = model.model.modelDescription.inputDescriptionsByName
        guard let imgInputDesc = inputDesc["image"],
              let imgsz = imgInputDesc.imageConstraint
        else { return }
        
        let resizedImage = uiImage.resized(to: CGSize(width: imgsz.pixelsWide, height: imgsz.pixelsHigh)).cgImage
        
        guard let pixelBuffer = resizedImage?.pixelBufferFromCGImage(pixelFormatType: kCVPixelFormatType_32BGRA) else {
            NSLog("Failed to create pixelBuffer from image")
            return
        }
    
        let outputs: YOLOv8n_coco128_seg_06_05_23Output
            
        do {
            outputs = try model.prediction(image: pixelBuffer)
        } catch {
            NSLog("Error while calling prediction on model: \(error)")
            return
        }
        
        // Boxes
        let var_1052 = outputs.var_1053 // (1,37,1344)
        // Masks
        let p = outputs.p // (1,32,64,64)
        
        NSLog("Got output 'var_1052' with shape: \(var_1052.shape)")
        NSLog("Got output 'p' with shape: \(p.shape)")
        
        let numSegmentationMasks = 32
        let numClasses = Int(truncating: var_1052.shape[1]) - 4 - numSegmentationMasks
        
        NSLog("Model has \(numClasses) classes")
        
        // convert output to array of predictions
        var predictions = getPredictionsFromOutput(
            output: var_1052,
            rows: Int(truncating: var_1052.shape[1]), // xywh + 1 class + 32 masks
            columns: Int(truncating: var_1052.shape[2]),
            numberOfClasses: numClasses,
            inputImgSize: CGSize(width: imgsz.pixelsWide, height: imgsz.pixelsHigh)
        )
        
        NSLog("Got \(predictions.count) predicted boxes")
        NSLog("Remove predictions with score lower than 0.8")

        // remove predictions with confidence score lower than threshold
        predictions.removeAll { $0.score < 0.8 }
        
        NSLog("\(predictions.count) predicted boxes left after removing predictions with score lower than 0.8")
        
        guard !predictions.isEmpty else {
            return
        }
        
        NSLog("Perform non maximum suppression")
        
        // Group predictions by class
        let groupedPredictions = Dictionary(grouping: predictions) { prediction in
            prediction.classIndex
        }
        
        var nmsPredictions: [Prediction] = []
        let _ = groupedPredictions.mapValues { predictions in
            nmsPredictions.append(
                contentsOf: nonMaximumSuppression(
                    predictions: predictions,
                    iouThreshold: 0.6,
                    limit: 100))
        }
        
        NSLog("\(nmsPredictions.count) boxes left after performing nms with iou threshold of 0.6")
        
        guard !nmsPredictions.isEmpty else {
            return
        }
        
        self.predictions = nmsPredictions
        
        let maskProtos = getMaskProtosFromOutput(
            output: p,
            rows: Int(truncating: p.shape[3]),
            columns: Int(truncating: p.shape[2]),
            tubes: Int(truncating: p.shape[1])
        )
        
        NSLog("Got \(maskProtos.count) mask protos")
        
        let maskPredictions = masksFromProtos(
            boxPredictions: nmsPredictions,
            maskProtos: maskProtos,
            maskSize: (
                width: Int(truncating: p.shape[3]),
                height: Int(truncating: p.shape[2])
            ),
            originalImgSize: uiImage.size
        )
        
        self.maskPredictions = maskPredictions
    }
}

// MARK: Vision Inference
extension ContentViewModel {
    private func runVisionInference() {
        func handleResults(_ results: [VNObservation], inputSize: MLImageConstraint, originalImgSize: CGSize) {
            guard let boxesOutput = results[1, default: nil] as? VNCoreMLFeatureValueObservation,
                  let masksOutput = results[0, default: nil] as? VNCoreMLFeatureValueObservation
            else {
                return
            }
            
            NSLog("Got output with index 1 (boxes) with shape: \(String(describing: boxesOutput.featureValue.multiArrayValue?.shape))")
            NSLog("Got output with index 0 (masks) with shape: \(String(describing: masksOutput.featureValue.multiArrayValue?.shape))")
            
            guard let boxes = boxesOutput.featureValue.multiArrayValue else {
                return
            }
            
            let numSegmentationMasks = 32
            let numClasses = Int(truncating: boxes.shape[1]) - 4 - numSegmentationMasks
            
            NSLog("Model has \(numClasses) classes")
            
            // convert output to array of predictions
            var predictions = getPredictionsFromOutput(
                output: boxes,
                rows: Int(truncating: boxes.shape[1]), // xywh + 1 class + 32 masks
                columns: Int(truncating: boxes.shape[2]),
                numberOfClasses: numClasses,
                inputImgSize: CGSize(width: inputSize.pixelsWide, height: inputSize.pixelsHigh)
            )

            NSLog("Got \(predictions.count) predicted boxes")
            NSLog("Remove predictions with score lower than 0.8")
            
            // remove predictions with confidence score lower than threshold
            predictions.removeAll { $0.score < 0.8 }
            
            NSLog("\(predictions.count) predicted boxes left after removing predictions with score lower than 0.8")
            
            guard !predictions.isEmpty else {
                return
            }
            
            NSLog("Perform non maximum suppression")
            
            // Group predictions by class
            let groupedPredictions = Dictionary(grouping: predictions) { prediction in
                prediction.classIndex
            }
            
            var nmsPredictions: [Prediction] = []
            let _ = groupedPredictions.mapValues { predictions in
                nmsPredictions.append(
                    contentsOf: nonMaximumSuppression(
                        predictions: predictions,
                        iouThreshold: 0.6,
                        limit: 100))
            }
            
            NSLog("\(nmsPredictions.count) boxes left after performing nms with iou threshold of 0.6")
            
            guard !nmsPredictions.isEmpty else {
                return
            }
            
            self.predictions = nmsPredictions
            
            guard let masks = masksOutput.featureValue.multiArrayValue else {
                print("No masks output")
                return
            }
            
            let maskProtos = getMaskProtosFromOutput(
                output: masks,
                rows: Int(truncating: masks.shape[3]),
                columns: Int(truncating: masks.shape[2]),
                tubes: Int(truncating: masks.shape[1])
            )
            
            NSLog("Got \(maskProtos.count) mask protos")
            
            let maskPredictions = masksFromProtos(
                boxPredictions: nmsPredictions,
                maskProtos: maskProtos,
                maskSize: (
                    width: Int(truncating: masks.shape[3]),
                    height: Int(truncating: masks.shape[2])
                ),
                originalImgSize: originalImgSize
            )
            
            self.maskPredictions = maskPredictions
        }
        
        guard let uiImage else { return }
        
        NSLog("Start inference using Vision")
        
        var requests = [VNRequest]()
        do {
            let config = MLModelConfiguration()
            
            guard let model = try? YOLOv8n_coco128_seg_06_05_23(configuration: config) else {
                print("failed to init model")
                return
            }
            
            let inputDesc = model.model.modelDescription.inputDescriptionsByName
            guard let imgInputDesc = inputDesc["image"],
                  let imgsz = imgInputDesc.imageConstraint
            else { return }
            
            // Create an instance of VNCoreMLModel from MLModel
            let visionModel = try VNCoreMLModel(for: model.model)
            // Create an instance of VNCoreMLRequest that contains the previously created VNCoreMLModel
            // and a closure that will be called after model initialization.
            let objectRecognition = VNCoreMLRequest(
                model: visionModel,
                completionHandler: { (request, error) in
                    if let error = error {
                        print("VNCoreMLRequest complete with error: \(error)")
                    }
                    
                    if let results = request.results {
                        handleResults(results, inputSize: imgsz, originalImgSize: uiImage.size)
                    }
                })
            objectRecognition.imageCropAndScaleOption = .scaleFit
            
            // Store request in request variable so that it can later be called using a VNImageRequestHandler
            requests = [objectRecognition]
        } catch let error as NSError {
            print("Model loading went wrong: \(error)")
        }
        
        guard let cgImage = uiImage.cgImage else { return }
        
        let imageRequestHandler = VNImageRequestHandler(
            cgImage: cgImage,
            orientation: .up
        )
        do {
            try imageRequestHandler.perform(requests)
        } catch {
            print(error)
        }
    }
}

// MARK: PyTorch Mobile Inference
extension ContentViewModel {
    private func runPyTorchInference() {
        guard let uiImage else { return }
        
        let inputSize = CGSize(width: 256, height: 256)
        
//        let boxesOutputShape = [1, 37, 1344]
//        let masksOutputShape = [1, 64, 64, 32]
        
        let boxesOutputShape = [1, 116, 8400]
        let masksOutputShape = [1, 160, 160, 32]
        
        NSLog("Start inference using PyTorch Mobile")
        
        guard let modelFilePath = Bundle.main.url(
            forResource: "YOLOv8n_ulcus_18-05-23.torchscript",
            withExtension: "ptl"
        )?.path else {
            NSLog("Invalid file path for pytorch model")
            return
        }
        
        guard let inferenceModule = InferenceModule(
            fileAtPath: modelFilePath,
            inputSize: inputSize,
            outputSizes: [
                boxesOutputShape.reduce(0, +) as NSNumber,
                masksOutputShape.reduce(0, +) as NSNumber]
        ) else {
            NSLog("Failed to create instance of InferenceModule")
            return
        }
        
        guard var pixelBuffer = uiImage.resized(to: inputSize).normalized() else {
            return
        }
        
        guard let outputs = inferenceModule.detect(image: &pixelBuffer) else {
            return
        }
        
        let boxesOutput = outputs[0] // Shape = (1,37,1344)
        let masksOutput = outputs[1] // Shape = (1,64,64,32)
        
        // convert output to array of predictions
        var predictions = getPredictionsFromOutput(
            output: boxesOutput as [NSNumber],
            rows: boxesOutputShape[1], // xywh + 1 class + 32 masks
            columns: boxesOutputShape[2],
            numberOfClasses: boxesOutputShape[0],
            inputImgSize: inputSize
        )
        
        NSLog("Got \(predictions.count) predicted boxes")
        NSLog("Remove predictions with score lower than 0.8")
        
        // remove predictions with confidence score lower than threshold
        predictions.removeAll { $0.score < 0.8 }
        
        NSLog("\(predictions.count) predicted boxes left after removing predictions with score lower than 0.8")
        
        guard !predictions.isEmpty else {
            return
        }
        
        NSLog("Perform non maximum suppression")
        
        // Group predictions by class
        let groupedPredictions = Dictionary(grouping: predictions) { prediction in
            prediction.classIndex
        }
        
        var nmsPredictions: [Prediction] = []
        let _ = groupedPredictions.mapValues { predictions in
            nmsPredictions.append(
                contentsOf: nonMaximumSuppression(
                    predictions: predictions,
                    iouThreshold: 0.6,
                    limit: 100))
        }
        
        NSLog("\(nmsPredictions.count) boxes left after performing nms with iou threshold of 0.6")
        
        guard !nmsPredictions.isEmpty else {
            return
        }
        
        self.predictions = nmsPredictions
    
        let maskProtos = getMaskProtosFromOutput(
            output: masksOutput as [NSNumber],
            rows: 64,
            columns: 64,
            tubes: 32
        )
        
        NSLog("Got \(maskProtos.count) mask protos")
        
        let maskPredictions = masksFromProtos(
            boxPredictions: nmsPredictions,
            maskProtos: maskProtos,
            maskSize: (width: masksOutputShape[1], height: masksOutputShape[2]),
            originalImgSize: uiImage.size
        )
        
        self.maskPredictions = maskPredictions
    }
}

// MARK: TFLite Inference
extension ContentViewModel {
    private func runTFLiteInference() {
        guard let uiImage else { return }
        
        NSLog("Start inference using TFLite")
        
        let modelFilePath = Bundle.main.url(
            forResource: "yolov8l-seg_coco128_float16",
            withExtension: "tflite")!.path
        
        let interpreter: Interpreter
        do {
            interpreter = try Interpreter(
                modelPath: modelFilePath,
                delegates: []
            )
            
            try interpreter.allocateTensors()
        } catch {
            NSLog("Error while initializing interpreter: \(error)")
            return
        }
        
        let input: Tensor
        do {
            input = try interpreter.input(at: 0)
        } catch let error {
            NSLog("Failed to get input with error: \(error.localizedDescription)")
            return
        }
        
        let inputSize = CGSize(
            width: input.shape.dimensions[1],
            height: input.shape.dimensions[2]
        )
        
        guard let data = uiImage.resized(to: inputSize).normalizedDataFromImage() else {
            return
        }
        
        let boxesOutputTensor: Tensor // Shape = (1,37,1344)
        let masksOutputTensor: Tensor // Shape = (1,64,64,32)
        
        do {
            try interpreter.copy(data, toInputAt: 0)
            try interpreter.invoke()
            
            boxesOutputTensor = try interpreter.output(at: 0)
            masksOutputTensor = try interpreter.output(at: 1)
        } catch let error {
            NSLog("Failed to invoke the interpreter with error: \(error.localizedDescription)")
            return
        }
        
        let boxesOutputShapeDim = boxesOutputTensor.shape.dimensions
        let masksOutputShapeDim = masksOutputTensor.shape.dimensions
        
        NSLog("Got output with index 0 (boxes) with shape: \(boxesOutputShapeDim)")
        NSLog("Got output with index 1 (masks) with shape: \(masksOutputShapeDim)")
        
        let numSegmentationMasks = 32
        let numClasses = boxesOutputShapeDim[1] - 4 - numSegmentationMasks
        
        NSLog("Model has \(numClasses) classes")
        
        let boxesOutput = ([Float](unsafeData: boxesOutputTensor.data) ?? [])
        
        // convert output to array of predictions
        var predictions = getPredictionsFromOutput(
            output: boxesOutput as [NSNumber],
            rows: boxesOutputShapeDim[1], // xywh + 1 class + 32 masks
            columns: boxesOutputShapeDim[2],
            numberOfClasses: numClasses,
            inputImgSize: inputSize
        )
        
        NSLog("Got \(predictions.count) predicted boxes")
        NSLog("Remove predictions with score lower than 0.8")
        
        // remove predictions with confidence score lower than threshold
        predictions.removeAll { $0.score < 0.3 }
        
        NSLog("\(predictions.count) predicted boxes left after removing predictions with score lower than 0.3")
        
        guard !predictions.isEmpty else {
            return
        }
        
        NSLog("Perform non maximum suppression")
        
        // Group predictions by class
        let groupedPredictions = Dictionary(grouping: predictions) { prediction in
            prediction.classIndex
        }
        
        var nmsPredictions: [Prediction] = []
        let _ = groupedPredictions.mapValues { predictions in
            nmsPredictions.append(
                contentsOf: nonMaximumSuppression(
                    predictions: predictions,
                    iouThreshold: 0.6,
                    limit: 100))
        }
        
        NSLog("\(nmsPredictions.count) boxes left after performing nms with iou threshold of 0.6")
        
        guard !nmsPredictions.isEmpty else {
            return
        }
        
        self.predictions = nmsPredictions
        
        let masksOutput = ([Float](unsafeData: masksOutputTensor.data) ?? [])
        let maskProtos = getMaskProtosFromOutput(
            output: masksOutput as [NSNumber],
            rows: masksOutputShapeDim[1],
            columns: masksOutputShapeDim[2],
            tubes: numSegmentationMasks
        )

        NSLog("Got \(maskProtos.count) mask protos")

        // High memory footprint ~300 MB
        let maskPredictions = masksFromProtos(
            boxPredictions: nmsPredictions,
            maskProtos: maskProtos,
            maskSize: (width: masksOutputShapeDim[1], height: masksOutputShapeDim[2]),
            originalImgSize: uiImage.size
        )

        self.maskPredictions = maskPredictions
    }
}

func sigmoid(value: Float) -> Float {
    return 1.0 / (1.0 + exp(-value))
}

// MARK: Outputs to predictions
extension ContentViewModel {
    func getPredictionsFromOutput(
        output: MLMultiArray,
        rows: Int,
        columns: Int,
        numberOfClasses: Int,
        inputImgSize: CGSize
    ) -> [Prediction] {
        guard output.count != 0 else {
            return []
        }
        var predictions = [Prediction]()
        for i in 0..<columns {
            // box in xywh
            let centerX = Float(truncating: output[0*columns+i])
            let centerY = Float(truncating: output[1*columns+i])
            let width   = Float(truncating: output[2*columns+i])
            let height  = Float(truncating: output[3*columns+i])
            
            let (classIndex, score) = {
                var classIndex: Int = 0
                var heighestScore: Float = 0
                for j in 0..<numberOfClasses {
                    let score = Float(truncating: output[(4+j)*columns+i])
                    if score > heighestScore {
                        heighestScore = score
                        classIndex = j
                    }
                }
                return (classIndex, heighestScore)
            }()
            
            let maskScores = {
                var scores: [Float] = []
                for j in 0..<32 {
                    scores.append(Float(truncating: output[4+numberOfClasses+j*columns+i]))
                }
                return scores
            }()
            
            // box in xyxy
            let left = centerX - width/2
            let top = centerY - height/2
            let right = centerX + width/2
            let bottom = centerY + height/2
            
            let prediction = Prediction(
                classIndex: classIndex,
                score: score,
                xyxy: (left, top, right, bottom),
                maskScores: maskScores,
                inputImgSize: inputImgSize
            )
            predictions.append(prediction)
        }
        
        return predictions
    }
    
    func getPredictionsFromOutput(
        output: [NSNumber],
        rows: Int,
        columns: Int,
        numberOfClasses: Int,
        inputImgSize: CGSize
    ) -> [Prediction] {
        guard !output.isEmpty else {
            return []
        }
        var predictions = [Prediction]()
        for i in 0..<columns {
            // box in xywh
            let centerX = Float(truncating: output[0*columns+i])
            let centerY = Float(truncating: output[1*columns+i])
            let width   = Float(truncating: output[2*columns+i])
            let height  = Float(truncating: output[3*columns+i])
            
            let (classIndex, score) = {
                var classIndex: Int = 0
                var heighestScore: Float = 0
                for j in 0..<numberOfClasses {
                    let score = Float(truncating: output[(4+j)*columns+i])
                    if score > heighestScore {
                        heighestScore = score
                        classIndex = j
                    }
                }
                return (classIndex, heighestScore)
            }()
            
            let maskScores = {
                var scores: [Float] = []
                for k in 0..<32 {
                    scores.append(Float(truncating: output[(4+numberOfClasses+k)*columns+i]))
                }
                return scores
            }()
            
            // box in xyxy
            let left = centerX - width/2
            let top = centerY - height/2
            let right = centerX + width/2
            let bottom = centerY + height/2
            
            let prediction = Prediction(
                classIndex: classIndex,
                score: score,
                xyxy: (left, top, right, bottom),
                maskScores: maskScores,
                inputImgSize: inputImgSize
            )
            predictions.append(prediction)
        }
        
        return predictions
    }
    
    func getMaskProtosFromOutput(
        output: MLMultiArray,
        rows: Int,
        columns: Int,
        tubes: Int
    ) -> [[UInt8]] {
        var masks: [[UInt8]] = []
        for tube in 0..<tubes {
            var mask: [UInt8] = []
            for row in 0..<(rows*columns) {
                let index = tube+(row*tubes)
                mask.append(UInt8(truncating: output[index]))
            }
            masks.append(mask)
        }
        return masks
    }
    
    func getMaskProtosFromOutput(
        output: [NSNumber],
        rows: Int,
        columns: Int,
        tubes: Int
    ) -> [[UInt8]] {
        var masks: [[UInt8]] = []
        for tube in 0..<tubes {
            var mask: [UInt8] = []
            for row in 0..<(rows*columns) {
                let index = tube+(row*tubes)
                mask.append(UInt8(truncating: output[index]))
            }
            masks.append(mask)
        }
        
        return masks
    }
}

extension ContentViewModel {
    func masksFromProtos(
        boxPredictions: [Prediction],
        maskProtos: [[UInt8]],
        maskSize: (width: Int, height: Int),
        originalImgSize: CGSize
    ) -> [MaskPrediction] {
        var maskPredictions: [MaskPrediction] = []
        for prediction in boxPredictions {
            
            let maskProtoScores = prediction.maskScores
            
            var finalMask: [Float] = []
            for (index, maskProto) in maskProtos.enumerated() {
                // multiply mask proto with weight value
                let weight = maskProtoScores[index]
                finalMask = finalMask.add(maskProto.map { Float($0) * weight })
            }
            
            finalMask = finalMask.map { (sigmoid(value: $0) > 0.5 ? 1 : 0) * 255 }
            
            let uint8Mask = finalMask.map { UInt8($0) }
            
            let croppedMask = crop(mask: uint8Mask, maskSize: maskSize, box: prediction.xyxy)
            
            maskPredictions.append(
                MaskPrediction(
                    classIndex: prediction.classIndex,
                    mask: croppedMask,
                    maskSize: maskSize,
                    originalImgSize: originalImgSize
                )
            )
        }
        
        return maskPredictions
    }
    
    private func crop(
        mask: [UInt8],
        maskSize: (width: Int, height: Int),
        box: XYXY
    ) -> [UInt8] {
        let rows = maskSize.height
        let columns = maskSize.width
        
        let x1 = Int(box.x1 / 4)+1
        let y1 = Int(box.y1 / 4)+1
        let x2 = Int(box.x2 / 4)+1
        let y2 = Int(box.y2 / 4)+1
        
        var croppedArr: [UInt8] = []
        for row in 0..<rows {
            for column in 0..<columns {
                if column >= x1 && column <= x2 && row >= y1 && row <= y2 {
                    croppedArr.append(mask[row*columns+column])
                } else {
                    croppedArr.append(0)
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
        guard !predictions.isEmpty else {
            return []
        }
        
        let sortedIndices = predictions.indices.sorted {
            predictions[$0].score > predictions[$1].score
        }
        
        var selected: [Prediction] = []
        var active = [Bool](repeating: true, count: predictions.count)
        var numActive = active.count

        // The algorithm is simple: Start with the box that has the highest score.
        // Remove any remaining boxes that overlap it more than the given threshold
        // amount. If there are any boxes left (i.e. these did not overlap with any
        // previous boxes), then repeat this procedure, until no more boxes remain
        // or the limit has been reached.
        outer: for i in 0..<predictions.count {
            
            if active[i] {
                
                let boxA = predictions[sortedIndices[i]]
                selected.append(boxA)
                
                if selected.count >= limit { break }

                for j in i+1..<predictions.count {
                
                    if active[j] {
                
                        let boxB = predictions[sortedIndices[j]]
                        
                        if IOU(a: boxA.xyxy, b: boxB.xyxy) > iouThreshold {
                            
                            active[j] = false
                            numActive -= 1
                           
                            if numActive <= 0 { break outer }
                        
                        }
                    
                    }
                
                }
            }
            
        }
        return selected
    }
    
    private func IOU(a: XYXY, b: XYXY) -> Float {
        // Calculate the intersection coordinates
        let x1 = max(a.x1, b.x1)
        let y1 = max(a.y1, b.y1)
        let x2 = max(a.x2, b.x2)
        let y2 = max(a.y1, b.y2)
        
        // Calculate the intersection area
        let intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
        
        // Calculate the union area
        let area1 = (a.x2 - a.x1) * (a.y2 - a.y1)
        let area2 = (b.x2 - b.x1) * (b.y2 - b.y1)
        let union = area1 + area2 - intersection
        
        // Calculate the IoU score
        let iou = intersection / union
        
        return iou
    }
}
