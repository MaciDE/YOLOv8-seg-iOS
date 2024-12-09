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
    @Published var selectedDetector: Int = 0
    
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
        switch selectedDetector {
        case 0:
            await runCoreMLInference()
        case 1:
            await runPyTorchInference()
        case 2:
            await runTFLiteInference()
        case 3:
            await runVisionInference()
        default:
            await MainActor.run { [weak self] in
                self?.processing = false
            }
            break
        }
    }
}

// MARK: CoreML Inference
extension ContentViewModel {
    private func runCoreMLInference() async {
        guard let uiImage else { return }
        
        NSLog("Start inference using CoreML")
        
        let config = MLModelConfiguration()
        
        guard let model = try? coco128_yolo11n_seg(configuration: config) else {
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
    
        Task {
            defer {
                Task {
                    await MainActor.run { [weak self] in
                        self?.processing = false
                    }
                    await setStatus(to: nil)
                }
            }
            
            let outputs: coco128_yolo11n_segOutput
            
            do {
                outputs = try model.prediction(image: pixelBuffer)
            } catch {
                NSLog("Error while calling prediction on model: \(error)")
                return
            }
            
            let boxesOutput = outputs.var_1366
            let masksOutput = outputs.p
            
            let numSegmentationMasks = 32
            let numClasses = Int(truncating: boxesOutput.shape[1]) - 4 - numSegmentationMasks
            
            NSLog("Model has \(numClasses) classes")
            
            // Convert output to array of predictions
            var predictions = getPredictionsFromOutput(
                output: boxesOutput,
                rows: Int(truncating: boxesOutput.shape[1]),
                columns: Int(truncating: boxesOutput.shape[2]),
                numberOfClasses: numClasses,
                inputImgSize: CGSize(width: imgsz.pixelsWide, height: imgsz.pixelsHigh)
            )
            
            NSLog("Got \(predictions.count) predicted boxes")
            NSLog("Remove predictions with score lower than 0.3")
            
            // Remove predictions with confidence score lower than threshold
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
            
            await MainActor.run { [weak self, nmsPredictions] in
                self?.predictions = nmsPredictions
            }
            
            let maskProtos = getMaskProtosFromOutput(
                output: masksOutput,
                rows: Int(truncating: masksOutput.shape[3]),
                columns: Int(truncating: masksOutput.shape[2]),
                tubes: Int(truncating: masksOutput.shape[1])
            )
            
            NSLog("Got \(maskProtos.count) mask protos")
            
            let maskPredictions = masksFromProtos(
                boxPredictions: nmsPredictions,
                maskProtos: maskProtos,
                maskSize: (
                    width: Int(truncating: masksOutput.shape[3]),
                    height: Int(truncating: masksOutput.shape[2])
                ),
                originalImgSize: uiImage.size
            )
            
            await MainActor.run { [weak self, maskPredictions] in
                self?.maskPredictions = maskPredictions
            }
        }
    }
}

// MARK: Vision Inference
extension ContentViewModel {
    private func runVisionInference() async {
        @Sendable func handleResults(_ results: [VNObservation], inputSize: MLImageConstraint, originalImgSize: CGSize) async {
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
            
            // Convert output to array of predictions
            await setStatus(to: .parsingBoxPredictions)
            var predictions = getPredictionsFromOutput(
                output: boxes,
                rows: Int(truncating: boxes.shape[1]),
                columns: Int(truncating: boxes.shape[2]),
                numberOfClasses: numClasses,
                inputImgSize: CGSize(width: inputSize.pixelsWide, height: inputSize.pixelsHigh)
            )

            NSLog("Got \(predictions.count) predicted boxes")
            NSLog("Remove predictions with score lower than 0.3")
            
            await setStatus(to: .performingNMS)
            
            // Remove predictions with confidence score lower than threshold
            predictions.removeAll { $0.score < confidenceThreshold }
            
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
            try imageRequestHandler.perform(requests)
        } catch {
            print(error)
        }
    }
}

// MARK: PyTorch Mobile Inference
extension ContentViewModel {
    private func runPyTorchInference() async {
        guard let uiImage else { return }
        
        await setStatus(to: .preProcessing)
        
        let inputSize = CGSize(width: 640, height: 640)

        let boxesOutputShape = [1, 116, 8400]
        let masksOutputShape = [1, 32, 160, 160]

        NSLog("Start inference using PyTorch Mobile")

        guard let modelFilePath = Bundle.main.url(
            forResource: "coco128-yolo11n-seg.torchscript",
            withExtension: "ptl"
        )?.path else {
            NSLog("Invalid file path for pytorch model")
            return
        }
        
        Task {
            defer {
                Task {
                    await MainActor.run { [weak self] in
                        self?.processing = false
                    }
                    await setStatus(to: nil)
                }
            }
            
            guard let inferenceModule = InferenceModule(
                fileAtPath: modelFilePath,
                inputSize: inputSize,
                outputSizes: [
                    boxesOutputShape.reduce(1, *) as NSNumber,
                    masksOutputShape.reduce(1, *) as NSNumber
                ]
            ) else {
                NSLog("Failed to create instance of InferenceModule")
                return
            }
            
            guard var pixelBuffer = uiImage.resized(to: inputSize).normalized() else {
                return
            }
            
            await setStatus(to: .inferencing)
            guard let outputs = inferenceModule.detect(image: &pixelBuffer) else {
                return
            }
            await setStatus(to: .postProcessing)
            
            let boxesOutput = outputs[0]
            
            let numSegmentationMasks = 32
            let numClasses = boxesOutputShape[1] - 4 - numSegmentationMasks
            
            NSLog("Model has \(numClasses) classes")
            
            // Convert output to array of predictions
            await setStatus(to: .parsingBoxPredictions)
            var predictions = getPredictionsFromOutput(
                output: boxesOutput as [NSNumber],
                rows: boxesOutputShape[1],
                columns: boxesOutputShape[2],
                numberOfClasses: numClasses,
                inputImgSize: inputSize
            )
            
            NSLog("Got \(predictions.count) predicted boxes")
            NSLog("Remove predictions with score lower than 0.3")
            
            await setStatus(to: .performingNMS)
            
            // Remove predictions with confidence score lower than threshold
            predictions.removeAll { $0.score < confidenceThreshold }
            
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
            
            let masksOutput = outputs[1]
            await setStatus(to: .parsingMaskProtos)
            let maskProtos = getMaskProtosFromOutputPyTorch(
                output: masksOutput as [NSNumber],
                rows: masksOutputShape[2],
                columns: masksOutputShape[3],
                tubes: numSegmentationMasks
            )

            NSLog("Got \(maskProtos.count) mask protos")
            await setStatus(to: .generateMasksFromProtos)
            let maskPredictions = masksFromProtos(
                boxPredictions: nmsPredictions,
                maskProtos: maskProtos,
                maskSize: (width: masksOutputShape[2], height: masksOutputShape[3]),
                originalImgSize: uiImage.size
            )

            await MainActor.run { [weak self, maskPredictions] in
                self?.maskPredictions = maskPredictions
            }
        }
    }
}

// MARK: TFLite Inference
extension ContentViewModel {
    private func runTFLiteInference() async {
        guard let uiImage else { return }
        
        await setStatus(to: .preProcessing)
        
        NSLog("Start inference using TFLite")
        
        let modelFilePath = Bundle.main.url(
            forResource: "coco128-yolo11n-seg_float16",
            withExtension: "tflite")!.path

        Task {
            defer {
                Task {
                    await MainActor.run { [weak self] in
                        self?.processing = false
                    }
                    await setStatus(to: nil)
                }
            }
            
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
            
            let boxesOutputTensor: Tensor
            let masksOutputTensor: Tensor
            
            do {
                try interpreter.copy(data, toInputAt: 0)
                await setStatus(to: .inferencing)
                try interpreter.invoke()
                await setStatus(to: .postProcessing)
                
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
            
            // Convert output to array of predictions
            await setStatus(to: .parsingBoxPredictions)
            var predictions = getPredictionsFromOutput(
                output: boxesOutput as [NSNumber],
                rows: boxesOutputShapeDim[1],
                columns: boxesOutputShapeDim[2],
                numberOfClasses: numClasses,
                inputImgSize: inputSize
            )
            
            NSLog("Got \(predictions.count) predicted boxes")
            NSLog("Remove predictions with score lower than 0.3")
            
            await setStatus(to: .performingNMS)
            
            // Remove predictions with confidence score lower than threshold
            predictions.removeAll { $0.score < confidenceThreshold }
            
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
                        iouThreshold: iouThreshold,
                        limit: 100))
            }
            
            NSLog("\(nmsPredictions.count) boxes left after performing nms with iou threshold of 0.6")
            
            guard !nmsPredictions.isEmpty else {
                return
            }
            
            // Scale boxes to input size
            nmsPredictions = nmsPredictions.map { prediction in
                return Prediction(
                    classIndex: prediction.classIndex,
                    score: prediction.score,
                    xyxy: (
                        prediction.xyxy.x1 * Float(inputSize.width),
                        prediction.xyxy.y1 * Float(inputSize.height),
                        prediction.xyxy.x2 * Float(inputSize.width),
                        prediction.xyxy.y2 * Float(inputSize.height)
                    ),
                    maskCoefficients: prediction.maskCoefficients,
                    inputImgSize: prediction.inputImgSize)
            }
            
            await MainActor.run { [weak self, nmsPredictions] in
                self?.predictions = nmsPredictions
            }
            
            let masksOutput = ([Float](unsafeData: masksOutputTensor.data) ?? [])
            await setStatus(to: .parsingMaskProtos)
            let maskProtos = getMaskProtosFromOutput(
                output: masksOutput as [NSNumber],
                rows: masksOutputShapeDim[1],
                columns: masksOutputShapeDim[2],
                tubes: numSegmentationMasks
            )
            
            NSLog("Got \(maskProtos.count) mask protos")
            await setStatus(to: .generateMasksFromProtos)
            let maskPredictions = masksFromProtos(
                boxPredictions: nmsPredictions,
                maskProtos: maskProtos,
                maskSize: (width: masksOutputShapeDim[1], height: masksOutputShapeDim[2]),
                originalImgSize: uiImage.size
            )
            
            await MainActor.run { [weak self, maskPredictions] in
                self?.maskPredictions = maskPredictions
            }
        }
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
            
            let maskCoefficients = {
                var coefficients: [Float] = []
                for k in 0..<32 {
                    coefficients.append(Float(truncating: output[(4+numberOfClasses+k)*columns+i]))
                }
                return coefficients
            }()
            
            // Convert box from xywh to xyxy
            let left = centerX - width/2
            let top = centerY - height/2
            let right = centerX + width/2
            let bottom = centerY + height/2
            
            let prediction = Prediction(
                classIndex: classIndex,
                score: score,
                xyxy: (left, top, right, bottom),
                maskCoefficients: maskCoefficients,
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
            
            let maskCoefficients = {
                var coefficients: [Float] = []
                for k in 0..<32 {
                    coefficients.append(Float(truncating: output[(4+numberOfClasses+k)*columns+i]))
                }
                return coefficients
            }()
            
            // Convert box from xywh to xyxy
            let left = centerX - width/2
            let top = centerY - height/2
            let right = centerX + width/2
            let bottom = centerY + height/2
            
            let prediction = Prediction(
                classIndex: classIndex,
                score: score,
                xyxy: (left, top, right, bottom),
                maskCoefficients: maskCoefficients,
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
            for i in 0..<(rows*columns) {
                let index = tube*(rows*columns)+i
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
    
    func getMaskProtosFromOutputPyTorch(
        output: [NSNumber],
        rows: Int,
        columns: Int,
        tubes: Int
    ) -> [[UInt8]] {
        var masks: [[UInt8]] = []
        for tube in 0..<tubes {
            var mask: [UInt8] = []
            for i in 0..<(rows*columns) {
                let index = tube*(rows*columns)+i
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
        NSLog("Generate masks from prototypes")
        var maskPredictions: [MaskPrediction] = []
        for prediction in boxPredictions {
            
            let maskCoefficients = prediction.maskCoefficients
            
            var finalMask: [Float] = []
            for (index, maskProto) in maskProtos.enumerated() {
                let weight = maskCoefficients[index]
                finalMask = finalMask.add(maskProto.map { Float($0) * weight })
            }
            
            NSLog("Apply sigmoid")
            finalMask = finalMask.map { sigmoid(value: $0) }
            
            NSLog("Crop mask to bounding box")
            let croppedMask = crop(
                mask: finalMask,
                maskSize: maskSize,
                box: prediction.xyxy)

            let scale = min(
                max(
                    Int(originalImgSize.width) / maskSize.width,
                    Int(originalImgSize.height) / maskSize.height),
                6)
            let targetSize = (
                width: maskSize.width * scale,
                height: maskSize.height * scale)
            
            NSLog("Upsample mask with size \(maskSize) to \(targetSize)")
            let upsampledMask = croppedMask
                .map { Float(($0 > maskThreshold ? 1 : 0)) }
                .upsample(
                    initialSize: maskSize,
                    scale: scale)
                .map { UInt8(($0 > maskThreshold ? 1 : 0) * 255) }
            
            maskPredictions.append(
                MaskPrediction(
                    classIndex: prediction.classIndex,
                    mask: upsampledMask,
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
        
        let x1 = Int(box.x1 / 4)
        let y1 = Int(box.y1 / 4)
        let x2 = Int(box.x2 / 4)
        let y2 = Int(box.y2 / 4)
        
        var croppedArr: [Float] = []
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

extension ContentViewModel {
    @MainActor
    fileprivate func setStatus(to status: Status?) {
        self.status = status
    }
}
