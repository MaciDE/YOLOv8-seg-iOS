//
//  ContentView.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 18.05.23.
//

import SwiftUI
import _PhotosUI_SwiftUI
import CoreImage

struct ContentView: View {
    
    @ObservedObject var viewModel: ContentViewModel
    
    @State var presentMaskPreview: Bool = false
    
    var body: some View {
        VStack(spacing: 8) {
            Group {
                if let uiImage = viewModel.uiImage {
                    Image(uiImage: uiImage)
                        .resizable()
                        .scaledToFit()
                        .aspectRatio(contentMode: .fit)
                } else {
                    Color
                        .gray
                        .aspectRatio(contentMode: .fit)
                }
            }
            .overlay(buildMaskOverlay())
            .overlay(
                DetectionViewRepresentable(
                    predictions: $viewModel.predictions))
            .frame(maxHeight: 400)
            
            VStack(spacing: 8) {
                
                PhotosPicker("Pick Image", selection: $viewModel.imageSelection, matching: .images)
                
                Picker(
                    "Framework",
                    selection: $viewModel.selectedDetector
                ) {
                    Text("CoreML")
                        .tag(0)
                    Text("PyTorch")
                        .tag(1)
                    Text("TFLite")
                        .tag(2)
                    Text("Vision")
                        .tag(3)
                }
                .pickerStyle(.segmented)
                
                HStack {
                    Spacer()
                    
                    Button {
                        viewModel.runInference()
                    } label: {
                        Text("Run inference")
                    }
                    
                    Spacer()
                    
                    Button {
                        viewModel.predictions = []
                        viewModel.maskPredictions = []
                    } label: {
                        Text("Clear predictions")
                    }
                    
                    Spacer()
                }
                
            }
            .padding(.horizontal)
            .padding(.top, 32)

            if !viewModel.maskPredictions.isEmpty {
                Button("Show all masks") {
                    presentMaskPreview.toggle()
                }
            }
            
            Spacer()
            
        }.sheet(isPresented: $presentMaskPreview) {
            buildMasksSheet()
        }
    }
    
    @ViewBuilder private func buildMaskImage(mask: UIImage?) -> some View {
        if let mask {
            Image(uiImage: mask)
                .resizable()
                .scaledToFit()
                .aspectRatio(contentMode: .fit)
        }
    }
    
    @ViewBuilder private func buildMasksSheet() -> some View {
        ScrollView {
            VStack(alignment: .center, spacing: 8) {
                ForEach(Array(viewModel.maskPredictions.enumerated()), id: \.offset) { index, maskPrediction in
                    VStack(alignment: .center) {
                        Group {
                            if let maskImg = maskPrediction.getMaskImage()?.resized(to: CGSize(width: 256, height: 256)) {
                                if viewModel.maskIndices?.contains(index) ?? false {
                                    Image(uiImage: maskImg)
                                        .border(.red, width: 4)
                                } else {
                                    Image(uiImage: maskImg)
                                }
                            } else {
                                let _ = print("maskImg is nil")
                            }
                        }
                        Divider()
                    }.frame(maxWidth: .infinity, alignment: .center)
                }
            }.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
    }
    
    @ViewBuilder private func buildMaskOverlay() -> some View {
        ZStack {
            ForEach(Array((viewModel.maskPredictions).enumerated()), id: \.offset) { _, mask in
                buildMaskImage(
                    mask: mask.getMaskImage()?
                        .applyFilter(.init(name: "CIMaskToAlpha")!)
                        .applyFilter(.init(name: "CIFalseColor",
                                           parameters: ["inputColor0" : CIColor(color: UIColor.white),
                                                        "inputColor1": CIColor(color: colors[mask.classIndex])])!)
                )
                .opacity(0.7)
            }
        }
    }
}

let colors = (0..<80).map { _ in UIColor.random }
