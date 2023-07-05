// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#import "InferenceModule.h"
#import <Libtorch-Lite/Libtorch-Lite.h>

@implementation InferenceModule {
    @protected torch::jit::mobile::Module _impl;
    @private int inputWidth;
    @private int inputHeight;
    @private NSArray<NSNumber*>* outputSizes;
}

/// Creates an instance of InferenceModule.
///
/// - Parameters
///     - filePath: The model's location.
///     - inputSize: The model's input size.
///     - outputSizes: The model's output sizes.
- (nullable instancetype)initWithFileAtPath:(NSString*)filePath
                                  inputSize:(CGSize)inputSize
                                outputSizes:(NSArray<NSNumber*>*)outputSizes
{
    self = [super init];
    if (self) {
        try {
            _impl = torch::jit::_load_for_mobile(filePath.UTF8String);
        } catch (const std::exception& exception) {
            NSLog(@"Exception while loading model: %s", exception.what());
            return nil;
        }
        self->inputWidth  = inputSize.width;
        self->inputHeight = inputSize.height;
        self->outputSizes  = outputSizes;
    }
    return self;
}

/// Start inference on buffer using PyTorch Mobile module.
///
/// - Parameter imageBuffer:
/// - Returns: The model's output tensor in form of an array of NSNumbers.
- (NSArray<NSArray<NSNumber*>*>*)detectImage:(void*)imageBuffer
{
    try {
        // Create tensor from imageBuffer
        at::Tensor tensor = torch::from_blob(imageBuffer, { 1, 3, inputHeight, inputWidth }, at::kFloat);
        
        auto outputTuple = _impl.forward({ tensor }).toTuple();
        
        c10::ivalue::TupleElements elements = outputTuple->elements();
        size_t numOfOutputs = elements.size();
        
        if (numOfOutputs != outputSizes.count) {
            return nil;
        }
        
        NSMutableArray* results = [[NSMutableArray alloc] init];
        for (int i = 0; i < numOfOutputs; i++) {
            at::Tensor tensor = elements[i].toTensor();

            // Cast outputTensor to array of float values
            float* floatBuffer = tensor.data_ptr<float>();
            if (!floatBuffer) {
                [results addObject:@[]];
                continue;
            }
            
            NSMutableArray* tensorArray = [[NSMutableArray alloc] init];

            for (int j = 0; j < [outputSizes[i] intValue]; j++) {
              [tensorArray addObject:@(floatBuffer[j])];
            }

            [results addObject:tensorArray];
        }

        return [results copy];
    } catch (const std::exception& exception) {
        NSLog(@"Exception while running inference: %s", exception.what());
    }
    return nil;
}

@end
