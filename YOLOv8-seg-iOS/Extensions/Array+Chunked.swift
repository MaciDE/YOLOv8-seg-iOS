//
//  Array+Chunked.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 11.06.23.
//

import Foundation

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}
