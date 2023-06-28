//
//  Array+Subscript.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 08.06.23.
//

import Foundation

extension Array {
    public subscript(
        index: Int,
        default defaultValue: @autoclosure () -> Element?
    ) -> Element? {
        guard index >= 0, index < endIndex else {
            return defaultValue()
        }

        return self[index]
    }
}
