//
//  Array+Add.swift
//  YOLOv8-seg-iOS
//
//  Created by Marcel Opitz on 28.06.23.
//

import Foundation

extension Array where Element == Float {
    func add(_ arr: [Float]) -> [Float] {
        zip(self, arr).map(+) + (self.count < arr.count ? arr[self.count ..< arr.count] : self[arr.count ..< self.count])
    }
}
