import Foundation

package struct PPDocLayoutRuntimeOptions: Sendable {
    package var threshold: Float
    package var thresholdByClass: [String: Float]?
    package var layoutNMS: Bool
    package var layoutUnclipRatio: (Double, Double)
    package var layoutMergeBBoxesMode: [Int: LayoutMergeMode]
    package var labelTaskMapping: [String: Set<String>]
    package var id2label: [Int: String]?

    package init(
        threshold: Float = PPDocLayoutMLXContract.defaultDetectionThreshold,
        thresholdByClass: [String: Float]? = nil,
        layoutNMS: Bool = true,
        layoutUnclipRatio: (Double, Double) = PPDocLayoutMLXContract.defaultUnclipRatio,
        layoutMergeBBoxesMode: [Int: LayoutMergeMode] = PPDocLayoutMLXContract.layoutMergeBBoxesMode,
        labelTaskMapping: [String: Set<String>] = PPDocLayoutMLXContract.labelTaskMapping,
        id2label: [Int: String]? = nil
    ) {
        self.threshold = threshold
        self.thresholdByClass = thresholdByClass
        self.layoutNMS = layoutNMS
        self.layoutUnclipRatio = layoutUnclipRatio
        self.layoutMergeBBoxesMode = layoutMergeBBoxesMode
        self.labelTaskMapping = labelTaskMapping
        self.id2label = id2label
    }
}
