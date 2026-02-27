import CoreGraphics
import Foundation

import GlmOCRLayoutMLX

internal protocol LayoutInferenceClient: Sendable {
    func detectLayout(image: CGImage) async throws -> [PipelineLayoutRegion]
}

internal actor MLXLayoutInferenceClient: LayoutInferenceClient {
    private let runner: PPDocLayoutMLXRunner

    internal init(config: GlmOCRConfig) {
        self.runner = PPDocLayoutMLXRunner(
            modelID: config.layoutModelID,
            options: Self.makeRuntimeOptions(from: config.layout)
        )
    }

    internal func detectLayout(image: CGImage) async throws -> [PipelineLayoutRegion] {
        let detections = try await runner.detect(image: image)
        let imageWidth = max(1, image.width)
        let imageHeight = max(1, image.height)

        var regions: [PipelineLayoutRegion] = []
        regions.reserveCapacity(detections.count)

        var validIndex = 0
        for detection in detections {
            let task = mapTask(detection.task)
            if task == .abandon {
                continue
            }

            regions.append(
                PipelineLayoutRegion(
                    index: validIndex,
                    label: detection.label,
                    task: task,
                    score: Double(detection.score),
                    bbox2D: detection.bbox2D,
                    polygon2D: normalizePolygon(
                        detection.polygon,
                        imageWidth: imageWidth,
                        imageHeight: imageHeight
                    ),
                    order: detection.order
                )
            )
            validIndex += 1
        }

        return regions
    }

    private func mapTask(_ task: PPDocLayoutTask) -> PipelineTask {
        switch task {
        case .text:
            return .text
        case .table:
            return .table
        case .formula:
            return .formula
        case .skip:
            return .skip
        case .abandon:
            return .abandon
        }
    }

    private func normalizePolygon(
        _ polygon: [[Double]],
        imageWidth: Int,
        imageHeight: Int
    ) -> [[Int]] {
        guard !polygon.isEmpty else {
            return []
        }

        return polygon.compactMap { point in
            guard point.count >= 2 else {
                return nil
            }

            let x = max(0, min(1000, Int((point[0] / Double(imageWidth)) * 1000.0)))
            let y = max(0, min(1000, Int((point[1] / Double(imageHeight)) * 1000.0)))
            return [x, y]
        }
    }

    private static func makeRuntimeOptions(from config: GlmOCRLayoutConfig) -> PPDocLayoutRuntimeOptions {
        let mergeModes = config.layoutMergeBBoxesMode.reduce(into: [Int: LayoutMergeMode]()) { partial, pair in
            guard let mode = LayoutMergeMode(rawValue: pair.value) else {
                return
            }
            partial[pair.key] = mode
        }

        let taskMapping = config.labelTaskMapping.reduce(into: [String: Set<String>]()) { partial, pair in
            partial[pair.key] = Set(pair.value)
        }

        return PPDocLayoutRuntimeOptions(
            threshold: config.threshold,
            thresholdByClass: config.thresholdByClass,
            layoutNMS: config.layoutNMS,
            layoutUnclipRatio: config.layoutUnclipRatio,
            layoutMergeBBoxesMode: mergeModes.isEmpty
                ? GlmOCRLayoutConfig.defaultLayoutMergeBBoxesMode.reduce(into: [Int: LayoutMergeMode]()) { partial, pair in
                    if let mode = LayoutMergeMode(rawValue: pair.value) {
                        partial[pair.key] = mode
                    }
                }
                : mergeModes,
            labelTaskMapping: taskMapping.isEmpty
                ? GlmOCRLayoutConfig.defaultLabelTaskMapping.reduce(into: [String: Set<String>]()) { partial, pair in
                    partial[pair.key] = Set(pair.value)
                }
                : taskMapping,
            id2label: config.id2label
        )
    }
}
