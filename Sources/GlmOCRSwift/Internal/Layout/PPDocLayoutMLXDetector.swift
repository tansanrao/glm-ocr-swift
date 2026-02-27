import CoreGraphics
import Foundation

internal actor PPDocLayoutMLXDetector: LayoutDetector, PipelineLayoutDetecting {
    private let inferenceClient: any LayoutInferenceClient

    internal init(
        config: GlmOCRConfig,
        inferenceClient: (any LayoutInferenceClient)? = nil
    ) {
        if let inferenceClient {
            self.inferenceClient = inferenceClient
        } else {
            self.inferenceClient = MLXLayoutInferenceClient(config: config)
        }
    }

    internal func detect(pages: [CGImage], options: ParseOptions) async throws -> [[LayoutRegion]] {
        let detailed = try await detectDetailed(pages: pages, options: options)
        return detailed.map { page in
            page.map { region in
                LayoutRegion(
                    index: region.index,
                    label: region.label,
                    score: region.score,
                    bbox2D: region.bbox2D
                )
            }
        }
    }

    internal func detectDetailed(
        pages: [CGImage],
        options: ParseOptions
    ) async throws -> [[PipelineLayoutRegion]] {
        _ = options

        guard !pages.isEmpty else {
            return []
        }

        var pageRegions: [[PipelineLayoutRegion]] = []
        pageRegions.reserveCapacity(pages.count)

        for page in pages {
            try Task.checkCancellation()
            pageRegions.append(try await inferenceClient.detectLayout(image: page))
        }

        return pageRegions
    }
}
