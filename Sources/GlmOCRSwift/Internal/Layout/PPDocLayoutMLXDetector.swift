import CoreGraphics
import Foundation

internal actor PPDocLayoutMLXDetector: LayoutDetector, PipelineLayoutDetecting {
    private let inferenceClient: any LayoutInferenceClient
    private nonisolated static let traceEnabled = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_PIPELINE_TRACE"] == "1"

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

        for (pageIndex, page) in pages.enumerated() {
            try Task.checkCancellation()
            Self.trace("detectDetailed.page.start index=\(pageIndex) size=\(page.width)x\(page.height)")
            let regions = try await inferenceClient.detectLayout(image: page)
            Self.trace("detectDetailed.page.done index=\(pageIndex) regions=\(regions.count)")
            pageRegions.append(regions)
        }

        return pageRegions
    }

    private nonisolated static func trace(_ message: String) {
        guard traceEnabled else {
            return
        }
        let payload = "[PPDocLayoutMLXDetector] \(message)\n"
        let data = payload.data(using: .utf8) ?? Data()
        FileHandle.standardError.write(data)
    }
}
