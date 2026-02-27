import CoreGraphics
import Foundation

internal enum PipelineTask: String, Sendable, Equatable {
    case text
    case table
    case formula
    case skip
    case abandon

    internal var ocrTask: OCRTask? {
        switch self {
        case .text:
            return .text
        case .table:
            return .table
        case .formula:
            return .formula
        case .skip, .abandon:
            return nil
        }
    }
}

internal struct PipelineLayoutRegion: Sendable, Equatable {
    internal let index: Int
    internal let label: String
    internal let task: PipelineTask
    internal let score: Double
    internal let bbox2D: [Int]
    internal let polygon2D: [[Int]]
    internal let order: Int

    internal init(
        index: Int,
        label: String,
        task: PipelineTask,
        score: Double,
        bbox2D: [Int],
        polygon2D: [[Int]],
        order: Int
    ) {
        self.index = index
        self.label = label
        self.task = task
        self.score = score
        self.bbox2D = bbox2D
        self.polygon2D = polygon2D
        self.order = order
    }
}

internal protocol PipelineLayoutDetecting: Sendable {
    func detectDetailed(
        pages: [CGImage],
        options: ParseOptions
    ) async throws -> [[PipelineLayoutRegion]]
}

internal struct PipelineRegionRecord: Sendable, Equatable {
    internal var index: Int
    internal var nativeLabel: String
    internal var task: PipelineTask
    internal var bbox2D: [Int]?
    internal var content: String?

    internal init(
        index: Int,
        nativeLabel: String,
        task: PipelineTask,
        bbox2D: [Int]?,
        content: String?
    ) {
        self.index = index
        self.nativeLabel = nativeLabel
        self.task = task
        self.bbox2D = bbox2D
        self.content = content
    }
}

internal struct PipelineRecognitionJobKey: Hashable, Sendable {
    internal let pageIndex: Int
    internal let regionPosition: Int

    internal init(pageIndex: Int, regionPosition: Int) {
        self.pageIndex = pageIndex
        self.regionPosition = regionPosition
    }
}

internal struct PipelineRecognitionJob: @unchecked Sendable {
    internal let key: PipelineRecognitionJobKey
    internal let image: CGImage
    internal let task: OCRTask
    internal let promptOverride: String?

    internal init(
        key: PipelineRecognitionJobKey,
        image: CGImage,
        task: OCRTask,
        promptOverride: String? = nil
    ) {
        self.key = key
        self.image = image
        self.task = task
        self.promptOverride = promptOverride
    }
}

internal struct PipelineRegionCropResult: @unchecked Sendable {
    internal let image: CGImage
    internal let warning: String?

    internal init(image: CGImage, warning: String?) {
        self.image = image
        self.warning = warning
    }
}
