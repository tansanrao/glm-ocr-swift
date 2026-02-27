import Foundation
import MLX

public enum GlmOcrRecognizerMLXError: Error, Equatable, Sendable {
    case invalidModelDirectory(String)
    case missingRequiredFile(String)
    case invalidConfiguration(String)
    case tokenizerFailed(String)
    case weightLoadingFailed(String)
    case processingFailed(String)
    case generationFailed(String)
    case pythonFallbackUnavailable(String)
}

public struct GlmOcrTHW: Sendable, Codable, Equatable {
    public let t: Int
    public let h: Int
    public let w: Int

    public init(t: Int, h: Int, w: Int) {
        self.t = t
        self.h = h
        self.w = w
    }

    public var product: Int {
        t * h * w
    }
}

public struct GlmOcrPreparedInput: @unchecked Sendable {
    public let inputIDs: [Int]
    public let attentionMask: [Int]
    public let pixelValues: MLXArray
    public let imageGridTHW: [GlmOcrTHW]
    public let imageTokenID: Int

    public init(
        inputIDs: [Int],
        attentionMask: [Int],
        pixelValues: MLXArray,
        imageGridTHW: [GlmOcrTHW],
        imageTokenID: Int
    ) {
        self.inputIDs = inputIDs
        self.attentionMask = attentionMask
        self.pixelValues = pixelValues
        self.imageGridTHW = imageGridTHW
        self.imageTokenID = imageTokenID
    }
}

public struct GlmOcrInputSignature: Sendable, Equatable {
    public let tokenCount: Int
    public let imageTokenCount: Int
    public let imageGridTHW: [[Int]]
    public let tokens: [Int]

    public init(
        tokenCount: Int,
        imageTokenCount: Int,
        imageGridTHW: [[Int]],
        tokens: [Int]
    ) {
        self.tokenCount = tokenCount
        self.imageTokenCount = imageTokenCount
        self.imageGridTHW = imageGridTHW
        self.tokens = tokens
    }
}

public struct GlmOcrGenerationOptions: Sendable, Equatable {
    public let maxTokens: Int
    public let temperature: Float
    public let prefillStepSize: Int
    public let topP: Float
    public let topK: Int
    public let repetitionPenalty: Float

    public init(
        maxTokens: Int,
        temperature: Float,
        prefillStepSize: Int,
        topP: Float,
        topK: Int,
        repetitionPenalty: Float
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.prefillStepSize = prefillStepSize
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
    }

    public static func fromEnvironment(_ environment: [String: String] = ProcessInfo.processInfo.environment)
        -> GlmOcrGenerationOptions
    {
        let maxTokens = Int(environment["GLMOCR_MAX_TOKENS"] ?? "") ?? 4_096
        let temperature = Float(environment["GLMOCR_TEMPERATURE"] ?? "") ?? 0.8
        let prefillStepSize = Int(environment["GLMOCR_PREFILL_STEP_SIZE"] ?? "") ?? 2_048
        let topP = Float(environment["GLMOCR_TOP_P"] ?? "") ?? 0.9
        let topK = Int(environment["GLMOCR_TOP_K"] ?? "") ?? 50
        let repetitionPenalty = Float(environment["GLMOCR_REPETITION_PENALTY"] ?? "") ?? 1.1

        return GlmOcrGenerationOptions(
            maxTokens: max(1, maxTokens),
            temperature: max(0, temperature),
            prefillStepSize: max(1, prefillStepSize),
            topP: max(0, topP),
            topK: max(1, topK),
            repetitionPenalty: max(0.01, repetitionPenalty)
        )
    }
}

public struct GlmOcrGenerationConfig: Sendable, Equatable {
    public let eosTokenIDs: [Int]
    public let padTokenID: Int?

    public init(eosTokenIDs: [Int], padTokenID: Int?) {
        self.eosTokenIDs = eosTokenIDs
        self.padTokenID = padTokenID
    }
}

public struct GlmOcrModelConfig: Sendable, Equatable {
    public struct RopeParameters: Sendable, Equatable {
        public let mropeSection: [Int]
        public let ropeTheta: Float
        public let partialRotaryFactor: Float

        public init(
            mropeSection: [Int],
            ropeTheta: Float,
            partialRotaryFactor: Float
        ) {
            self.mropeSection = mropeSection
            self.ropeTheta = ropeTheta
            self.partialRotaryFactor = partialRotaryFactor
        }
    }

    public struct TextConfig: Sendable, Equatable {
        public let vocabSize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let numHiddenLayers: Int
        public let headDim: Int
        public let rmsNormEps: Float
        public let attentionBias: Bool
        public let maxPositionEmbeddings: Int
        public let ropeParameters: RopeParameters

        public init(
            vocabSize: Int,
            hiddenSize: Int,
            intermediateSize: Int,
            numAttentionHeads: Int,
            numKeyValueHeads: Int,
            numHiddenLayers: Int,
            headDim: Int,
            rmsNormEps: Float,
            attentionBias: Bool,
            maxPositionEmbeddings: Int,
            ropeParameters: RopeParameters
        ) {
            self.vocabSize = vocabSize
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.numAttentionHeads = numAttentionHeads
            self.numKeyValueHeads = numKeyValueHeads
            self.numHiddenLayers = numHiddenLayers
            self.headDim = headDim
            self.rmsNormEps = rmsNormEps
            self.attentionBias = attentionBias
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.ropeParameters = ropeParameters
        }
    }

    public struct VisionConfig: Sendable, Equatable {
        public let depth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHeads: Int
        public let patchSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int
        public let inChannels: Int
        public let outHiddenSize: Int
        public let rmsNormEps: Float
        public let attentionBias: Bool

        public init(
            depth: Int,
            hiddenSize: Int,
            intermediateSize: Int,
            numHeads: Int,
            patchSize: Int,
            spatialMergeSize: Int,
            temporalPatchSize: Int,
            inChannels: Int,
            outHiddenSize: Int,
            rmsNormEps: Float,
            attentionBias: Bool
        ) {
            self.depth = depth
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.numHeads = numHeads
            self.patchSize = patchSize
            self.spatialMergeSize = spatialMergeSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.outHiddenSize = outHiddenSize
            self.rmsNormEps = rmsNormEps
            self.attentionBias = attentionBias
        }
    }

    public let modelType: String
    public let vocabSize: Int
    public let imageTokenID: Int
    public let videoTokenID: Int
    public let imageStartTokenID: Int
    public let imageEndTokenID: Int
    public let eosTokenIDs: [Int]
    public let padTokenID: Int
    public let tieWordEmbeddings: Bool
    public let textConfig: TextConfig
    public let visionConfig: VisionConfig

    public init(
        modelType: String,
        vocabSize: Int,
        imageTokenID: Int,
        videoTokenID: Int,
        imageStartTokenID: Int,
        imageEndTokenID: Int,
        eosTokenIDs: [Int],
        padTokenID: Int,
        tieWordEmbeddings: Bool,
        textConfig: TextConfig,
        visionConfig: VisionConfig
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.imageTokenID = imageTokenID
        self.videoTokenID = videoTokenID
        self.imageStartTokenID = imageStartTokenID
        self.imageEndTokenID = imageEndTokenID
        self.eosTokenIDs = eosTokenIDs
        self.padTokenID = padTokenID
        self.tieWordEmbeddings = tieWordEmbeddings
        self.textConfig = textConfig
        self.visionConfig = visionConfig
    }
}

public struct GlmOcrProcessorConfig: Sendable, Equatable {
    public let imageMean: [Float]
    public let imageStd: [Float]
    public let patchSize: Int
    public let mergeSize: Int
    public let temporalPatchSize: Int
    public let minPixels: Int
    public let maxPixels: Int
    public let imageToken: String

    public init(
        imageMean: [Float],
        imageStd: [Float],
        patchSize: Int,
        mergeSize: Int,
        temporalPatchSize: Int,
        minPixels: Int,
        maxPixels: Int,
        imageToken: String
    ) {
        self.imageMean = imageMean
        self.imageStd = imageStd
        self.patchSize = patchSize
        self.mergeSize = mergeSize
        self.temporalPatchSize = temporalPatchSize
        self.minPixels = minPixels
        self.maxPixels = maxPixels
        self.imageToken = imageToken
    }
}

public struct GlmOcrModelBundle: Sendable, Equatable {
    public let modelDirectory: URL
    public let modelConfig: GlmOcrModelConfig
    public let processorConfig: GlmOcrProcessorConfig
    public let generationConfig: GlmOcrGenerationConfig

    public init(
        modelDirectory: URL,
        modelConfig: GlmOcrModelConfig,
        processorConfig: GlmOcrProcessorConfig,
        generationConfig: GlmOcrGenerationConfig
    ) {
        self.modelDirectory = modelDirectory
        self.modelConfig = modelConfig
        self.processorConfig = processorConfig
        self.generationConfig = generationConfig
    }
}
