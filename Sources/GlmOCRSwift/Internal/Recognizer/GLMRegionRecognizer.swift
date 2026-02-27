import CoreGraphics
import CoreImage
import Foundation
import GlmOCRRecognizerMLX

internal protocol PromptRegionRecognizing: Sendable {
    func recognize(_ region: CGImage, prompt: String) async throws -> String
}

internal actor GLMRegionRecognizer: RegionRecognizer, PromptRegionRecognizing {
    private let config: GlmOCRConfig
    private let prompts: GlmOCRPromptConfig
    private let generationOptions: GlmOcrGenerationOptions
    private let inferenceClient: any GLMInferenceClient
    private var containerTask: Task<any GLMInferenceContainer, Error>?

    internal init(
        config: GlmOCRConfig,
        inferenceClient: (any GLMInferenceClient)? = nil
    ) {
        self.config = config
        self.prompts = config.prompts
        self.generationOptions = GlmOcrGenerationOptions(
            maxTokens: config.recognitionOptions.maxTokens,
            temperature: config.recognitionOptions.temperature,
            prefillStepSize: config.recognitionOptions.prefillStepSize,
            topP: config.recognitionOptions.topP,
            topK: config.recognitionOptions.topK,
            repetitionPenalty: config.recognitionOptions.repetitionPenalty
        )
        self.inferenceClient = inferenceClient ?? MLXGLMInferenceClient()
    }

    internal func recognize(_ region: CGImage, task: OCRTask) async throws -> String {
        let prompt = RecognitionPromptMapper.prompt(for: task, prompts: prompts)
        return try await recognize(region, prompt: prompt)
    }

    internal func recognize(_ region: CGImage, prompt: String) async throws -> String {
        try Task.checkCancellation()

        let ciImage = CIImage(cgImage: region)
        let container = try await modelContainer()
        let rawOutput = try await inferenceClient.recognize(
            container: container,
            prompt: prompt,
            image: ciImage,
            generationOptions: generationOptions
        )

        try Task.checkCancellation()
        return rawOutput.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
    }

    private func modelContainer() async throws -> any GLMInferenceContainer {
        if let existingTask = containerTask {
            return try await existingTask.value
        }

        let modelID = config.recognizerModelID
        let loadTask = Task {
            try await inferenceClient.loadContainer(modelID: modelID)
        }

        containerTask = loadTask

        do {
            return try await loadTask.value
        } catch {
            containerTask = nil
            throw error
        }
    }
}
