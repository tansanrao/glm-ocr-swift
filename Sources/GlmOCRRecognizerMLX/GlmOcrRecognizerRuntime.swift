import CoreGraphics
import Foundation
import MLX
import MLXNN
import Tokenizers

public actor GlmOcrRecognizerRuntime {
    public let modelDirectory: URL
    public let modelBundle: GlmOcrModelBundle

    private let tokenizer: any Tokenizer
    private let processor: GlmOcrRecognizerProcessor
    private let model: GlmOcrRecognizerModel
    private let eosTokenIDs: Set<Int>

    private let repetitionContextSize = 20
    private let traceEnabled = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_NATIVE_TRACE"] == "1"
    private let traceAllTokens = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_NATIVE_TOKENS"] == "1"
    private let tracePixels = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_PIXEL_TRACE"] == "1"

    private static func traceWrite(_ line: String) {
        let payload = "[GlmOcrRecognizerRuntime] \(line)\n"
        let data = payload.data(using: .utf8) ?? Data()
        FileHandle.standardError.write(data)
    }

    private static func imageRGBSummary(_ image: CGImage) -> (count: Int, sum: Float, prefix: [Float])? {
        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else {
            return nil
        }

        let pixelCount = width * height
        var raw = [UInt8](repeating: 0, count: pixelCount * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &raw,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            return nil
        }

        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var values = [Float]()
        values.reserveCapacity(pixelCount * 3)
        for index in 0 ..< pixelCount {
            let base = index * 4
            values.append(Float(raw[base]))
            values.append(Float(raw[base + 1]))
            values.append(Float(raw[base + 2]))
        }

        return (values.count, values.reduce(Float(0), +), Array(values.prefix(8)))
    }

    public init(modelDirectory: URL) async throws {
        if ProcessInfo.processInfo.environment["GLMOCR_DEBUG_NATIVE_TRACE"] == "1" {
            Self.traceWrite("init.start modelDirectory=\(modelDirectory.path)")
        }
        let bundle = try GlmOcrRecognizerConfigLoader.loadBundle(modelDirectory: modelDirectory)

        let tokenizer: any Tokenizer
        do {
            tokenizer = try await AutoTokenizer.from(directory: modelDirectory)
        } catch {
            throw GlmOcrRecognizerMLXError.tokenizerFailed(error.localizedDescription)
        }

        let model: GlmOcrRecognizerModel
        do {
            model = try Self.loadModel(
                modelDirectory: modelDirectory,
                modelConfig: bundle.modelConfig,
                trace: ProcessInfo.processInfo.environment["GLMOCR_DEBUG_NATIVE_TRACE"] == "1"
            )
        } catch let error as GlmOcrRecognizerMLXError {
            throw error
        } catch {
            throw GlmOcrRecognizerMLXError.weightLoadingFailed(error.localizedDescription)
        }

        MLXRandom.seed(0)

        self.modelDirectory = modelDirectory
        self.modelBundle = bundle
        self.tokenizer = tokenizer
        self.processor = GlmOcrRecognizerProcessor(
            tokenizer: tokenizer,
            modelConfig: bundle.modelConfig,
            processorConfig: bundle.processorConfig
        )
        self.model = model

        let configuredEOS = bundle.generationConfig.eosTokenIDs.isEmpty
            ? bundle.modelConfig.eosTokenIDs
            : bundle.generationConfig.eosTokenIDs
        self.eosTokenIDs = Set(configuredEOS)
        if ProcessInfo.processInfo.environment["GLMOCR_DEBUG_NATIVE_TRACE"] == "1" {
            Self.traceWrite("init.complete")
        }
    }

    public func prepareInput(prompt: String, image: CGImage) throws -> GlmOcrPreparedInput {
        try processor.prepare(prompt: prompt, image: image)
    }

    public func inputSignature(prompt: String, image: CGImage) throws -> GlmOcrInputSignature {
        try processor.inputSignature(prompt: prompt, image: image)
    }

    public func recognize(
        prompt: String,
        image: CGImage,
        options: GlmOcrGenerationOptions = .fromEnvironment()
    ) async throws -> String {
        trace("recognize.start")
        if tracePixels, let source = Self.imageRGBSummary(image) {
            trace("recognize.sourceRGB count=\(source.count) sum=\(source.sum) prefix=\(source.prefix)")
        }
        let prepared = try processor.prepare(prompt: prompt, image: image)
        trace("recognize.prepared tokens=\(prepared.inputIDs.count) grid=\(prepared.imageGridTHW.map { "[\($0.t),\($0.h),\($0.w)]" }.joined(separator: ","))")
        if tracePixels {
            let values = prepared.pixelValues.asArray(Float.self)
            let prefix = Array(values.prefix(8))
            let checksum = values.reduce(Float(0), +)
            trace("recognize.pixels count=\(values.count) sum=\(checksum) prefix=\(prefix)")
        }
        let generatedTokenIDs = try generate(prepared: prepared, options: options)
        if traceAllTokens {
            trace("recognize.tokens ids=\(generatedTokenIDs)")
        }
        trace("recognize.generated tokenCount=\(generatedTokenIDs.count)")
        let decoded = tokenizer.decode(tokens: generatedTokenIDs)
        if traceAllTokens {
            trace("recognize.decoded text=\(decoded)")
        }
        return decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func loadModel(
        modelDirectory: URL,
        modelConfig: GlmOcrModelConfig,
        trace: Bool
    ) throws -> GlmOcrRecognizerModel {
        if trace {
            traceWrite("loadModel.start")
        }
        let model = GlmOcrRecognizerModel(config: modelConfig)

        var safetensorFiles: [URL] = []
        if let enumerator = FileManager.default.enumerator(
            at: modelDirectory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) {
            for case let url as URL in enumerator {
                if url.pathExtension == "safetensors" {
                    safetensorFiles.append(url)
                }
            }
        }

        guard !safetensorFiles.isEmpty else {
            throw GlmOcrRecognizerMLXError.missingRequiredFile(
                "No .safetensors files found in \(modelDirectory.path)"
            )
        }
        if trace {
            traceWrite("loadModel.shards count=\(safetensorFiles.count)")
        }

        safetensorFiles.sort { $0.path < $1.path }

        var rawWeights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            if trace {
                traceWrite("loadModel.read shard=\(file.lastPathComponent)")
            }
            let shard = try loadArrays(url: file)
            for (key, value) in shard {
                rawWeights[key] = value
            }
        }
        if trace {
            traceWrite("loadModel.rawWeights count=\(rawWeights.count)")
        }

        let sanitized = model.sanitize(weights: rawWeights)
        if trace {
            traceWrite("loadModel.sanitized count=\(sanitized.count)")
        }
        let parameters = ModuleParameters.unflattened(sanitized)

        do {
            try model.update(parameters: parameters, verify: [.noUnusedKeys, .shapeMismatch])
        } catch {
            throw GlmOcrRecognizerMLXError.weightLoadingFailed(
                "Model parameter update failed: \(error.localizedDescription)"
            )
        }
        if trace {
            traceWrite("loadModel.update complete")
        }

        eval(model)
        if trace {
            traceWrite("loadModel.eval complete")
        }
        return model
    }

    private func generate(
        prepared: GlmOcrPreparedInput,
        options: GlmOcrGenerationOptions
    ) throws -> [Int] {
        var inputIDs = MLXArray(prepared.inputIDs).reshaped(1, prepared.inputIDs.count).asType(.int32)
        var attentionMask = MLXArray(prepared.attentionMask).reshaped(1, prepared.attentionMask.count).asType(.int32)
        let pixelValues = prepared.pixelValues

        let gridValues = prepared.imageGridTHW.flatMap { thw in
            [Int32(thw.t), Int32(thw.h), Int32(thw.w)]
        }
        let imageGridTHW = MLXArray(gridValues).reshaped(prepared.imageGridTHW.count, 3).asType(.int32)

        var cache: [GlmOcrKVCache?] = (0 ..< modelBundle.modelConfig.textConfig.numHiddenLayers).map { _ in
            GlmOcrSimpleKVCache()
        }

        var embeddings = model.getInputEmbeddings(
            inputIDs: inputIDs,
            pixelValues: pixelValues,
            attentionMask: attentionMask,
            imageGridTHW: imageGridTHW
        ).inputEmbeddings
        trace("generate.embeddings shape=\(embeddings.shape)")

        if options.prefillStepSize > 0, embeddings.dim(1) > options.prefillStepSize {
            trace("generate.prefill chunking step=\(options.prefillStepSize)")
            while embeddings.dim(1) > 1 {
                let nToProcess = min(options.prefillStepSize, embeddings.dim(1) - 1)
                trace("generate.prefill chunk=\(nToProcess) remaining=\(embeddings.dim(1))")

                let chunkIDs = inputIDs[0..., ..<nToProcess]
                let chunkEmbeddings = embeddings[0..., ..<nToProcess, 0...]

                let chunkLogits = model.logits(
                    inputIDs: chunkIDs,
                    inputEmbeddings: chunkEmbeddings,
                    attentionMask: attentionMask,
                    cache: cache,
                    // Vision inputs are already merged into `chunkEmbeddings`.
                    // Passing them again resets multimodal rope state and can corrupt decode caches.
                    pixelValues: nil,
                    imageGridTHW: nil
                )
                eval(chunkLogits)
                trace("generate.prefill chunk complete")

                embeddings = embeddings[0..., nToProcess..., 0...]
                inputIDs = inputIDs[0..., nToProcess...]
                attentionMask = attentionMask[0..., nToProcess...]

                Memory.clearCache()
            }

            let lastIndex = inputIDs.dim(1) - 1
            inputIDs = inputIDs[0..., lastIndex...]
            attentionMask = attentionMask[0..., lastIndex...]
            trace("generate.prefill done promptReduced=\(inputIDs.shape)")
        }

        var historyTokens: [Int] = []

        let first = try generationStep(
            inputIDs: inputIDs,
            inputEmbeddings: embeddings,
            cache: cache,
            attentionMask: attentionMask,
            historyTokens: &historyTokens,
            options: options
        )

        var currentToken = first
        trace("generate.first token=\(currentToken)")
        var generated: [Int] = []
        generated.reserveCapacity(options.maxTokens)

        for index in 0 ..< options.maxTokens {
            if eosTokenIDs.contains(currentToken) {
                break
            }

            generated.append(currentToken)

            if index % 256 == 0 {
                Memory.clearCache()
            }

            if index == options.maxTokens - 1 {
                break
            }

            let nextInput = MLXArray(currentToken).reshaped(1, 1).asType(.int32)
            currentToken = try generationStep(
                inputIDs: nextInput,
                inputEmbeddings: nil,
                cache: cache,
                attentionMask: nil,
                historyTokens: &historyTokens,
                options: options
            )
            if traceAllTokens || index < 4 {
                trace("generate.step index=\(index + 1) token=\(currentToken)")
            }
        }

        return generated
    }

    private func generationStep(
        inputIDs: MLXArray,
        inputEmbeddings: MLXArray?,
        cache: [GlmOcrKVCache?],
        attentionMask: MLXArray?,
        historyTokens: inout [Int],
        options: GlmOcrGenerationOptions
    ) throws -> Int {
        let logits3D: MLXArray

        if let inputEmbeddings {
            trace("generationStep.prefillDecode inputShape=\(inputIDs.shape) embedShape=\(inputEmbeddings.shape)")
            logits3D = model.logits(
                inputIDs: inputIDs,
                inputEmbeddings: inputEmbeddings,
                attentionMask: attentionMask,
                cache: cache,
                pixelValues: nil,
                imageGridTHW: nil
            )
        } else {
            trace("generationStep.decodeOnly inputShape=\(inputIDs.shape)")
            logits3D = model.decodeLogits(
                inputIDs: inputIDs,
                attentionMask: attentionMask,
                cache: cache,
                positionIDs: nil
            )
        }

        var logits = logits3D[0..., -1, 0...]
        trace("generationStep.logits shape=\(logits.shape)")

        let stepTokens = inputIDs.asArray(Int32.self).map(Int.init)
        historyTokens.append(contentsOf: stepTokens)

        if options.repetitionPenalty != 1 {
            logits = applyRepetitionPenalty(
                logits: logits,
                penalty: options.repetitionPenalty,
                tokens: historyTokens
            )
        }

        let logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)
        let sampled = sample(logprobs: logprobs, options: options)

        eval(sampled)
        trace("generationStep.sampled")
        return sampled.item(Int.self)
    }

    private func applyRepetitionPenalty(
        logits: MLXArray,
        penalty: Float,
        tokens: [Int]
    ) -> MLXArray {
        guard penalty > 0, !tokens.isEmpty else {
            return logits
        }

        let context = Array(tokens.suffix(repetitionContextSize))
        guard !context.isEmpty else {
            return logits
        }

        var logits = logits
        let contextArray = MLXArray(context).asType(.int32)
        var selected = logits[0..., contextArray]
        selected = `where`(
            selected .< 0,
            selected * penalty,
            selected / penalty
        )
        logits[0..., contextArray] = selected
        return logits
    }

    private func sample(logprobs: MLXArray, options: GlmOcrGenerationOptions) -> MLXArray {
        if options.temperature == 0 {
            return argMax(logprobs, axis: -1)
        }

        var filtered = logprobs

        if options.topP > 0, options.topP < 1 {
            filtered = applyTopP(filtered, topP: options.topP)
        }

        if options.topK > 0 {
            filtered = applyTopK(filtered, topK: options.topK)
        }

        let scaled = filtered * (1 / options.temperature)
        return MLXRandom.categorical(scaled, axis: -1).asType(.int32)
    }

    private func applyTopP(_ logprobs: MLXArray, topP: Float) -> MLXArray {
        let probs = exp(logprobs)
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedProbs = takeAlong(probs, sortedIndices, axis: -1)

        var cumulative = sortedProbs.cumsum(axis: -1)
        let inverseIndices = putAlong(
            zeros(like: sortedIndices),
            sortedIndices,
            values: MLXArray(0 ..< sortedIndices.dim(-1)).asType(sortedIndices.dtype),
            axis: -1
        )
        cumulative = takeAlong(cumulative, inverseIndices, axis: -1)

        let negInf = MLXArray(-Float.infinity, dtype: logprobs.dtype)
        return `where`(cumulative .> (1 - topP), logprobs, negInf)
    }

    private func applyTopK(_ logprobs: MLXArray, topK: Int) -> MLXArray {
        let vocabSize = logprobs.dim(-1)
        guard topK > 0, topK < vocabSize else {
            return logprobs
        }

        let maskIndices = argPartition(-logprobs, kth: topK - 1, axis: -1)[0..., topK...]
        let negInf = MLXArray(-Float.infinity, dtype: logprobs.dtype)
        return putAlong(logprobs, maskIndices, values: negInf, axis: -1)
    }

    private func trace(_ message: String) {
        if traceEnabled {
            Self.traceWrite(message)
        }
    }
}
