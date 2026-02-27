import CoreImage
import Foundation

import GlmOCRRecognizerMLX

internal protocol GLMInferenceContainer: Sendable {}

internal protocol GLMInferenceClient: Sendable {
    func loadContainer(modelID: String) async throws -> any GLMInferenceContainer
    func recognize(
        container: any GLMInferenceContainer,
        prompt: String,
        image: CIImage,
        generationOptions: GlmOcrGenerationOptions
    ) async throws -> String
}

internal enum GLMInferenceClientError: Error, Equatable, Sendable {
    case invalidContainerType
    case imageConversionFailed
    case unresolvedModelDirectory(String)
}

internal struct MLXContainerHandle: GLMInferenceContainer {
    internal let runtime: GlmOcrRecognizerRuntime

    internal init(runtime: GlmOcrRecognizerRuntime) {
        self.runtime = runtime
    }
}

internal struct MLXGLMInferenceClient: GLMInferenceClient {
    private static let ciContext = CIContext(options: nil)

    internal init() {}

    internal func loadContainer(modelID: String) async throws -> any GLMInferenceContainer {
        let resolvedDirectory: URL
        let candidateDirectory = URL(fileURLWithPath: modelID)
        if FileManager.default.fileExists(atPath: candidateDirectory.path) {
            resolvedDirectory = candidateDirectory
        } else if let cached = cachedSnapshotDirectory(for: modelID) {
            resolvedDirectory = cached
        } else {
            throw GLMInferenceClientError.unresolvedModelDirectory(modelID)
        }

        let runtime = try await GlmOcrRecognizerRuntime(modelDirectory: resolvedDirectory)
        return MLXContainerHandle(runtime: runtime)
    }

    internal func recognize(
        container: any GLMInferenceContainer,
        prompt: String,
        image: CIImage,
        generationOptions: GlmOcrGenerationOptions
    ) async throws -> String {
        guard let handle = container as? MLXContainerHandle else {
            throw GLMInferenceClientError.invalidContainerType
        }

        guard let cgImage = Self.ciContext.createCGImage(image, from: image.extent) else {
            throw GLMInferenceClientError.imageConversionFailed
        }

        return try await handle.runtime.recognize(
            prompt: prompt,
            image: cgImage,
            options: generationOptions
        )
    }

    private func cachedSnapshotDirectory(for modelID: String) -> URL? {
        let sanitized = modelID.replacingOccurrences(of: "/", with: "--")
        guard let appSupportDirectory = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first else {
            return nil
        }

        let snapshotsRoot = appSupportDirectory
            .appending(path: "GlmOCRSwift")
            .appending(path: "huggingface")
            .appending(path: "hub")
            .appending(path: "models--\(sanitized)")
            .appending(path: "snapshots")

        guard let candidates = try? FileManager.default.contentsOfDirectory(
            at: snapshotsRoot,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        let ranked = candidates.sorted { lhs, rhs in
            let lhsDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? .distantPast
            let rhsDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? .distantPast
            return lhsDate > rhsDate
        }

        return ranked.first
    }
}
