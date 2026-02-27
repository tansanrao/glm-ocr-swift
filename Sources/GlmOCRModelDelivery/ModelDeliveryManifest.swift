import Foundation
import HuggingFace

internal struct ModelDeliveryManifest: Codable, Sendable, Equatable {
    internal var version: Int
    internal var models: [ModelDeliveryModelSpec]

    internal init(version: Int, models: [ModelDeliveryModelSpec]) {
        self.version = version
        self.models = models
    }

    internal func modelSpec(for modelID: String) -> ModelDeliveryModelSpec? {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        return models.first(where: { $0.modelID == trimmed })
    }

    internal static func bundled() throws -> ModelDeliveryManifest {
        guard let url = Bundle.module.url(forResource: "model-manifest", withExtension: "json") else {
            throw ModelDeliveryError.invalidConfiguration("Missing bundled model-manifest.json resource")
        }

        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(ModelDeliveryManifest.self, from: data)
        } catch {
            throw ModelDeliveryError.invalidConfiguration(
                "Unable to decode bundled model-manifest.json: \(error.localizedDescription)"
            )
        }
    }

    internal static func fallbackSpec(for modelID: String) throws -> ModelDeliveryModelSpec {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard Repo.ID(rawValue: trimmed) != nil else {
            throw ModelDeliveryError.unsupportedModelID(trimmed)
        }

        return ModelDeliveryModelSpec(
            modelID: trimmed,
            repoKind: .model,
            revision: "main",
            downloadGlobs: ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja"],
            requiredFiles: ["config.json"],
            requireAnySafetensors: true
        )
    }
}

internal struct ModelDeliveryModelSpec: Codable, Sendable, Equatable {
    internal var modelID: String
    internal var repoKind: Repo.Kind
    internal var revision: String
    internal var downloadGlobs: [String]
    internal var requiredFiles: [String]
    internal var requireAnySafetensors: Bool

    internal init(
        modelID: String,
        repoKind: Repo.Kind,
        revision: String,
        downloadGlobs: [String],
        requiredFiles: [String],
        requireAnySafetensors: Bool
    ) {
        self.modelID = modelID
        self.repoKind = repoKind
        self.revision = revision
        self.downloadGlobs = downloadGlobs
        self.requiredFiles = requiredFiles
        self.requireAnySafetensors = requireAnySafetensors
    }

    internal var repoID: Repo.ID? {
        Repo.ID(rawValue: modelID)
    }
}
