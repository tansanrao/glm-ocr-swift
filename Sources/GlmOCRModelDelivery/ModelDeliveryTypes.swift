import Foundation

public struct ModelDeliveryRequest: Sendable, Equatable {
    public let recognizerModelID: String
    public let layoutModelID: String

    public init(recognizerModelID: String, layoutModelID: String) {
        self.recognizerModelID = recognizerModelID
        self.layoutModelID = layoutModelID
    }
}

public struct ModelDeliveryResolvedPaths: Sendable, Equatable {
    public let recognizerModelDirectory: URL
    public let layoutModelDirectory: URL

    public init(recognizerModelDirectory: URL, layoutModelDirectory: URL) {
        self.recognizerModelDirectory = recognizerModelDirectory
        self.layoutModelDirectory = layoutModelDirectory
    }
}

public protocol ModelDeliveryManaging: Sendable {
    func ensureReady(config: ModelDeliveryRequest) async throws -> ModelDeliveryResolvedPaths
    func verifyOfflineReadiness(config: ModelDeliveryRequest) async throws -> ModelDeliveryResolvedPaths
}

public enum ModelDeliveryError: Error, Sendable, Equatable {
    case invalidConfiguration(String)
    case unsupportedModelID(String)
    case missingRequiredFile(modelID: String, path: String)
    case missingSafetensor(modelID: String)
    case missingPersistedState(modelID: String)
    case missingFileMetadata(modelID: String, path: String)
    case checksumMismatch(modelID: String, path: String, expected: String, actual: String)
    case cacheUnavailable(String)
    case hubFailure(String)
    case ioFailure(String)
}

extension ModelDeliveryError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message):
            return "Invalid model delivery configuration: \(message)"
        case .unsupportedModelID(let modelID):
            return "Unsupported or invalid model id '\(modelID)'"
        case .missingRequiredFile(let modelID, let path):
            return "Model '\(modelID)' is missing required file '\(path)'"
        case .missingSafetensor(let modelID):
            return "Model '\(modelID)' has no .safetensors files"
        case .missingPersistedState(let modelID):
            return "No persisted model delivery state found for '\(modelID)'"
        case .missingFileMetadata(let modelID, let path):
            return "Missing persisted metadata for '\(path)' in model '\(modelID)'"
        case .checksumMismatch(let modelID, let path, let expected, let actual):
            return "Checksum mismatch for '\(path)' in model '\(modelID)' (expected \(expected), got \(actual))"
        case .cacheUnavailable(let message):
            return "Model cache unavailable: \(message)"
        case .hubFailure(let message):
            return "Hugging Face download failed: \(message)"
        case .ioFailure(let message):
            return "Model delivery I/O failure: \(message)"
        }
    }
}

internal struct ModelDeliveryState: Codable, Sendable, Equatable {
    internal var version: Int
    internal var models: [String: ModelDeliveryModelState]

    internal init(version: Int = 1, models: [String: ModelDeliveryModelState] = [:]) {
        self.version = version
        self.models = models
    }
}

internal struct ModelDeliveryModelState: Codable, Sendable, Equatable {
    internal var modelID: String
    internal var revision: String
    internal var snapshotPath: String
    internal var updatedAtUTC: Date
    internal var files: [ModelDeliveryFileState]

    internal init(
        modelID: String,
        revision: String,
        snapshotPath: String,
        updatedAtUTC: Date,
        files: [ModelDeliveryFileState]
    ) {
        self.modelID = modelID
        self.revision = revision
        self.snapshotPath = snapshotPath
        self.updatedAtUTC = updatedAtUTC
        self.files = files
    }
}

internal struct ModelDeliveryFileState: Codable, Sendable, Equatable {
    internal var relativePath: String
    internal var etag: String
    internal var commitHash: String?

    internal init(relativePath: String, etag: String, commitHash: String?) {
        self.relativePath = relativePath
        self.etag = etag
        self.commitHash = commitHash
    }
}
