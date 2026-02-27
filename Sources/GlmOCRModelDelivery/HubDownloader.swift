import Foundation
import HuggingFace

public struct HubRemoteFileInfo: Sendable, Equatable {
    public let etag: String?
    public let revision: String?

    public init(etag: String?, revision: String?) {
        self.etag = etag
        self.revision = revision
    }
}

public protocol HubDownloading: Sendable {
    func downloadSnapshot(
        of repo: Repo.ID,
        kind: Repo.Kind,
        revision: String,
        matching globs: [String],
        localFilesOnly: Bool,
        maxConcurrentDownloads: Int
    ) async throws -> URL

    func getFileInfo(
        at repoPath: String,
        in repo: Repo.ID,
        kind: Repo.Kind,
        revision: String
    ) async throws -> HubRemoteFileInfo
}

public struct HuggingFaceHubDownloader: HubDownloading {
    private let client: HubClient

    public init(client: HubClient) {
        self.client = client
    }

    public func downloadSnapshot(
        of repo: Repo.ID,
        kind: Repo.Kind,
        revision: String,
        matching globs: [String],
        localFilesOnly: Bool,
        maxConcurrentDownloads: Int
    ) async throws -> URL {
        try await client.downloadSnapshot(
            of: repo,
            kind: kind,
            revision: revision,
            matching: globs,
            localFilesOnly: localFilesOnly,
            maxConcurrentDownloads: maxConcurrentDownloads,
            progressHandler: nil
        )
    }

    public func getFileInfo(
        at repoPath: String,
        in repo: Repo.ID,
        kind: Repo.Kind,
        revision: String
    ) async throws -> HubRemoteFileInfo {
        let file = try await client.getFile(
            at: repoPath,
            in: repo,
            kind: kind,
            revision: revision
        )

        return HubRemoteFileInfo(
            etag: file.etag,
            revision: file.revision
        )
    }
}
