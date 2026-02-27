import Foundation

internal actor AsyncLimiter {
    private let limit: Int
    private var availablePermits: Int
    private var waiters: [CheckedContinuation<Void, Never>] = []

    internal init(limit: Int) {
        let boundedLimit = max(1, limit)
        self.limit = boundedLimit
        self.availablePermits = boundedLimit
    }

    internal func withPermit<T: Sendable>(
        _ operation: @Sendable () async throws -> T
    ) async throws -> T {
        await acquire()
        defer {
            release()
        }
        return try await operation()
    }

    private func acquire() async {
        if availablePermits > 0 {
            availablePermits -= 1
            return
        }

        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    private func release() {
        if let continuation = waiters.first {
            waiters.removeFirst()
            continuation.resume()
            return
        }

        availablePermits = min(availablePermits + 1, limit)
    }
}
