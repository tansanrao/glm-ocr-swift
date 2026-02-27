import Foundation

internal enum PPDocLayoutMLXError: Error, Sendable, Equatable {
    case invalidModelID(String)
    case modelDownloadFailed(String)
    case missingModelFile(String)
    case modelConfigurationDecodeFailed(String)
    case modelInitializationFailed(String)
    case invalidInputShape(expected: [Int], actual: [Int])
    case missingOutput(String)
    case invalidOutputShape(name: String, expected: [Int], actual: [Int])
    case preprocessFailed(String)
}
