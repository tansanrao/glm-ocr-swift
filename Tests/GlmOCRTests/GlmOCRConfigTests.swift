import XCTest

@testable import GlmOCRSwift

final class GlmOCRConfigTests: XCTestCase {
    func testDefaultValuesMatchExpected() {
        let config = GlmOCRConfig()

        XCTAssertEqual(config.recognizerModelID, "mlx-community/GLM-OCR-bf16")
        XCTAssertEqual(config.layoutModelID, "PaddlePaddle/PP-DocLayoutV3_safetensors")
        XCTAssertEqual(config.maxConcurrentRecognitions, 1)
        XCTAssertTrue(config.enableLayout)
        XCTAssertNil(config.defaultMaxPages)
        XCTAssertEqual(config.recognitionOptions.maxTokens, 4_096)
        XCTAssertEqual(config.recognitionOptions.temperature, 0.8)
        XCTAssertEqual(config.recognitionOptions.prefillStepSize, 2_048)
        XCTAssertEqual(config.recognitionOptions.topP, 0.9)
        XCTAssertEqual(config.recognitionOptions.topK, 50)
        XCTAssertEqual(config.recognitionOptions.repetitionPenalty, 1.1)
    }

    func testDefaultConfigPassesValidation() throws {
        let config = GlmOCRConfig()
        try config.validate()
    }

    func testInvalidConfigurationRejectsEmptyModelIDs() {
        XCTAssertThrowsError(try GlmOCRConfig(
            recognizerModelID: "   ",
            layoutModelID: "    "
        ).validate())
    }

    func testRecognitionOptionsRoundTrip() throws {
        let config = GlmOCRConfig()
        let encoded = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(GlmOCRConfig.self, from: encoded)

        XCTAssertEqual(decoded, config)
    }
}
