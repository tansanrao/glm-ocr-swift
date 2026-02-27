import XCTest

@testable import GlmOCRSwift

final class GlmOCRConfigTests: XCTestCase {
    func testDefaultValuesMatchExpected() {
        let config = GlmOCRConfig()

        XCTAssertEqual(config.recognizerModelID, "mlx-community/GLM-OCR-bf16")
        XCTAssertEqual(config.layoutModelID, "PaddlePaddle/PP-DocLayoutV3_safetensors")
        XCTAssertEqual(config.maxConcurrentRecognitions, 2)
        XCTAssertTrue(config.enableLayout)
        XCTAssertNil(config.defaultMaxPages)
    }

    func testDefaultConfigPassesValidation() throws {
        var config = GlmOCRConfig()
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
