import XCTest

@testable import GlmOCRSwift

final class ParseOptionsTests: XCTestCase {
    func testDefaultsMatchExpectation() {
        let options = ParseOptions()

        XCTAssertTrue(options.includeMarkdown)
        XCTAssertTrue(options.includeDiagnostics)
        XCTAssertNil(options.maxPages)
    }

    func testValidationRejectsNonPositiveMaxPages() {
        XCTAssertThrowsError(try ParseOptions(maxPages: 0).validate())
    }

    func testDecodingFallbackToDefaults() throws {
        let blankJson = "{}"
        let parsed = try JSONDecoder().decode(ParseOptions.self, from: Data(blankJson.utf8))

        XCTAssertTrue(parsed.includeMarkdown)
        XCTAssertTrue(parsed.includeDiagnostics)
        XCTAssertNil(parsed.maxPages)
    }

    func testPublicContractStructsRemainEquatable() {
        let diagnostics = ParseDiagnostics()
        let region = OCRRegion(index: 0, label: "text", content: "ok", bbox2D: nil)
        let page = OCRPageResult(regions: [region])
        let result = OCRDocumentResult(pages: [page], markdown: "ok", diagnostics: diagnostics)

        XCTAssertEqual(region, OCRRegion(index: 0, label: "text", content: "ok", bbox2D: nil))
        XCTAssertEqual(page, OCRPageResult(regions: [region]))
        XCTAssertEqual(result, OCRDocumentResult(pages: [page], markdown: "ok", diagnostics: diagnostics))
    }
}
