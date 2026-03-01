import CoreGraphics
import Foundation
import XCTest

import GlmOCRPDFium

final class PDFiumSmokeTests: XCTestCase {
    func testPDFiumCanLoadAndRasterizeGeneratedPDF() throws {
        let pdfData = try makeSinglePagePDFData()

        let document = try PDFiumDocument(data: pdfData)
        XCTAssertEqual(document.pageCount, 1)

        let page = try document.page(at: 0)
        let rasterizer = PDFiumPageRasterizer()
        let image = try rasterizer.render(
            page: page,
            dpi: 144.0,
            maxRenderedLongSide: 2_048.0
        )

        XCTAssertGreaterThan(image.width, 0)
        XCTAssertGreaterThan(image.height, 0)
    }

    private func makeSinglePagePDFData() throws -> Data {
        let output = NSMutableData()
        var mediaBox = CGRect(x: 0, y: 0, width: 200, height: 240)

        guard let consumer = CGDataConsumer(data: output as CFMutableData),
              let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil)
        else {
            throw SmokeTestError.unableToCreatePDFContext
        }

        context.beginPDFPage(nil)
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(mediaBox)

        let insetRect = mediaBox.insetBy(dx: 24, dy: 24)
        context.setFillColor(CGColor(red: 0.2, green: 0.4, blue: 0.9, alpha: 1))
        context.fill(insetRect)
        context.endPDFPage()
        context.closePDF()

        return output as Data
    }
}

private enum SmokeTestError: Error {
    case unableToCreatePDFContext
}
