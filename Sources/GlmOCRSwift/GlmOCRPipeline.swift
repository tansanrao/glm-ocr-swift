import CoreGraphics
import CryptoKit
import Foundation

import GlmOCRModelDelivery

public actor GlmOCRPipeline {
    public nonisolated let config: GlmOCRConfig
    internal nonisolated let runtimeConfig: GlmOCRConfig
    private let pageLoader: any PipelinePageLoading
    private let layoutDetector: any PipelineLayoutDetecting
    private let regionRecognizer: any RegionRecognizer
    private let regionCropper: any PipelineRegionCropping
    private let formatter: PipelineFormatter

    public init(config: GlmOCRConfig) async throws {
        let modelManager: SandboxModelManager
        do {
            modelManager = try SandboxModelManager()
        } catch {
            throw GlmOCRError.modelDeliveryFailed(error.localizedDescription)
        }

        try await self.init(
            config: config,
            modelDeliveryManager: modelManager
        )
    }

    internal init(
        config: GlmOCRConfig,
        modelDeliveryManager: any ModelDeliveryManaging
    ) async throws {
        try config.validate()

        let resolved: ModelDeliveryResolvedPaths
        do {
            resolved = try await modelDeliveryManager.ensureReady(
                config: ModelDeliveryRequest(
                    recognizerModelID: config.recognizerModelID,
                    layoutModelID: config.layoutModelID
                )
            )
        } catch {
            throw GlmOCRError.modelDeliveryFailed(error.localizedDescription)
        }

        var runtimeConfig = config
        runtimeConfig.recognizerModelID = resolved.recognizerModelDirectory.path
        runtimeConfig.layoutModelID = resolved.layoutModelDirectory.path

        self.config = config
        self.runtimeConfig = runtimeConfig
        self.pageLoader = PipelinePageLoader(
            pdfDPI: runtimeConfig.pdfDPI,
            maxRenderedLongSide: runtimeConfig.pdfMaxRenderedLongSide,
            defaultMaxPages: runtimeConfig.defaultMaxPages
        )
        self.layoutDetector = PPDocLayoutMLXDetector(config: runtimeConfig)
        self.regionRecognizer = GLMRegionRecognizer(config: runtimeConfig)
        self.regionCropper = PipelineRegionCropper()
        self.formatter = PipelineFormatter()
    }

    internal init(
        config: GlmOCRConfig,
        pageLoader: any PipelinePageLoading,
        layoutDetector: any PipelineLayoutDetecting,
        regionRecognizer: any RegionRecognizer,
        regionCropper: any PipelineRegionCropping = PipelineRegionCropper(),
        formatter: PipelineFormatter = PipelineFormatter()
    ) throws {
        try config.validate()
        self.config = config
        self.runtimeConfig = config
        self.pageLoader = pageLoader
        self.layoutDetector = layoutDetector
        self.regionRecognizer = regionRecognizer
        self.regionCropper = regionCropper
        self.formatter = formatter
    }

    public func parse(_ input: InputDocument, options: ParseOptions) async throws -> OCRDocumentResult {
        try options.validate()
        try Task.checkCancellation()

        var warnings: [String] = []
        var timingsMs: [String: Double] = [:]
        let totalStart = Date()
        let effectiveMaxPages = resolvedEffectiveMaxPages(
            optionsMaxPages: options.maxPages,
            defaultMaxPages: config.defaultMaxPages
        )

        let pageLoadStart = Date()
        let pages = try pageLoader.loadPages(
            from: input,
            maxPages: options.maxPages
        )
        timingsMs["page_load"] = elapsedMilliseconds(since: pageLoadStart)
        let pageRenderDebug = dumpPageRenderDiagnosticsIfRequested(
            input: input,
            pages: pages
        )

        guard !pages.isEmpty else {
            throw GlmOCRError.invalidConfiguration("Input document produced zero pages")
        }

        let pagesAndMarkdown: (pages: [OCRPageResult], markdown: String)
        if config.enableLayout {
            let layoutStart = Date()
            let detailedDetections = try await layoutDetector.detectDetailed(
                pages: pages,
                options: options
            )
            let layoutDuration = elapsedMilliseconds(since: layoutStart)
            timingsMs["layout_preprocess"] = layoutDuration
            timingsMs["layout_inference"] = layoutDuration
            timingsMs["layout_postprocess"] = layoutDuration

            guard detailedDetections.count == pages.count else {
                throw GlmOCRError.invalidConfiguration(
                    "Layout detector produced \(detailedDetections.count) pages for \(pages.count) input pages"
                )
            }

            let ocrPreprocessStart = Date()
            let ocrPreprocessed = try ocrPreprocessLayoutRegions(
                pages: pages,
                detections: detailedDetections
            )
            warnings.append(contentsOf: ocrPreprocessed.warnings)
            timingsMs["ocr_preprocess"] = elapsedMilliseconds(since: ocrPreprocessStart)

            let ocrPreprocessOnly = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_OCR_PREPROCESS_ONLY"] == "1"
            let merged: (pageRegions: [[PipelineRegionRecord]], warnings: [String])
            if ocrPreprocessOnly {
                timingsMs["ocr_inference"] = 0
                merged = (ocrPreprocessed.pageRegions, [])
            } else {
                let ocrInferenceStart = Date()
                let ocrInferred = try await ocrInferenceRecognizeQueuedRegions(
                    jobs: ocrPreprocessed.recognitionJobs
                )
                timingsMs["ocr_inference"] = elapsedMilliseconds(since: ocrInferenceStart)
                merged = ocrInferenceMergeResults(
                    pageRegions: ocrPreprocessed.pageRegions,
                    inferenceResults: ocrInferred
                )
            }
            warnings.append(contentsOf: merged.warnings)
            dumpOCRPostprocessInputIfRequested(pageRegions: merged.pageRegions)

            let ocrPostprocessStart = Date()
            pagesAndMarkdown = ocrPostprocessFormatLayout(pageRegions: merged.pageRegions)
            timingsMs["ocr_postprocess"] = elapsedMilliseconds(since: ocrPostprocessStart)
        } else {
            timingsMs["ocr_preprocess"] = 0
            let ocrInferenceStart = Date()
            let recognizedContents = try await recognizeWholePages(
                pages: pages
            )
            warnings.append(contentsOf: recognizedContents.warnings)
            timingsMs["ocr_inference"] = elapsedMilliseconds(since: ocrInferenceStart)

            let ocrPostprocessStart = Date()
            pagesAndMarkdown = ocrPostprocessFormatNoLayout(contents: recognizedContents.contents)
            timingsMs["ocr_postprocess"] = elapsedMilliseconds(since: ocrPostprocessStart)
        }

        timingsMs["total"] = elapsedMilliseconds(since: totalStart)

        let metadata: [String: String] = [
            "layoutEnabled": config.enableLayout ? "true" : "false",
            "pageCount": String(pages.count),
            "maxConcurrentRecognitions": String(config.maxConcurrentRecognitions),
            "maxPagesOption": options.maxPages.map(String.init) ?? "nil",
            "defaultMaxPages": config.defaultMaxPages.map(String.init) ?? "nil",
            "effectiveMaxPages": effectiveMaxPages.map(String.init) ?? "nil",
            "pdfDPI": String(config.pdfDPI),
            "pdfMaxRenderedLongSide": String(config.pdfMaxRenderedLongSide),
            "noLayoutPromptHash": truncatedSHA256(config.prompts.noLayoutPrompt),
            "textPromptHash": truncatedSHA256(config.prompts.textPrompt),
            "tablePromptHash": truncatedSHA256(config.prompts.tablePrompt),
            "formulaPromptHash": truncatedSHA256(config.prompts.formulaPrompt),
            "pageRenderDebugDump": pageRenderDebug.path ?? "nil",
            "pageRenderDebugCount": pageRenderDebug.path == nil ? "0" : String(pageRenderDebug.count)
        ]

        let diagnostics: ParseDiagnostics
        if options.includeDiagnostics {
            diagnostics = ParseDiagnostics(
                warnings: warnings,
                timingsMs: timingsMs,
                metadata: metadata
            )
        } else {
            diagnostics = ParseDiagnostics()
        }

        return OCRDocumentResult(
            pages: pagesAndMarkdown.pages,
            markdown: options.includeMarkdown ? pagesAndMarkdown.markdown : "",
            diagnostics: diagnostics
        )
    }

    private func recognizeWholePages(
        pages: [CGImage]
    ) async throws -> (contents: [String], warnings: [String]) {
        guard !pages.isEmpty else {
            return ([], [])
        }

        let limiter = AsyncLimiter(limit: config.maxConcurrentRecognitions)
        let recognizer = regionRecognizer
        let promptRecognizer = recognizer as? PromptRegionRecognizing

        var results = Array(repeating: "", count: pages.count)
        var warnings: [String] = []
        let jobs = pages.enumerated().map { index, image in
            PipelineRecognitionJob(
                key: PipelineRecognitionJobKey(pageIndex: index, regionPosition: 0),
                image: image,
                task: .text,
                promptOverride: config.prompts.noLayoutPrompt
            )
        }

        try await withThrowingTaskGroup(of: (Int, Result<String, Error>).self) { group in
            for job in jobs {
                group.addTask {
                    do {
                        try Task.checkCancellation()
                        let recognized = try await limiter.withPermit {
                            try await self.recognize(
                                job: job,
                                recognizer: recognizer,
                                promptRecognizer: promptRecognizer
                            )
                        }
                        return (job.key.pageIndex, .success(recognized))
                    } catch {
                        if error is CancellationError {
                            throw error
                        }
                        return (job.key.pageIndex, .failure(error))
                    }
                }
            }

            for try await (pageIndex, result) in group {
                switch result {
                case .success(let text):
                    results[pageIndex] = text
                case .failure(let error):
                    results[pageIndex] = ""
                    warnings.append("page[\(pageIndex)] recognition failed: \(error)")
                }
            }
        }

        return (results, warnings)
    }

    private func ocrPreprocessLayoutRegions(
        pages: [CGImage],
        detections: [[PipelineLayoutRegion]]
    ) throws -> OCRPreprocessOutput {
        let pageUnits = ocrPreprocessBuildPageUnits(pages: pages, detections: detections)
        return try ocrPreprocessCropAndQueueRegions(units: pageUnits)
    }

    private func ocrPreprocessBuildPageUnits(
        pages: [CGImage],
        detections: [[PipelineLayoutRegion]]
    ) -> [OCRPreprocessPageUnit] {
        pages.indices.map { pageIndex in
            return OCRPreprocessPageUnit(
                pageIndex: pageIndex,
                image: pages[pageIndex],
                detections: detections[pageIndex]
            )
        }
    }

    private func ocrPreprocessCropAndQueueRegions(
        units: [OCRPreprocessPageUnit]
    ) throws -> OCRPreprocessOutput {
        var pageRegions: [[PipelineRegionRecord]] = Array(repeating: [], count: units.count)
        var recognitionJobs: [PipelineRecognitionJob] = []
        var warnings: [String] = []
        var ocrPreprocessDebugEntries: [[String: Any]] = []
        let ocrPreprocessDumpPath = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_OCR_PREPROCESS_DUMP"]
        let shouldDumpOCRPreprocess = ocrPreprocessDumpPath?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty == false

        for unit in units {
            try Task.checkCancellation()

            for detection in unit.detections {
                if detection.task == .abandon {
                    continue
                }

                let regionPosition = pageRegions[unit.pageIndex].count
                var record = PipelineRegionRecord(
                    index: detection.index,
                    nativeLabel: detection.label,
                    task: detection.task,
                    bbox2D: detection.bbox2D,
                    content: nil
                )

                let debugEntry: [String: Any]? = shouldDumpOCRPreprocess ? [
                    "pageIndex": unit.pageIndex,
                    "detectionIndex": detection.index,
                    "order": detection.order,
                    "regionPosition": regionPosition,
                    "task": detection.task.rawValue,
                    "nativeLabel": detection.label,
                    "bbox2D": detection.bbox2D,
                    "polygon2D": detection.polygon2D
                ] : nil

                if let ocrTask = detection.task.ocrTask {
                    do {
                        let cropResult = try regionCropper.cropRegion(
                            page: unit.image,
                            bbox2D: detection.bbox2D,
                            polygon2D: detection.polygon2D,
                            pageIndex: unit.pageIndex,
                            regionIndex: detection.index
                        )

                        if let warning = cropResult.warning {
                            warnings.append(warning)
                        }

                        recognitionJobs.append(
                            PipelineRecognitionJob(
                                key: PipelineRecognitionJobKey(
                                    pageIndex: unit.pageIndex,
                                    regionPosition: regionPosition
                                ),
                                image: cropResult.image,
                                task: ocrTask
                            )
                        )

                        if var debugEntry {
                            if let cropMetadata = cropDebugMetadata(for: cropResult.image) {
                                debugEntry["cropWidth"] = cropMetadata.width
                                debugEntry["cropHeight"] = cropMetadata.height
                                debugEntry["cropChannels"] = cropMetadata.channels
                                debugEntry["cropPixelSHA256"] = cropMetadata.sha256
                            } else {
                                debugEntry["cropWidth"] = NSNull()
                                debugEntry["cropHeight"] = NSNull()
                                debugEntry["cropChannels"] = NSNull()
                                debugEntry["cropPixelSHA256"] = NSNull()
                            }
                            ocrPreprocessDebugEntries.append(debugEntry)
                        }
                    } catch {
                        record.content = ""
                        warnings.append(
                            "page[\(unit.pageIndex)] region[\(detection.index)] crop failed: \(error)"
                        )
                        if var debugEntry {
                            debugEntry["cropWidth"] = NSNull()
                            debugEntry["cropHeight"] = NSNull()
                            debugEntry["cropChannels"] = NSNull()
                            debugEntry["cropPixelSHA256"] = NSNull()
                            ocrPreprocessDebugEntries.append(debugEntry)
                        }
                    }
                } else if var debugEntry {
                    debugEntry["cropWidth"] = NSNull()
                    debugEntry["cropHeight"] = NSNull()
                    debugEntry["cropChannels"] = NSNull()
                    debugEntry["cropPixelSHA256"] = NSNull()
                    ocrPreprocessDebugEntries.append(debugEntry)
                }

                pageRegions[unit.pageIndex].append(record)
            }
        }

        if let ocrPreprocessDumpPath,
           !ocrPreprocessDumpPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           let data = try? JSONSerialization.data(withJSONObject: ocrPreprocessDebugEntries, options: [.prettyPrinted])
        {
            try? data.write(to: URL(fileURLWithPath: ocrPreprocessDumpPath), options: .atomic)
        }

        return OCRPreprocessOutput(
            pageRegions: pageRegions,
            recognitionJobs: recognitionJobs,
            warnings: warnings
        )
    }

    private func ocrInferenceRecognizeQueuedRegions(
        jobs: [PipelineRecognitionJob]
    ) async throws -> OCRInferenceOutput {
        let recognizer = regionRecognizer
        let promptRecognizer = recognizer as? PromptRegionRecognizing
        let limiter = AsyncLimiter(limit: config.maxConcurrentRecognitions)

        var recognitionResults: [PipelineRecognitionJobKey: Result<String, Error>] = [:]
        recognitionResults.reserveCapacity(jobs.count)

        try await withThrowingTaskGroup(of: (PipelineRecognitionJobKey, Result<String, Error>).self) { group in
            for job in jobs {
                group.addTask {
                    do {
                        try Task.checkCancellation()
                        let recognized = try await limiter.withPermit {
                            try await self.recognize(
                                job: job,
                                recognizer: recognizer,
                                promptRecognizer: promptRecognizer
                            )
                        }
                        return (job.key, .success(recognized))
                    } catch {
                        if error is CancellationError {
                            throw error
                        }
                        return (job.key, .failure(error))
                    }
                }
            }

            for try await (key, result) in group {
                recognitionResults[key] = result
            }
        }

        return OCRInferenceOutput(results: recognitionResults)
    }

    private func ocrInferenceMergeResults(
        pageRegions: [[PipelineRegionRecord]],
        inferenceResults: OCRInferenceOutput
    ) -> (pageRegions: [[PipelineRegionRecord]], warnings: [String]) {
        var mergedPageRegions = pageRegions
        var warnings: [String] = []

        for (key, result) in inferenceResults.results {
            guard key.pageIndex < mergedPageRegions.count,
                  key.regionPosition < mergedPageRegions[key.pageIndex].count
            else {
                continue
            }

            switch result {
            case .success(let text):
                mergedPageRegions[key.pageIndex][key.regionPosition].content = text
            case .failure(let error):
                mergedPageRegions[key.pageIndex][key.regionPosition].content = ""
                let regionIndex = mergedPageRegions[key.pageIndex][key.regionPosition].index
                warnings.append(
                    "page[\(key.pageIndex)] region[\(regionIndex)] recognition failed: \(error)"
                )
            }
        }

        return (mergedPageRegions, warnings)
    }

    private func ocrPostprocessFormatLayout(
        pageRegions: [[PipelineRegionRecord]]
    ) -> (pages: [OCRPageResult], markdown: String) {
        formatter.formatLayout(pageRegions: pageRegions)
    }

    private func ocrPostprocessFormatNoLayout(
        contents: [String]
    ) -> (pages: [OCRPageResult], markdown: String) {
        formatter.formatNoLayout(contents: contents)
    }

    private func recognize(
        job: PipelineRecognitionJob,
        recognizer: any RegionRecognizer,
        promptRecognizer: (any PromptRegionRecognizing)?
    ) async throws -> String {
        if let promptRecognizer,
           let prompt = job.promptOverride?.trimmingCharacters(in: .whitespacesAndNewlines),
           !prompt.isEmpty
        {
            return try await promptRecognizer.recognize(job.image, prompt: prompt)
        }

        return try await recognizer.recognize(job.image, task: job.task)
    }

    private func elapsedMilliseconds(since start: Date) -> Double {
        Date().timeIntervalSince(start) * 1000.0
    }

    private func resolvedEffectiveMaxPages(
        optionsMaxPages: Int?,
        defaultMaxPages: Int?
    ) -> Int? {
        // Page cap semantics: if both are present, the smaller cap wins.
        if let optionsMaxPages, let defaultMaxPages {
            return min(optionsMaxPages, defaultMaxPages)
        }
        return optionsMaxPages ?? defaultMaxPages
    }

    private func dumpPageRenderDiagnosticsIfRequested(
        input: InputDocument,
        pages: [CGImage]
    ) -> (path: String?, count: Int) {
        guard case .pdfData = input else {
            return (nil, 0)
        }
        guard let dumpPath = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_PAGE_RENDER_DUMP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !dumpPath.isEmpty
        else {
            return (nil, 0)
        }

        let entries: [[String: Any]] = pages.enumerated().map { index, image in
            var payload: [String: Any] = [
                "pageIndex": index,
                "width": image.width,
                "height": image.height
            ]
            if let metadata = cropDebugMetadata(for: image) {
                payload["rgbSHA256"] = metadata.sha256
            } else {
                payload["rgbSHA256"] = NSNull()
            }
            return payload
        }

        if JSONSerialization.isValidJSONObject(entries),
           let data = try? JSONSerialization.data(withJSONObject: entries, options: [.prettyPrinted])
        {
            try? data.write(to: URL(fileURLWithPath: dumpPath), options: .atomic)
        }

        return (dumpPath, entries.count)
    }

    private func truncatedSHA256(_ value: String, length: Int = 16) -> String {
        let digest = SHA256.hash(data: Data(value.utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return String(hex.prefix(max(1, length)))
    }

    private func cropDebugMetadata(for image: CGImage) -> (width: Int, height: Int, channels: Int, sha256: String)? {
        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else {
            return nil
        }

        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            return nil
        }

        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for pixel in 0 ..< (width * height) {
            let rgbaOffset = pixel * 4
            let rgbOffset = pixel * 3
            rgb[rgbOffset] = rgba[rgbaOffset]
            rgb[rgbOffset + 1] = rgba[rgbaOffset + 1]
            rgb[rgbOffset + 2] = rgba[rgbaOffset + 2]
        }

        let digest = SHA256.hash(data: Data(rgb))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return (width, height, 3, hex)
    }

    private func dumpOCRPostprocessInputIfRequested(pageRegions: [[PipelineRegionRecord]]) {
        guard let dumpPath = ProcessInfo.processInfo.environment["GLMOCR_DEBUG_OCR_POSTPROCESS_INPUT_DUMP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !dumpPath.isEmpty
        else {
            return
        }

        let payload: [[[String: Any]]] = pageRegions.map { page in
            page.map { region in
                var entry: [String: Any] = [
                    "index": region.index,
                    "nativeLabel": region.nativeLabel,
                    "task": region.task.rawValue
                ]
                if let bbox2D = region.bbox2D {
                    entry["bbox2D"] = bbox2D
                } else {
                    entry["bbox2D"] = NSNull()
                }
                if let content = region.content {
                    entry["content"] = content
                } else {
                    entry["content"] = NSNull()
                }
                return entry
            }
        }

        guard JSONSerialization.isValidJSONObject(payload),
              let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
        else {
            return
        }

        try? data.write(to: URL(fileURLWithPath: dumpPath), options: .atomic)
    }
}

private struct OCRPreprocessPageUnit: @unchecked Sendable {
    let pageIndex: Int
    let image: CGImage
    let detections: [PipelineLayoutRegion]
}

private struct OCRPreprocessOutput: @unchecked Sendable {
    let pageRegions: [[PipelineRegionRecord]]
    let recognitionJobs: [PipelineRecognitionJob]
    let warnings: [String]
}

private struct OCRInferenceOutput: @unchecked Sendable {
    let results: [PipelineRecognitionJobKey: Result<String, Error>]
}
