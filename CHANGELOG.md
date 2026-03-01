# Changelog

## Unreleased

- Added iOS PDFium support via vendored multi-platform `PDFium.xcframework` slices (`ios-arm64`, `ios-arm64-simulator`, `macos-arm64`).
- Updated PDF page rendering selection to use `GlmOCRPDFium` whenever it is linked into the build.
- Added `PDFiumSmokeTests` covering in-memory PDF creation + PDFium page rasterization.
- Updated package platform minimums to iOS 26+ and macOS 26+.
- Added non-macOS benchmark CLI entrypoint guard so package iOS build/test actions compile cleanly.

## 0.2.0

- Added markdown bundle export config (`GlmOCRMarkdownBundleConfig`) enabled by default.
- Added in-memory bundle output (`OCRMarkdownBundle`) with:
  - rewritten markdown figure references,
  - HEIC figure assets (`OCRFigureAsset`),
  - JSON sidecar payload.
- Layout-mode parses now export figure crops for `image` and `chart` labels when bundle mode is enabled.
- Added `GlmOCRConfig.performance` with batch/inflight/render/preprocess/bundle concurrency controls.
- Added batch-first OCR inference scheduling with ordered result merge and per-batch diagnostics.
- Added batched recognizer inference path (`recognizeBatch`) in the inference client and MLX runtime.
- Added truthful split layout timings (`layout_preprocess`, `layout_inference`, `layout_postprocess`) and new diagnostics counters.
- Added bounded parallelization for PDF page rendering, OCR crop preprocessing, and markdown figure encoding.
- Added layout postprocess fast path for contour boundary extraction with compatibility toggle.
- Added `GlmOCRBenchmark` executable target for end-to-end PDF parse benchmarking and JSON output.

## 0.1.0

Initial public release of `GlmOCRSwift`.
