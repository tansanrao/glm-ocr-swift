# ``GlmOCRSwift``

Swift package for strict on-device GLM-OCR pipeline integration.

This is a **strict Swift port of the Python GLM-OCR SDK from https://github.com/zai-org/GLM-OCR**.

## Overview

`GlmOCRSwift` defines the public API contract for:

- document input types
- layout and recognition protocol boundaries
- parse result structures
- actor-based pipeline orchestration

The end-to-end pipeline path is implemented for `.image`, `.imageData`, and `.pdfData` inputs with layout-driven region recognition, markdown/json composition, and diagnostics output.

Local execution behavior:

- No-layout recognition uses `GlmOCRConfig.prompts.noLayoutPrompt`.
- Layout recognition uses task prompt mapping (`text` / `table` / `formula`) from `GlmOCRConfig.prompts`.
- PDF rendering uses `GlmOCRConfig.pdfDPI` and `GlmOCRConfig.pdfMaxRenderedLongSide`.
- Effective page cap:
  - if both `ParseOptions.maxPages` and `GlmOCRConfig.defaultMaxPages` are set, the smaller value is used
  - if only one is set, that value is used
  - if neither is set, there is no page cap
- `GlmOCRConfig.defaultMaxPages` applies to PDF inputs only.

## Topics

### Configuration

- ``GlmOCRConfig``
- ``GlmOCRRecognitionOptions``
- ``GlmOCRPromptConfig``
- ``ParseOptions``

### Pipeline

- ``GlmOCRPipeline``
- ``InputDocument``

### Results

- ``LayoutRegion``
- ``OCRRegion``
- ``OCRPageResult``
- ``OCRDocumentResult``
- ``ParseDiagnostics``

### Protocols

- ``LayoutDetector``
- ``RegionRecognizer``
