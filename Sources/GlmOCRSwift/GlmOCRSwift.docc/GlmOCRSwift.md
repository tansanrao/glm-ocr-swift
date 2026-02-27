# ``GlmOCRSwift``

On-device OCR pipeline for Swift.

`GlmOCRSwift` ports the GLM-OCR workflow to native Swift + MLX.
It orchestrates page loading, optional layout detection, region recognition, and markdown/result assembly.

## Overview

Use ``GlmOCRPipeline`` as the primary entry point.

Execution is local:

- inputs: ``InputDocument`` (`.image`, `.imageData`, `.pdfData`)
- optional layout stage: `PP-DocLayoutV3` MLX backend
- recognition stage: `GLM-OCR` MLX backend
- formatting: region/page output + markdown
- diagnostics: timings, metadata, warnings

Page cap behavior:

- both set: `min(ParseOptions.maxPages, GlmOCRConfig.defaultMaxPages)`
- one set: use that value
- none set: no cap

## Topics

### Essentials

- ``GlmOCRPipeline``
- ``GlmOCRConfig``
- ``ParseOptions``

### Configuration and prompts

- ``GlmOCRRecognitionOptions``
- ``GlmOCRPromptConfig``
- ``GlmOCRLayoutConfig``

### Input and output

- ``InputDocument``
- ``LayoutRegion``
- ``OCRRegion``
- ``OCRPageResult``
- ``OCRDocumentResult``
- ``ParseDiagnostics``

### Extension protocols

- ``LayoutDetector``
- ``RegionRecognizer``

### Errors

- ``GlmOCRError``

### Articles

- <doc:Architecture>
