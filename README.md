# GLM-OCR Swift

`GlmOCRSwift` is a **strict Swift port of the Python GLM-OCR SDK from https://github.com/zai-org/GLM-OCR**.

The package is local-only: all model loading and inference run directly in Swift/MLX with no Python runtime dependency.

## Install

Add the package dependency in your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/<owner>/glm-ocr-swift.git", from: "0.1.0")
],
products: [
    .library(name: "YourApp", targets: ["YourApp"])
]
```

Then add `GlmOCRSwift` as a dependency of your target.

## Minimal usage

```swift
import GlmOCRSwift

let config = GlmOCRConfig()
let options = ParseOptions()

let pipeline = try await GlmOCRPipeline(config: config)
let imageURL = URL(filePath: "/tmp/page.png")
let data = try Data(contentsOf: imageURL)
let result = try await pipeline.parse(.imageData(data), options: options)

print(result.markdown)
```

`result.markdown` contains the formatted OCR output. Diagnostics are available in `result.diagnostics`.

## Supported platforms and model behavior

- Platforms: iOS 17+, macOS 14+
- Layout and recognition models are downloaded and cached locally on first use.
- Default model ids:
  - `recognizerModelID`: `mlx-community/GLM-OCR-bf16`
  - `layoutModelID`: `PaddlePaddle/PP-DocLayoutV3_safetensors`
- PDF behavior:
  - `ParseOptions.maxPages` caps selected pages.
  - `GlmOCRConfig.defaultMaxPages` applies as a secondary PDF page cap.
  - If both are set, the smaller page cap is used.

## API docs

- [Swift documentation index](Sources/GlmOCRSwift/GlmOCRSwift.docc/GlmOCRSwift.md)

## Release

Initial release: `v0.1.0`.
