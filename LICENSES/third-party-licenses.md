# Third-party licenses

This repository is published under Apache-2.0 and vendors/transitively depends on the libraries below.

## Compatibility policy

- Apache-2.0, MIT, and BSD-3-Clause licenses are approved for distribution in this project.
- Dependencies with unverified licensing status are blockers until confirmed.

## Direct dependencies (from `Package.swift`)

| Dependency | License | Source |
| --- | --- | --- |
| swift-huggingface | Apache-2.0 | https://github.com/huggingface/swift-huggingface |
| mlx-swift | MIT | https://github.com/ml-explore/mlx-swift |
| swift-tokenizers | Apache-2.0 | https://github.com/DePasqualeOrg/swift-tokenizers |

## Transitive dependencies (from `Package.resolved`)

| Dependency | License | Source | License status |
| --- | --- | --- | --- |
| async-http-client | Apache-2.0 | https://github.com/swift-server/async-http-client | **verified** |
| EventSource | MIT | https://github.com/mattt/EventSource | **verified** |
| swift-algorithms | Apache-2.0 | https://github.com/apple/swift-algorithms | **verified** |
| swift-asn1 | Apache-2.0 | https://github.com/apple/swift-asn1 | **verified** |
| swift-async-algorithms | Apache-2.0 | https://github.com/apple/swift-async-algorithms | **verified** |
| swift-atomics | Apache-2.0 | https://github.com/apple/swift-atomics | **verified** |
| swift-certificates | Apache-2.0 | https://github.com/apple/swift-certificates | **verified** |
| swift-collections | Apache-2.0 | https://github.com/apple/swift-collections | **verified** |
| swift-configuration | Apache-2.0 | https://github.com/apple/swift-configuration | **verified** |
| swift-crypto | Apache-2.0 | https://github.com/apple/swift-crypto | **verified** |
| swift-distributed-tracing | Apache-2.0 | https://github.com/apple/swift-distributed-tracing | **verified** |
| swift-http-structured-headers | Apache-2.0 | https://github.com/apple/swift-http-structured-headers | **verified** |
| swift-http-types | Apache-2.0 | https://github.com/apple/swift-http-types | **verified** |
| swift-jinja | Apache-2.0 | https://github.com/huggingface/swift-jinja | **verified** |
| swift-log | Apache-2.0 | https://github.com/apple/swift-log | **verified** |
| swift-nio | Apache-2.0 | https://github.com/apple/swift-nio | **verified** |
| swift-nio-extras | Apache-2.0 | https://github.com/apple/swift-nio-extras | **verified** |
| swift-nio-http2 | Apache-2.0 | https://github.com/apple/swift-nio-http2 | **verified** |
| swift-nio-ssl | Apache-2.0 | https://github.com/apple/swift-nio-ssl | **verified** |
| swift-nio-transport-services | Apache-2.0 | https://github.com/apple/swift-nio-transport-services | **verified** |
| swift-numerics | Apache-2.0 | https://github.com/apple/swift-numerics | **verified** |
| swift-service-context | Apache-2.0 | https://github.com/apple/swift-service-context | **verified** |
| swift-service-lifecycle | Apache-2.0 | https://github.com/swift-server/swift-service-lifecycle | **verified** |
| swift-system | Apache-2.0 | https://github.com/apple/swift-system | **verified** |
| yyjson | MIT | https://github.com/ibireme/yyjson | **verified** |
| swift-xet | unverified | https://github.com/mattt/swift-xet | **BLOCKER** |

## Vendored binary component

| Component | License |
| --- | --- |
| Vendor/PDFium/PDFium.xcframework | BSD-3-Clause | [LICENSES/PDFium-BSD-3-Clause.txt](LICENSES/PDFium-BSD-3-Clause.txt) |

## Blocking items before release

- Resolve and verify `swift-xet` license. Publishing should be blocked until this dependency is confirmed as compatible and documented here.
