# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-01-19

### Added

- **Token Usage Tracking**: New `return_usage=True` parameter returns `ExtractionResult` with token statistics
- **Enhanced Retry Configuration**: `RetryConfig` dataclass with exponential backoff and `on_retry` callback
- **New Types**: `TokenUsage`, `ExtractionResult`, `RetryConfig` exported from package

### Changed

- Improved TSON import (removed hardcoded sys.path)
- Updated README with comprehensive documentation
- Version bumped to 1.1.0

## [1.0.0] - 2025-01-18

### Added

- Initial release
- Multi-provider gateway (OpenAI, Anthropic, Google, Groq, Together, Mistral, Ollama, OpenRouter)
- TSON optimization for 30-70% token savings
- Pydantic model extraction with automatic validation
- Input context optimization via TSON conversion
- JSON fallback mechanism for robust extraction
- Async client support (`AsyncOpenInstruct`)
- List/array extraction support
- Nested model extraction
