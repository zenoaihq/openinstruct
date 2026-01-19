# Contributing to OpenInstruct

Thank you for your interest in contributing to OpenInstruct!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/zenoai/openinstruct.git
cd openinstruct
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest tests_openinstruct/ -v

# Run specific test file
pytest tests_openinstruct/test_usage.py -v
```

## Code Style

We use `ruff` for linting and formatting:

```bash
# Check code
ruff check .

# Format code
ruff format .
```

## Type Checking

```bash
mypy openinstruct/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Adding a New Provider

1. Create a new file in `openinstruct/providers/`
2. Implement the `BaseProvider` interface
3. Register in `providers/__init__.py`
4. Add tests in `tests_openinstruct/`
5. Update README with new provider

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
