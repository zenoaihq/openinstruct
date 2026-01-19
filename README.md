# OpenInstruct

**Structured outputs for LLMs with 30-70% token savings**

Extract structured data from any LLM. TSON optimization reduces token costs while maintaining type safety.

[![PyPI version](https://badge.fury.io/py/openinstruct.svg)](https://pypi.org/project/openinstruct/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why OpenInstruct?

Getting structured data from LLMs is expensive and complex:

```python
# ❌ Without OpenInstruct: Manual JSON, verbose prompts, wasted tokens
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "..."}],
    tools=[{
        "type": "function",
        "function": {
            "name": "extract_user",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            },
        },
    }],
)
# Parse response manually
tool_call = response.choices[0].message.tool_calls[0]
user_data = json.loads(tool_call.function.arguments)
# Validate manually...
```

```python
# ✅ With OpenInstruct: Simple, validated, 30-70% fewer tokens with large payload
from openinstruct import OpenInstruct
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

client = OpenInstruct.from_provider("openai/gpt-4o")
user = client.extract(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John, 25 years old"}],
)
# user.name = "John", user.age = 25 ✅ Validated & typed
```

---

## Install

```bash
pip install openinstruct
```

---

## Token Savings

OpenInstruct uses TSON (Token-efficient Structured Object Notation) to reduce token consumption:

| Format | Tokens | Savings |
|--------|--------|---------|
| JSON | `{"name": "Alice", "age": 30}` | - |
| TSON | `{@name,age\|Alice,30}` | **~50%** |

For arrays of objects, savings can reach **70%+**.

### When NOT to Use TSON

Disable TSON optimization (`optimize=False` and `optimize_context=False`) in these cases:

| Scenario | Why |
|----------|-----|
| **Small payloads** | Overhead outweighs savings for simple objects |
| **Debugging** | JSON is more readable for troubleshooting |
| **Smaller/fine-tuned models** | May not understand TSON syntax well |
| **Native JSON mode** | If using provider's built-in structured output |
| **High-stakes extraction** | JSON has better LLM reliability |

```python
# Disable TSON for simple extractions
user = client.extract(
    response_model=User,
    messages=[...],
    optimize=False,  # Use JSON instead
)
```

**Rule of thumb:** Use TSON for large context data and arrays. Use JSON for simple single-object extractions.

---

## Works with Every Major Provider

```python
# OpenAI
client = OpenInstruct.from_provider("openai/gpt-4o")

# Anthropic
client = OpenInstruct.from_provider("anthropic/claude-3-5-sonnet")

# Google Gemini
client = OpenInstruct.from_provider("google/gemini-2.0-flash")

# Groq (fast inference)
client = OpenInstruct.from_provider("groq/llama-3.1-8b-instant")

# Ollama (local)
client = OpenInstruct.from_provider("ollama/llama3.2")

# OpenRouter (multiple providers)
client = OpenInstruct.from_provider("openrouter/openai/gpt-4o-mini")

# With explicit API key
client = OpenInstruct.from_provider("openai/gpt-4o", api_key="sk-...")
```

| Provider | Environment Variable |
|----------|---------------------|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `google` | `GOOGLE_API_KEY` |
| `groq` | `GROQ_API_KEY` |
| `together` | `TOGETHER_API_KEY` |
| `mistral` | `MISTRAL_API_KEY` |
| `ollama` | None (local) |
| `openrouter` | `OPENROUTER_API_KEY` |

---

## Features

### Automatic Retries with Backoff

Failed validations are automatically retried:

```python
from openinstruct import OpenInstruct, RetryConfig

config = RetryConfig(
    max_retries=3,
    retry_delay=0.5,      # 0.5s, 1s, 2s delays
    backoff_factor=2.0,
    on_retry=lambda attempt, error, response: print(f"Retry {attempt}"),
)

user = client.extract(
    response_model=User,
    messages=[...],
    retry_config=config,
)
```

### Token Usage Tracking

Track costs across requests:

```python
result = client.extract(
    response_model=User,
    messages=[...],
    return_usage=True,
)

print(result.data.name)              # "Alice"
print(result.usage.total_tokens)     # 175
print(result.attempts)               # 1
```

### Nested Objects

Extract complex, nested data:

```python
class Address(BaseModel):
    city: str
    country: str

class UserWithAddress(BaseModel):
    name: str
    email: str
    address: Address

user = client.extract(
    response_model=UserWithAddress,
    messages=[{"role": "user", "content": "John, john@example.com, NYC, USA"}],
)
# user.address.city = "NYC"
```

### List Extraction

Extract arrays of objects:

```python
users = client.extract(
    response_model=list[User],
    messages=[{"role": "user", "content": "List 5 random users"}],
)
# Returns list of validated User objects
```

### Input Optimization

Large context data is automatically converted to TSON:

```python
sales_data = [
    {"month": "Jan", "revenue": 50000},
    {"month": "Feb", "revenue": 62000},
    # ... 100 more rows
]

class Analysis(BaseModel):
    total_revenue: float
    best_month: str

result = client.extract(
    response_model=Analysis,
    messages=[{"role": "user", "content": "Analyze: {data}"}],
    context={"data": sales_data},  # 60% smaller in tokens
)
```

### Async Support

```python
from openinstruct import AsyncOpenInstruct

async def main():
    client = AsyncOpenInstruct.from_provider("openai/gpt-4o")
    
    user = await client.extract(
        response_model=User,
        messages=[...],
    )
    
    await client.close()
```

---

## API Reference

### `OpenInstruct.from_provider()`

```python
client = OpenInstruct.from_provider(
    provider_model: str,    # "provider/model" format
    api_key: str = None,    # Optional API key
    base_url: str = None,   # Custom endpoint
    timeout: float = 60.0,
)
```

### `client.extract()`

```python
result = client.extract(
    response_model: Type[T],       # Pydantic model or list[Model]
    messages: list[dict],          # Chat messages
    context: dict = None,          # Data to inject
    optimize: bool = True,         # Use TSON for LLM output
    optimize_context: bool = True, # Use TSON for context data
    retry_config: RetryConfig = None,
    return_usage: bool = False,
    temperature: float = 0.0,
    max_tokens: int = None,
)
```

---

## Comparison with Instructor

| Feature | OpenInstruct | Instructor |
|---------|--------------|------------|
| Token Savings | ✅ 30-70% (TSON) | ❌ JSON only |
| Input Optimization | ✅ Context as TSON | ❌ |
| Multi-Provider | ✅ 8+ providers | ✅ |
| Token Tracking | ✅ Built-in | ❌ |
| Retry with Backoff | ✅ Configurable | ✅ Basic |
| Streaming | 🚧 Coming soon | ✅ |

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE)

---

**Version:** 1.1.0

*Built for efficiency. Optimized for LLMs.*
