"""
OpenInstruct - Token-efficient Structured Extraction for LLMs

A library that combines:
• Multi-provider gateway (OpenAI, Anthropic, Google, Groq, etc.)
• 30-70% token savings via TSON optimization
• Pydantic validation with automatic retries
• Token usage tracking and retry configuration

Basic usage:
    >>> from openinstruct import OpenInstruct
    >>> from pydantic import BaseModel
    >>>
    >>> class User(BaseModel):
    ...     name: str
    ...     age: int
    >>>
    >>> client = OpenInstruct.from_provider("openai/gpt-4o")
    >>> user = client.extract(
    ...     response_model=User,
    ...     messages=[{"role": "user", "content": "Create user Alice, 30"}],
    ... )
    >>> print(user.name)
    Alice
"""

__version__ = "1.1.0"
__author__ = "Zeno AI"
__license__ = "MIT"

from .client import OpenInstruct, AsyncOpenInstruct
from .providers import PROVIDERS, get_provider
from .types import TokenUsage, ExtractionResult, RetryConfig

__all__ = [
    "OpenInstruct",
    "AsyncOpenInstruct",
    "PROVIDERS",
    "get_provider",
    "TokenUsage",
    "ExtractionResult",
    "RetryConfig",
]
