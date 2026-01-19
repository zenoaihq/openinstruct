"""
Type definitions for OpenInstruct.

Includes dataclasses for token usage tracking and extraction results.
"""

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable, Any

from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


@dataclass
class TokenUsage:
    """
    Token usage statistics from an LLM API call.
    
    Attributes:
        prompt_tokens: Number of tokens in the input/prompt
        completion_tokens: Number of tokens in the output/completion
        total_tokens: Total tokens used (prompt + completion)
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage objects together (for accumulating across retries)."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class ExtractionResult(Generic[T]):
    """
    Result from an extraction operation including metadata.
    
    Returned when `return_usage=True` is passed to `extract()`.
    
    Attributes:
        data: The extracted and validated Pydantic model instance(s)
        usage: Token usage statistics (accumulated across retries)
        attempts: Number of attempts made (1 = success on first try)
        used_fallback: Whether JSON fallback was used after TSON failures
    
    Example:
        >>> result = client.extract(
        ...     response_model=User,
        ...     messages=[...],
        ...     return_usage=True,
        ... )
        >>> print(result.data.name)
        Alice
        >>> print(result.usage.total_tokens)
        150
    """
    data: T
    usage: Optional[TokenUsage] = None
    attempts: int = 1
    used_fallback: bool = False


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior on extraction failures.
    
    Attributes:
        max_retries: Maximum number of retry attempts (default: 2)
        retry_delay: Base delay in seconds between retries (default: 0.0)
        backoff_factor: Multiplier for exponential backoff (default: 2.0)
            - Delay = retry_delay * (backoff_factor ** attempt)
            - Example with delay=1.0, factor=2.0: 1s, 2s, 4s, ...
        on_retry: Optional callback called on each retry attempt
            - Signature: (attempt: int, error: Exception, response: str) -> None
    
    Example:
        >>> # Simple: just max retries
        >>> config = RetryConfig(max_retries=3)
        
        >>> # With exponential backoff
        >>> config = RetryConfig(
        ...     max_retries=3,
        ...     retry_delay=0.5,
        ...     backoff_factor=2.0,  # 0.5s, 1s, 2s delays
        ... )
        
        >>> # With logging callback
        >>> def log_retry(attempt, error, response):
        ...     print(f"Retry {attempt}: {error}")
        >>> config = RetryConfig(max_retries=2, on_retry=log_retry)
        
        >>> # Use in extract()
        >>> result = client.extract(
        ...     response_model=User,
        ...     messages=[...],
        ...     retry_config=config,
        ... )
    """
    max_retries: int = 2
    retry_delay: float = 0.0
    backoff_factor: float = 2.0
    on_retry: Optional[Callable[[int, Exception, str], None]] = None
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given retry attempt.
        
        Args:
            attempt: Current attempt number (0-indexed)
        
        Returns:
            Delay in seconds before the next retry
        """
        if self.retry_delay <= 0:
            return 0.0
        return self.retry_delay * (self.backoff_factor ** attempt)
