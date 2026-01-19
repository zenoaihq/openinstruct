"""
OpenInstruct - Token-efficient structured extraction from LLMs.

Main client class with multi-provider gateway and TSON optimization.
"""

import time
import httpx
from copy import deepcopy
from typing import Any, Type, TypeVar, Optional, Union

from pydantic import BaseModel, ValidationError

from .providers import get_provider, BaseProvider, PROVIDERS
from .extraction import (
    build_system_prompt,
    parse_response,
    validate_response,
    _is_list_type,
)
from .context import inject_context
from .retry import create_retry_message, should_retry
from .types import TokenUsage, ExtractionResult, RetryConfig

T = TypeVar('T', bound=BaseModel)


class OpenInstruct:
    """
    Token-efficient structured extraction from LLMs.
    
    Uses TSON format to reduce token consumption by 30-70% while
    supporting multiple LLM providers through a unified API.
    
    Example:
        >>> from openinstruct import OpenInstruct
        >>> from pydantic import BaseModel
        >>> 
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> 
        >>> # Multi-provider gateway
        >>> client = OpenInstruct.from_provider("openai/gpt-4o")
        >>> 
        >>> user = client.extract(
        ...     response_model=User,
        ...     messages=[{"role": "user", "content": "Create user Alice, 30"}],
        ... )
        >>> print(user.name)
        Alice
    """
    
    def __init__(
        self,
        provider: BaseProvider,
        timeout: float = 60.0,
    ):
        """
        Initialize OpenInstruct with a provider.
        
        Use the from_provider() factory method for easier initialization.
        
        Args:
            provider: Configured provider adapter
            timeout: Request timeout in seconds
        """
        self.provider = provider
        self.timeout = timeout
        
        # Create HTTP client
        self._client = httpx.Client(
            headers=provider.build_headers(),
            timeout=timeout,
        )
    
    @classmethod
    def from_provider(
        cls,
        provider_model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ) -> "OpenInstruct":
        """
        Create an OpenInstruct client from a provider/model string.
        
        Args:
            provider_model: Provider and model in format "provider/model"
            api_key: Optional API key (uses environment variable if not provided)
            base_url: Optional custom base URL (for Azure, proxies, self-hosted)
            timeout: Request timeout in seconds
        
        Returns:
            Configured OpenInstruct client
        
        Examples:
            >>> client = OpenInstruct.from_provider("openai/gpt-4o")
            >>> client = OpenInstruct.from_provider("anthropic/claude-3-5-sonnet")
            >>> 
            >>> # With explicit API key
            >>> client = OpenInstruct.from_provider("openai/gpt-4o", api_key="sk-...")
            >>> 
            >>> # Custom endpoint (Azure, self-hosted, etc.)
            >>> client = OpenInstruct.from_provider(
            ...     "openai/my-deployment",
            ...     api_key="...",
            ...     base_url="https://my-resource.openai.azure.com"
            ... )
        """
        provider = get_provider(provider_model, api_key=api_key, base_url=base_url)
        return cls(provider=provider, timeout=timeout)
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all available providers."""
        return sorted(PROVIDERS.keys())
    
    def extract(
        self,
        response_model: Type[T],
        messages: list[dict],
        context: Optional[dict[str, Any]] = None,
        optimize: bool = True,
        optimize_context: bool = True,
        max_retries: Optional[int] = None,
        retry_config: Optional[Union[RetryConfig, int]] = None,
        return_usage: bool = False,
        **kwargs
    ) -> Union[T, ExtractionResult[T]]:
        """
        Extract structured data from LLM response.
        
        Args:
            response_model: Pydantic model or list[Model] for response
            messages: List of message dicts with 'role' and 'content'
            context: Optional dict of data to inject into messages
            optimize: Use TSON format for LLM output (default True)
            optimize_context: Use TSON format for context data (default True)
            max_retries: DEPRECATED - use retry_config instead. Number of retries.
            retry_config: Retry configuration (RetryConfig or int for max_retries)
            return_usage: If True, return ExtractionResult with token usage
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            Validated Pydantic model instance(s), or ExtractionResult if return_usage=True
        
        Raises:
            ValidationError: If response doesn't match schema after retries
            httpx.HTTPError: If API call fails
        
        Examples:
            >>> user = client.extract(
            ...     response_model=User,
            ...     messages=[{"role": "user", "content": "Create a user"}],
            ... )
            
            >>> # With token usage tracking
            >>> result = client.extract(
            ...     response_model=User,
            ...     messages=[...],
            ...     return_usage=True,
            ... )
            >>> print(result.data.name, result.usage.total_tokens)
            
            >>> # With retry configuration
            >>> result = client.extract(
            ...     response_model=User,
            ...     messages=[...],
            ...     retry_config=RetryConfig(max_retries=3, retry_delay=1.0),
            ... )
        """
        # Handle retry configuration
        if retry_config is None:
            config = RetryConfig(max_retries=max_retries if max_retries is not None else 2)
        elif isinstance(retry_config, int):
            config = RetryConfig(max_retries=retry_config)
        else:
            config = retry_config
        
        messages = deepcopy(messages)
        has_tson_context = False
        
        if context:
            messages = inject_context(messages, context, optimize=optimize_context)
            has_tson_context = optimize_context and any(
                isinstance(v, (list, dict)) for v in context.values()
            )
        
        # Build system prompt
        system_prompt = build_system_prompt(
            response_model=response_model,
            optimize=optimize,
            has_tson_context=has_tson_context
        )

        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        last_error = None
        raw_response = None
        total_usage = TokenUsage()
        attempts = 0
        used_fallback = False
        
        for attempt in range(config.max_retries + 1):
            attempts = attempt + 1
            try:
                raw_response, usage = self._call_llm(full_messages, **kwargs)
                if usage:
                    total_usage = total_usage + usage
                    
                parsed = parse_response(raw_response, optimize=optimize)
                result = validate_response(parsed, response_model)
                
                if return_usage:
                    return ExtractionResult(
                        data=result,
                        usage=total_usage,
                        attempts=attempts,
                        used_fallback=False,
                    )
                return result
                
            except (ValidationError, ValueError, KeyError) as e:
                last_error = e
                
                if not should_retry(e, attempt, config.max_retries):
                    if optimize and attempt >= config.max_retries:
                        break  # Try JSON fallback
                    raise
                
                # Call on_retry callback if configured
                if config.on_retry:
                    config.on_retry(attempt + 1, e, raw_response or "")
                
                # Apply delay with exponential backoff
                delay = config.calculate_delay(attempt)
                if delay > 0:
                    time.sleep(delay)
                
                retry_msg = create_retry_message(
                    error=e,
                    optimize=optimize,
                    previous_response=raw_response
                )
                full_messages.append(retry_msg)
        
        # JSON Fallback
        if optimize and last_error:
            used_fallback = True
            try:
                json_system_prompt = build_system_prompt(
                    response_model=response_model,
                    optimize=False,
                    has_tson_context=has_tson_context
                )
                
                fallback_messages = [{"role": "system", "content": json_system_prompt}]
                fallback_messages.extend(messages)
                fallback_messages.append({
                    "role": "user", 
                    "content": "Please return your response as valid JSON."
                })
                
                raw_response, usage = self._call_llm(fallback_messages, **kwargs)
                if usage:
                    total_usage = total_usage + usage
                attempts += 1
                
                parsed = parse_response(raw_response, optimize=False)
                result = validate_response(parsed, response_model)
                
                if return_usage:
                    return ExtractionResult(
                        data=result,
                        usage=total_usage,
                        attempts=attempts,
                        used_fallback=True,
                    )
                return result
                
            except Exception:
                pass
        
        if last_error:
            raise last_error
    
    def _call_llm(self, messages: list[dict], **kwargs) -> tuple[str, Optional[TokenUsage]]:
        """Make the actual LLM API call."""
        payload = self.provider.build_request(messages, **kwargs)
        url = self.provider.get_url()
        
        response = self._client.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        content = self.provider.parse_response(data)
        
        # Parse usage if available
        usage_data = self.provider.parse_usage(data)
        usage = None
        if usage_data:
            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
        
        return content, usage
    
    def close(self):
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class AsyncOpenInstruct:
    """
    Async version of OpenInstruct.
    
    Same API as OpenInstruct but with async methods.
    """
    
    def __init__(
        self,
        provider: BaseProvider,
        timeout: float = 60.0,
    ):
        self.provider = provider
        self.timeout = timeout
        
        self._client = httpx.AsyncClient(
            headers=provider.build_headers(),
            timeout=timeout,
        )
    
    @classmethod
    def from_provider(
        cls,
        provider_model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ) -> "AsyncOpenInstruct":
        """Create an async client from provider/model string."""
        provider = get_provider(provider_model, api_key=api_key, base_url=base_url)
        return cls(provider=provider, timeout=timeout)
    
    async def extract(
        self,
        response_model: Type[T],
        messages: list[dict],
        context: Optional[dict[str, Any]] = None,
        optimize: bool = True,
        optimize_context: bool = True,
        max_retries: Optional[int] = None,
        retry_config: Optional[Union[RetryConfig, int]] = None,
        return_usage: bool = False,
        **kwargs
    ) -> Union[T, ExtractionResult[T]]:
        """Async version of OpenInstruct.extract()"""
        # Handle retry configuration
        if retry_config is None:
            config = RetryConfig(max_retries=max_retries if max_retries is not None else 2)
        elif isinstance(retry_config, int):
            config = RetryConfig(max_retries=retry_config)
        else:
            config = retry_config
        
        messages = deepcopy(messages)
        has_tson_context = False
        
        if context:
            messages = inject_context(messages, context, optimize=optimize_context)
            has_tson_context = optimize_context and any(
                isinstance(v, (list, dict)) for v in context.values()
            )
        
        system_prompt = build_system_prompt(
            response_model=response_model,
            optimize=optimize,
            has_tson_context=has_tson_context
        )
        
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        last_error = None
        raw_response = None
        total_usage = TokenUsage()
        attempts = 0
        used_fallback = False
        
        for attempt in range(config.max_retries + 1):
            attempts = attempt + 1
            try:
                raw_response, usage = await self._call_llm(full_messages, **kwargs)
                if usage:
                    total_usage = total_usage + usage
                    
                parsed = parse_response(raw_response, optimize=optimize)
                result = validate_response(parsed, response_model)
                
                if return_usage:
                    return ExtractionResult(
                        data=result,
                        usage=total_usage,
                        attempts=attempts,
                        used_fallback=False,
                    )
                return result
                
            except (ValidationError, ValueError, KeyError) as e:
                last_error = e
                
                if not should_retry(e, attempt, config.max_retries):
                    if optimize and attempt >= config.max_retries:
                        break
                    raise
                
                # Call on_retry callback if configured
                if config.on_retry:
                    config.on_retry(attempt + 1, e, raw_response or "")
                
                # Apply delay with exponential backoff (using asyncio.sleep)
                delay = config.calculate_delay(attempt)
                if delay > 0:
                    import asyncio
                    await asyncio.sleep(delay)
                
                retry_msg = create_retry_message(
                    error=e,
                    optimize=optimize,
                    previous_response=raw_response
                )
                full_messages.append(retry_msg)
        
        # JSON Fallback
        if optimize and last_error:
            used_fallback = True
            try:
                json_system_prompt = build_system_prompt(
                    response_model=response_model,
                    optimize=False,
                    has_tson_context=has_tson_context
                )
                
                fallback_messages = [{"role": "system", "content": json_system_prompt}]
                fallback_messages.extend(messages)
                fallback_messages.append({
                    "role": "user", 
                    "content": "Please return your response as valid JSON."
                })
                
                raw_response, usage = await self._call_llm(fallback_messages, **kwargs)
                if usage:
                    total_usage = total_usage + usage
                attempts += 1
                
                parsed = parse_response(raw_response, optimize=False)
                result = validate_response(parsed, response_model)
                
                if return_usage:
                    return ExtractionResult(
                        data=result,
                        usage=total_usage,
                        attempts=attempts,
                        used_fallback=True,
                    )
                return result
                
            except Exception:
                pass
        
        if last_error:
            raise last_error
    
    async def _call_llm(self, messages: list[dict], **kwargs) -> tuple[str, Optional[TokenUsage]]:
        """Async LLM API call."""
        payload = self.provider.build_request(messages, **kwargs)
        url = self.provider.get_url()
        
        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        content = self.provider.parse_response(data)
        
        # Parse usage if available
        usage_data = self.provider.parse_usage(data)
        usage = None
        if usage_data:
            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
        
        return content, usage
    
    async def close(self):
        """Close the async HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()
