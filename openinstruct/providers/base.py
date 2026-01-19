"""
Provider base class and registry.

Defines the interface for LLM providers and maintains the registry
of supported providers with their configurations.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional, AsyncIterator
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    base_url: str
    env_key: Optional[str]
    auth_type: str  # "bearer", "x-api-key", "query_param", "none"
    chat_endpoint: str = "/chat/completions"
    supports_streaming: bool = True
    

# Provider Registry
PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        env_key="OPENAI_API_KEY",
        auth_type="bearer",
    ),
    "anthropic": ProviderConfig(
        name="Anthropic",
        base_url="https://api.anthropic.com/v1",
        env_key="ANTHROPIC_API_KEY",
        auth_type="x-api-key",
        chat_endpoint="/messages",
    ),
    "google": ProviderConfig(
        name="Google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        env_key="GOOGLE_API_KEY",
        auth_type="query_param",
        chat_endpoint="/openai/chat/completions",
    ),
    "groq": ProviderConfig(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        env_key="GROQ_API_KEY",
        auth_type="bearer",
    ),
    "together": ProviderConfig(
        name="Together AI",
        base_url="https://api.together.xyz/v1",
        env_key="TOGETHER_API_KEY",
        auth_type="bearer",
    ),
    "mistral": ProviderConfig(
        name="Mistral AI",
        base_url="https://api.mistral.ai/v1",
        env_key="MISTRAL_API_KEY",
        auth_type="bearer",
    ),
    "ollama": ProviderConfig(
        name="Ollama",
        base_url="http://localhost:11434/v1",
        env_key=None,
        auth_type="none",
    ),
    "openrouter": ProviderConfig(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        auth_type="bearer",
    ),
}


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Each provider implements this interface to handle API-specific
    request/response formats.
    """
    
    def __init__(
        self,
        config: ProviderConfig,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0
    ):
        self.config = config
        self.model = model
        self.timeout = timeout
        
        # Allow base_url override (for Azure, custom proxies, etc.)
        self._base_url_override = base_url
        
        # Get API key from parameter or environment
        if api_key:
            self.api_key = api_key
        elif config.env_key:
            self.api_key = os.environ.get(config.env_key)
            if not self.api_key:
                raise ValueError(
                    f"API key required. Set {config.env_key} environment variable "
                    f"or pass api_key parameter."
                )
        else:
            self.api_key = None  # Some providers (like Ollama) don't need keys
    
    @abstractmethod
    def build_headers(self) -> dict[str, str]:
        """Build request headers including authentication."""
        pass
    
    @abstractmethod
    def build_request(
        self,
        messages: list[dict],
        **kwargs
    ) -> dict[str, Any]:
        """Build the API request payload."""
        pass
    
    @abstractmethod
    def parse_response(self, response: dict) -> str:
        """Extract content from API response."""
        pass
    
    @abstractmethod
    def parse_usage(self, response: dict) -> dict[str, int]:
        """Extract token usage from API response."""
        pass
    
    def get_url(self) -> str:
        """Get the full API endpoint URL."""
        base = self._base_url_override or self.config.base_url
        return f"{base}{self.config.chat_endpoint}"


def get_provider(
    provider_model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> BaseProvider:
    """
    Get a provider instance from a provider/model string.
    
    Args:
        provider_model: String in format "provider/model" (e.g., "openai/gpt-4o")
        api_key: Optional API key (uses environment variable if not provided)
        base_url: Optional custom base URL (for Azure, proxies, self-hosted, etc.)
    
    Returns:
        Configured provider instance
    
    Examples:
        >>> provider = get_provider("openai/gpt-4o")
        >>> provider = get_provider("anthropic/claude-3-5-sonnet", api_key="sk-...")
        >>> # Custom endpoint (Azure, self-hosted, etc.)
        >>> provider = get_provider(
        ...     "openai/my-deployment",
        ...     api_key="...",
        ...     base_url="https://my-resource.openai.azure.com/openai/deployments/my-deployment"
        ... )
    """
    if "/" not in provider_model:
        raise ValueError(
            f"Invalid format: '{provider_model}'. Expected 'provider/model' "
            f"(e.g., 'openai/gpt-4o')"
        )
    
    provider_name, model = provider_model.split("/", 1)
    provider_name = provider_name.lower()
    
    if provider_name not in PROVIDERS:
        available = ", ".join(sorted(PROVIDERS.keys()))
        raise ValueError(
            f"Unknown provider: '{provider_name}'. "
            f"Available providers: {available}"
        )
    
    config = PROVIDERS[provider_name]
    
    # Import the specific provider adapter
    from . import openai_provider, anthropic_provider, google_provider
    
    adapter_map = {
        "openai": openai_provider.OpenAIProvider,
        "groq": openai_provider.OpenAIProvider,  # Groq uses OpenAI-compatible API
        "together": openai_provider.OpenAIProvider,
        "mistral": openai_provider.OpenAIProvider,
        "ollama": openai_provider.OpenAIProvider,
        "openrouter": openai_provider.OpenAIProvider,
        "anthropic": anthropic_provider.AnthropicProvider,
        "google": google_provider.GoogleProvider,
    }
    
    provider_class = adapter_map.get(provider_name)
    if not provider_class:
        raise ValueError(f"No adapter implemented for provider: {provider_name}")
    
    return provider_class(config=config, model=model, api_key=api_key, base_url=base_url)
