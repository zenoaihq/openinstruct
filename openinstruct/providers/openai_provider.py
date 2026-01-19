"""
OpenAI-compatible provider adapter.

Works with OpenAI, Groq, Together AI, Mistral, Ollama, and other
OpenAI-compatible APIs.
"""

from typing import Any, Optional
from .base import BaseProvider, ProviderConfig


class OpenAIProvider(BaseProvider):
    """
    Provider for OpenAI and OpenAI-compatible APIs.
    
    Supports: OpenAI, Groq, Together AI, Mistral, Ollama, OpenRouter
    """
    
    def build_headers(self) -> dict[str, str]:
        """Build request headers with Bearer authentication."""
        headers = {"Content-Type": "application/json"}
        
        if self.api_key and self.config.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def build_request(
        self,
        messages: list[dict],
        **kwargs
    ) -> dict[str, Any]:
        """Build OpenAI-format request payload."""
        payload = {
            "model": self.model,
            "messages": messages,
        }
        
        # Add optional parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
        if "stream" in kwargs:
            payload["stream"] = kwargs["stream"]
        
        return payload
    
    def parse_response(self, response: dict) -> str:
        """Extract content from OpenAI response."""
        return response["choices"][0]["message"]["content"]
    
    def parse_usage(self, response: dict) -> dict[str, int]:
        """Extract token usage from OpenAI response."""
        usage = response.get("usage", {})
        return {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
