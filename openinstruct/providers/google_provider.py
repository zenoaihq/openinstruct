"""
Google Gemini provider adapter.

Uses Google's OpenAI-compatible endpoint.
"""

from typing import Any
from .base import BaseProvider, ProviderConfig


class GoogleProvider(BaseProvider):
    """
    Provider for Google Gemini models.
    
    Uses the OpenAI-compatible endpoint for simplicity.
    """
    
    def build_headers(self) -> dict[str, str]:
        """Build request headers (API key goes in query param)."""
        return {"Content-Type": "application/json"}
    
    def get_url(self) -> str:
        """Get URL with API key as query parameter."""
        base_url = f"{self.config.base_url}{self.config.chat_endpoint}"
        if self.api_key:
            return f"{base_url}?key={self.api_key}"
        return base_url
    
    def build_request(
        self,
        messages: list[dict],
        **kwargs
    ) -> dict[str, Any]:
        """Build OpenAI-compatible request for Gemini."""
        payload = {
            "model": self.model,
            "messages": messages,
        }
        
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
        
        return payload
    
    def parse_response(self, response: dict) -> str:
        """Extract content from Gemini response."""
        return response["choices"][0]["message"]["content"]
    
    def parse_usage(self, response: dict) -> dict[str, int]:
        """Extract token usage from Gemini response."""
        usage = response.get("usage", {})
        return {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
