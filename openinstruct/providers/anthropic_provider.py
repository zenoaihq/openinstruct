"""
Anthropic provider adapter.

Handles Anthropic's Messages API format.
"""

from typing import Any
from .base import BaseProvider, ProviderConfig


class AnthropicProvider(BaseProvider):
    """
    Provider for Anthropic's Claude models.
    
    Converts between OpenAI-style messages and Anthropic's Messages API format.
    """
    
    def build_headers(self) -> dict[str, str]:
        """Build request headers with x-api-key authentication."""
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        return headers
    
    def build_request(
        self,
        messages: list[dict],
        **kwargs
    ) -> dict[str, Any]:
        """
        Build Anthropic Messages API request payload.
        
        Converts OpenAI-style messages to Anthropic format.
        """
        # Separate system message from other messages
        system_content = None
        anthropic_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_content = content
            else:
                # Anthropic uses "user" and "assistant" roles
                anthropic_role = "assistant" if role == "assistant" else "user"
                anthropic_messages.append({
                    "role": anthropic_role,
                    "content": content
                })
        
        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        
        if system_content:
            payload["system"] = system_content
        
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        
        return payload
    
    def parse_response(self, response: dict) -> str:
        """Extract content from Anthropic response."""
        content = response.get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", "")
        return ""
    
    def parse_usage(self, response: dict) -> dict[str, int]:
        """Extract token usage from Anthropic response."""
        usage = response.get("usage", {})
        return {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        }
