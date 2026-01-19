"""
Tests for OpenInstruct providers module.
"""

import pytest
import os
from openinstruct.providers import (
    PROVIDERS,
    get_provider,
    ProviderConfig,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
)


class TestProviderRegistry:
    """Tests for the provider registry."""
    
    def test_providers_exist(self):
        """Test that expected providers are registered."""
        expected = ["openai", "anthropic", "google", "groq", "together", "mistral", "ollama", "openrouter"]
        for provider in expected:
            assert provider in PROVIDERS, f"Missing provider: {provider}"
    
    def test_provider_config_has_required_fields(self):
        """Test that all providers have required configuration."""
        for name, config in PROVIDERS.items():
            assert isinstance(config, ProviderConfig)
            assert config.name
            assert config.base_url
            assert config.auth_type in ["bearer", "x-api-key", "query_param", "none"]
    
    def test_openai_config(self):
        """Test OpenAI provider configuration."""
        config = PROVIDERS["openai"]
        assert config.base_url == "https://api.openai.com/v1"
        assert config.env_key == "OPENAI_API_KEY"
        assert config.auth_type == "bearer"
    
    def test_anthropic_config(self):
        """Test Anthropic provider configuration."""
        config = PROVIDERS["anthropic"]
        assert config.base_url == "https://api.anthropic.com/v1"
        assert config.env_key == "ANTHROPIC_API_KEY"
        assert config.auth_type == "x-api-key"


class TestGetProvider:
    """Tests for the get_provider function."""
    
    def test_invalid_format_raises_error(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid format"):
            get_provider("invalid-format")
    
    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown/some-model")
    
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises error for providers that need it."""
        # Temporarily unset the environment variable
        original = os.environ.get("OPENAI_API_KEY")
        if original:
            del os.environ["OPENAI_API_KEY"]
        
        try:
            with pytest.raises(ValueError, match="API key required"):
                get_provider("openai/gpt-4o")
        finally:
            if original:
                os.environ["OPENAI_API_KEY"] = original
    
    def test_explicit_api_key_works(self):
        """Test that explicit API key works."""
        provider = get_provider("openai/gpt-4o", api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "gpt-4o"
    
    def test_ollama_no_api_key_required(self):
        """Test that Ollama doesn't require an API key."""
        provider = get_provider("ollama/llama3.2")
        assert provider.api_key is None
        assert provider.model == "llama3.2"


class TestOpenAIProvider:
    """Tests for OpenAI provider adapter."""
    
    def test_build_headers(self):
        """Test header building."""
        provider = get_provider("openai/gpt-4o", api_key="test-key")
        headers = provider.build_headers()
        
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-key"
    
    def test_build_request(self):
        """Test request payload building."""
        provider = get_provider("openai/gpt-4o", api_key="test-key")
        
        messages = [{"role": "user", "content": "Hello"}]
        payload = provider.build_request(messages, temperature=0.7)
        
        assert payload["model"] == "gpt-4o"
        assert payload["messages"] == messages
        assert payload["temperature"] == 0.7
    
    def test_parse_response(self):
        """Test response parsing."""
        provider = get_provider("openai/gpt-4o", api_key="test-key")
        
        response = {
            "choices": [
                {"message": {"content": "Hello, world!"}}
            ]
        }
        
        content = provider.parse_response(response)
        assert content == "Hello, world!"
    
    def test_parse_usage(self):
        """Test usage parsing."""
        provider = get_provider("openai/gpt-4o", api_key="test-key")
        
        response = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        
        usage = provider.parse_usage(response)
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 20
        assert usage["total_tokens"] == 30


class TestAnthropicProvider:
    """Tests for Anthropic provider adapter."""
    
    def test_build_headers(self):
        """Test header building with x-api-key."""
        provider = get_provider("anthropic/claude-3-5-sonnet", api_key="test-key")
        headers = provider.build_headers()
        
        assert headers["x-api-key"] == "test-key"
        assert "anthropic-version" in headers
    
    def test_build_request_converts_messages(self):
        """Test that system message is separated."""
        provider = get_provider("anthropic/claude-3-5-sonnet", api_key="test-key")
        
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        
        payload = provider.build_request(messages)
        
        assert payload["system"] == "You are helpful."
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
    
    def test_parse_response(self):
        """Test Anthropic response parsing."""
        provider = get_provider("anthropic/claude-3-5-sonnet", api_key="test-key")
        
        response = {
            "content": [
                {"type": "text", "text": "Hello!"}
            ]
        }
        
        content = provider.parse_response(response)
        assert content == "Hello!"


class TestGoogleProvider:
    """Tests for Google provider adapter."""
    
    def test_get_url_includes_api_key(self):
        """Test that URL includes API key as query param."""
        provider = get_provider("google/gemini-2.0-flash", api_key="test-key")
        url = provider.get_url()
        
        assert "?key=test-key" in url
