"""
Tests for OpenInstruct client.
"""

import pytest
from pydantic import BaseModel
from openinstruct import OpenInstruct, AsyncOpenInstruct


class User(BaseModel):
    name: str
    age: int


class Address(BaseModel):
    city: str
    zip: str


class UserWithAddress(BaseModel):
    name: str
    address: Address


class TestOpenInstructInit:
    """Tests for OpenInstruct initialization."""
    
    def test_from_provider_creates_client(self):
        """Test that from_provider creates a valid client."""
        client = OpenInstruct.from_provider("openai/gpt-4o", api_key="test-key")
        assert client is not None
        assert client.provider is not None
        assert client.provider.model == "gpt-4o"
    
    def test_from_provider_with_different_providers(self):
        """Test from_provider with various providers."""
        providers = [
            ("openai/gpt-4o", "test-openai"),
            ("anthropic/claude-3-5-sonnet", "test-anthropic"),
            ("google/gemini-2.0-flash", "test-google"),
            ("groq/llama-3.1-8b-instant", "test-groq"),
        ]
        
        for provider_model, api_key in providers:
            client = OpenInstruct.from_provider(provider_model, api_key=api_key)
            provider_name = provider_model.split("/")[0]
            model_name = provider_model.split("/")[1]
            
            assert client.provider.model == model_name
    
    def test_list_providers(self):
        """Test listing available providers."""
        providers = OpenInstruct.list_providers()
        
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
        assert "groq" in providers
    
    def test_context_manager(self):
        """Test that client works as context manager."""
        with OpenInstruct.from_provider("openai/gpt-4o", api_key="test") as client:
            assert client is not None


class TestAsyncOpenInstruct:
    """Tests for AsyncOpenInstruct client."""
    
    def test_from_provider_creates_async_client(self):
        """Test that from_provider creates a valid async client."""
        client = AsyncOpenInstruct.from_provider("openai/gpt-4o", api_key="test-key")
        assert client is not None
        assert client.provider is not None
