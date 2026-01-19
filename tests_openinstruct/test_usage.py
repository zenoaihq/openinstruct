"""
Tests for new OpenInstruct features: TokenUsage, RetryConfig, ExtractionResult.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel

from openinstruct import OpenInstruct, TokenUsage, ExtractionResult, RetryConfig


class User(BaseModel):
    name: str
    age: int


class TestTokenUsage(unittest.TestCase):
    """Test TokenUsage dataclass."""
    
    def test_token_usage_addition(self):
        """Test that TokenUsage objects can be added together."""
        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage2 = TokenUsage(prompt_tokens=80, completion_tokens=30, total_tokens=110)
        
        combined = usage1 + usage2
        
        self.assertEqual(combined.prompt_tokens, 180)
        self.assertEqual(combined.completion_tokens, 80)
        self.assertEqual(combined.total_tokens, 260)
    
    def test_token_usage_defaults(self):
        """Test TokenUsage default values."""
        usage = TokenUsage()
        
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.completion_tokens, 0)
        self.assertEqual(usage.total_tokens, 0)


class TestRetryConfig(unittest.TestCase):
    """Test RetryConfig dataclass."""
    
    def test_default_values(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        self.assertEqual(config.max_retries, 2)
        self.assertEqual(config.retry_delay, 0.0)
        self.assertEqual(config.backoff_factor, 2.0)
        self.assertIsNone(config.on_retry)
    
    def test_calculate_delay_no_base(self):
        """Test delay calculation with no base delay."""
        config = RetryConfig(retry_delay=0.0)
        
        self.assertEqual(config.calculate_delay(0), 0.0)
        self.assertEqual(config.calculate_delay(1), 0.0)
        self.assertEqual(config.calculate_delay(2), 0.0)
    
    def test_calculate_delay_exponential(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(retry_delay=1.0, backoff_factor=2.0)
        
        # 1.0 * 2^0 = 1.0
        self.assertEqual(config.calculate_delay(0), 1.0)
        # 1.0 * 2^1 = 2.0
        self.assertEqual(config.calculate_delay(1), 2.0)
        # 1.0 * 2^2 = 4.0
        self.assertEqual(config.calculate_delay(2), 4.0)
    
    def test_custom_backoff_factor(self):
        """Test custom backoff factor."""
        config = RetryConfig(retry_delay=0.5, backoff_factor=3.0)
        
        # 0.5 * 3^0 = 0.5
        self.assertEqual(config.calculate_delay(0), 0.5)
        # 0.5 * 3^1 = 1.5
        self.assertEqual(config.calculate_delay(1), 1.5)
        # 0.5 * 3^2 = 4.5
        self.assertEqual(config.calculate_delay(2), 4.5)


class TestExtractionResult(unittest.TestCase):
    """Test ExtractionResult dataclass."""
    
    def test_extraction_result_fields(self):
        """Test ExtractionResult stores all fields correctly."""
        user = User(name="Alice", age=30)
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        
        result = ExtractionResult(
            data=user,
            usage=usage,
            attempts=2,
            used_fallback=True
        )
        
        self.assertEqual(result.data.name, "Alice")
        self.assertEqual(result.data.age, 30)
        self.assertEqual(result.usage.total_tokens, 150)
        self.assertEqual(result.attempts, 2)
        self.assertTrue(result.used_fallback)
    
    def test_extraction_result_defaults(self):
        """Test ExtractionResult default values."""
        user = User(name="Bob", age=25)
        result = ExtractionResult(data=user)
        
        self.assertIsNone(result.usage)
        self.assertEqual(result.attempts, 1)
        self.assertFalse(result.used_fallback)


class TestClientReturnUsage(unittest.TestCase):
    """Test client.extract() with return_usage=True."""
    
    def test_return_usage_returns_extraction_result(self):
        """Test that return_usage=True returns ExtractionResult."""
        # Mock provider
        provider = MagicMock()
        provider.build_headers.return_value = {}
        provider.build_request.return_value = {}
        provider.get_url.return_value = "http://test"
        provider.parse_response.return_value = '{@name,age|Alice,30}'
        provider.parse_usage.return_value = {
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "total_tokens": 125
        }
        
        # Initialize client
        client = OpenInstruct(provider=provider)
        
        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        client._client = MagicMock()
        client._client.post.return_value = mock_response
        
        # Extract with return_usage=True
        result = client.extract(
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            return_usage=True,
        )
        
        # Verify returns ExtractionResult
        self.assertIsInstance(result, ExtractionResult)
        self.assertEqual(result.data.name, "Alice")
        self.assertEqual(result.data.age, 30)
        self.assertEqual(result.usage.prompt_tokens, 100)
        self.assertEqual(result.usage.completion_tokens, 25)
        self.assertEqual(result.usage.total_tokens, 125)
        self.assertEqual(result.attempts, 1)
        self.assertFalse(result.used_fallback)
        
        print("\n✓ return_usage test passed!")


class TestRetryConfigIntegration(unittest.TestCase):
    """Test RetryConfig integration with client."""
    
    def test_retry_config_int_shorthand(self):
        """Test that integer can be passed as retry_config."""
        # Mock provider
        provider = MagicMock()
        provider.build_headers.return_value = {}
        provider.build_request.return_value = {}
        provider.get_url.return_value = "http://test"
        provider.parse_response.return_value = '{@name,age|Bob,25}'
        provider.parse_usage.return_value = None
        
        client = OpenInstruct(provider=provider)
        
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        client._client = MagicMock()
        client._client.post.return_value = mock_response
        
        # Should not raise - int shorthand for max_retries
        result = client.extract(
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            retry_config=5,  # Integer shorthand
        )
        
        self.assertEqual(result.name, "Bob")
        print("\n✓ retry_config int shorthand test passed!")
    
    def test_on_retry_callback(self):
        """Test that on_retry callback is called on failures."""
        retry_calls = []
        
        def on_retry_callback(attempt, error, response):
            retry_calls.append((attempt, str(error), response))
        
        # Mock provider
        provider = MagicMock()
        provider.build_headers.return_value = {}
        provider.build_request.return_value = {}
        provider.get_url.return_value = "http://test"
        provider.parse_usage.return_value = None
        
        # First call fails, second succeeds
        call_count = [0]
        def mock_parse_response(data):
            call_count[0] += 1
            if call_count[0] == 1:
                return 'invalid tson'
            return '{@name,age|Carol,35}'
        
        provider.parse_response.side_effect = mock_parse_response
        
        client = OpenInstruct(provider=provider)
        
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        client._client = MagicMock()
        client._client.post.return_value = mock_response
        
        config = RetryConfig(
            max_retries=2,
            on_retry=on_retry_callback
        )
        
        result = client.extract(
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            retry_config=config,
        )
        
        self.assertEqual(result.name, "Carol")
        self.assertEqual(len(retry_calls), 1)
        self.assertEqual(retry_calls[0][0], 1)  # First retry
        print("\n✓ on_retry callback test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("OpenInstruct New Features Test Suite")
    print("=" * 60)
    unittest.main(verbosity=2)
