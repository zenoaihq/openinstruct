"""
Test JSON fallback mechanism.
"""

import unittest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from openinstruct import OpenInstruct
from openinstruct import extraction

class User(BaseModel):
    name: str
    age: int

class TestFallback(unittest.TestCase):
    def test_fallback_logic(self):
        """
        Test that client switches to JSON mode after TSON failures.
        """
        # Mock provider
        provider = MagicMock()
        provider.build_headers.return_value = {}
        provider.build_request.return_value = {}
        provider.get_url.return_value = "http://test"
        provider.parse_response.return_value = "mock_response"
        
        # Initialize client
        client = OpenInstruct(provider=provider)
        
        # Mock HTTP client
        client._client = MagicMock()
        client._client.post.return_value.json.return_value = {}
        
        # Mock call_llm to avoid actual network calls
        # We want it to be called:
        # 1. TSON attempt 1
        # 2. TSON attempt 2 (retry)
        # 3. TSON attempt 3 (retry) -> Max retries reached
        # 4. JSON Fallback
        
        client._call_llm = MagicMock(return_value="mock_llm_response")
        
        # Patch parse_response to fail when optimize=True (TSON) and succeed when optimize=False (JSON)
        original_parse = extraction.parse_response
        
        def side_effect(text, optimize=True):
            if optimize:
                raise ValueError("Simulated TSON parse error")
            return {"name": "Fallback User", "age": 99}
            
        with patch('openinstruct.client.parse_response', side_effect=side_effect) as mock_parse:
            # We also need validate_response to work
            # It will receive the dict {"name": "Fallback User", "age": 99}
            
            user = client.extract(
                response_model=User,
                messages=[{"role": "user", "content": "test"}],
                optimize=True,
                max_retries=2
            )
            
            # Verify result came from JSON fallback
            self.assertEqual(user.name, "Fallback User")
            self.assertEqual(user.age, 99)
            
            # Verify calls
            # expected calls to parse_response:
            # 1. optimize=True (Attempt 0)
            # 2. optimize=True (Retry 1)
            # 3. optimize=True (Retry 2)
            # 4. optimize=False (Fallback)
            self.assertEqual(mock_parse.call_count, 4)
            
            # Verify the last call was with optimize=False
            args, kwargs = mock_parse.call_args
            self.assertFalse(kwargs.get('optimize'))
            
            print("\n✓ Fallback test passed!")

if __name__ == "__main__":
    unittest.main()
