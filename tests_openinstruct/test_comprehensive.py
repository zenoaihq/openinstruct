"""
Comprehensive OpenInstruct Test Script

Tests all major features with real API calls via OpenRouter.

Usage:
    1. Set OPENROUTER_API_KEY environment variable
    2. Run: python test_comprehensive.py

Tests cover:
    - Basic extraction
    - List extraction  
    - Context injection (TSON input optimization)
    - JSON mode (optimize=False)
    - Nested models
    - Error handling (invalid API key, invalid provider)
    - Different providers
"""

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from openinstruct import OpenInstruct, PROVIDERS


# ============================================================================
# Test Models
# ============================================================================

class User(BaseModel):
    """Simple user model."""
    name: str
    age: int


class Address(BaseModel):
    """Address model for nesting tests."""
    city: str
    country: str


class UserWithAddress(BaseModel):
    """User with nested address."""
    name: str
    email: str
    address: Address


class SalesAnalysis(BaseModel):
    """Model for context injection test."""
    total_revenue: float
    best_month: str
    trend: str = Field(description="up, down, or stable")


class Sentiment(BaseModel):
    """Sentiment analysis result."""
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(ge=0, le=1)
    keywords: list[str]


# ============================================================================
# Test Functions
# ============================================================================

def test_list_providers():
    """Test 1: List available providers."""
    print("\n" + "="*60)
    print("TEST 1: List Available Providers")
    print("="*60)
    
    providers = OpenInstruct.list_providers()
    print(f"Available providers ({len(providers)}):")
    for p in providers:
        config = PROVIDERS[p]
        print(f"  • {p}: {config.name} ({config.base_url})")
    
    assert len(providers) >= 8
    assert "openai" in providers
    assert "openrouter" in providers
    print("✓ PASSED")


def test_invalid_provider():
    """Test 2: Error handling for invalid provider."""
    print("\n" + "="*60)
    print("TEST 2: Invalid Provider Error Handling")
    print("="*60)
    
    try:
        client = OpenInstruct.from_provider("invalid/model", api_key="test")
        print("✗ FAILED - Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"Got expected error: {e}")
        assert "Unknown provider" in str(e)
        print("✓ PASSED")
        return True


def test_invalid_format():
    """Test 3: Error handling for invalid format."""
    print("\n" + "="*60)
    print("TEST 3: Invalid Format Error Handling")
    print("="*60)
    
    try:
        client = OpenInstruct.from_provider("no-slash-here", api_key="test")
        print("✗ FAILED - Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"Got expected error: {e}")
        assert "Invalid format" in str(e)
        print("✓ PASSED")
        return True


def test_missing_api_key():
    """Test 4: Error handling for missing API key."""
    print("\n" + "="*60)
    print("TEST 4: Missing API Key Error Handling")
    print("="*60)
    
    # Temporarily remove API key
    original = os.environ.get("OPENAI_API_KEY")
    if original:
        del os.environ["OPENAI_API_KEY"]
    
    try:
        client = OpenInstruct.from_provider("openai/gpt-4o")
        print("✗ FAILED - Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"Got expected error: {e}")
        assert "API key required" in str(e)
        print("✓ PASSED")
        return True
    finally:
        if original:
            os.environ["OPENAI_API_KEY"] = original


def test_custom_base_url():
    """Test 5: Custom base URL override."""
    print("\n" + "="*60)
    print("TEST 5: Custom Base URL")
    print("="*60)
    
    client = OpenInstruct.from_provider(
        "openai/my-model",
        api_key="test-key",
        base_url="https://custom.api.example.com/v1"
    )
    
    url = client.provider.get_url()
    print(f"Generated URL: {url}")
    
    assert "custom.api.example.com" in url
    assert "/chat/completions" in url
    print("✓ PASSED")


def test_simple_extraction(api_key: str):
    """Test 6: Simple extraction with TSON optimization."""
    print("\n" + "="*60)
    print("TEST 6: Simple Extraction (TSON mode)")
    print("="*60)
    
    client = OpenInstruct.from_provider("openrouter/openai/gpt-4o-mini", api_key=api_key)
    
    user = client.extract(
        response_model=User,
        messages=[{"role": "user", "content": "Create a user named Alice who is 30 years old"}],
        optimize=True,  # TSON mode (default)
    )
    
    print(f"Result: {user}")
    print(f"  name: {user.name}")
    print(f"  age: {user.age}")
    
    assert user.name.lower() == "alice"
    assert user.age == 30
    print("✓ PASSED")
    
    client.close()
    return user


def test_json_mode(api_key: str):
    """Test 7: Extraction with JSON mode (no TSON)."""
    print("\n" + "="*60)
    print("TEST 7: JSON Mode (optimize=False)")
    print("="*60)
    
    client = OpenInstruct.from_provider("openrouter/openai/gpt-4o-mini", api_key=api_key)
    
    user = client.extract(
        response_model=User,
        messages=[{"role": "user", "content": "Create a user named Bob who is 25 years old"}],
        optimize=False,  # JSON mode
    )
    
    print(f"Result: {user}")
    assert user.name.lower() == "bob"
    assert user.age == 25
    print("✓ PASSED")
    
    client.close()


def test_list_extraction(api_key: str):
    """Test 8: Extract a list of objects."""
    print("\n" + "="*60)
    print("TEST 8: List Extraction")
    print("="*60)
    
    client = OpenInstruct.from_provider("openrouter/openai/gpt-4o-mini", api_key=api_key)
    
    users = client.extract(
        response_model=list[User],
        messages=[{"role": "user", "content": "Create exactly 3 users: Alice (25), Bob (30), Carol (35)"}],
    )
    
    print(f"Result: {users}")
    print(f"Count: {len(users)}")
    for u in users:
        print(f"  • {u.name}, age {u.age}")
    
    assert len(users) == 3
    print("✓ PASSED")
    
    client.close()


def test_nested_model(api_key: str):
    """Test 9: Extract model with nested objects."""
    print("\n" + "="*60)
    print("TEST 9: Nested Model Extraction")
    print("="*60)
    
    client = OpenInstruct.from_provider("openrouter/openai/gpt-4o-mini", api_key=api_key)
    
    user = client.extract(
        response_model=UserWithAddress,
        messages=[{"role": "user", "content": "Create user John Doe, email john@example.com, with address as an object with city New York and country USA"}],
    )
    
    print(f"Result: {user}")
    print(f"  name: {user.name}")
    print(f"  email: {user.email}")
    print(f"  address.city: {user.address.city}")
    print(f"  address.country: {user.address.country}")
    
    assert "john" in user.name.lower()
    assert user.address.city.lower() in ["new york", "nyc"]
    print("✓ PASSED")
    
    client.close()


def test_context_injection(api_key: str):
    """Test 10: Context injection with TSON optimization."""
    print("\n" + "="*60)
    print("TEST 10: Context Injection (Input Optimization)")
    print("="*60)
    
    client = OpenInstruct.from_provider("openrouter/openai/gpt-4o-mini", api_key=api_key)
    
    # Sales data to analyze
    sales_data = [
        {"month": "January", "revenue": 10000},
        {"month": "February", "revenue": 15000},
        {"month": "March", "revenue": 12000},
    ]
    
    print(f"Input data: {sales_data}")
    
    result = client.extract(
        response_model=SalesAnalysis,
        messages=[{"role": "user", "content": "Analyze this sales data: {data}"}],
        context={"data": sales_data},  # Auto-converted to TSON
    )
    
    print(f"Result: {result}")
    print(f"  total_revenue: {result.total_revenue}")
    print(f"  best_month: {result.best_month}")
    print(f"  trend: {result.trend}")
    
    assert result.total_revenue == 37000 or result.total_revenue == 37000.0
    assert "feb" in result.best_month.lower()
    print("✓ PASSED")
    
    client.close()


def test_sentiment_analysis(api_key: str):
    """Test 11: Complex extraction with validation."""
    print("\n" + "="*60)
    print("TEST 11: Sentiment Analysis")
    print("="*60)
    
    client = OpenInstruct.from_provider("openrouter/openai/gpt-4o-mini", api_key=api_key)
    
    result = client.extract(
        response_model=Sentiment,
        messages=[{"role": "user", "content": "Analyze sentiment: 'I absolutely love this product! It's amazing and works perfectly.'"}],
    )
    
    print(f"Result: {result}")
    print(f"  sentiment: {result.sentiment}")
    print(f"  confidence: {result.confidence}")
    print(f"  keywords: {result.keywords}")
    
    assert result.sentiment.lower() == "positive"
    assert 0 <= result.confidence <= 1
    print("✓ PASSED")
    
    client.close()


def test_context_manager(api_key: str):
    """Test 12: Context manager usage."""
    print("\n" + "="*60)
    print("TEST 12: Context Manager")
    print("="*60)
    
    with OpenInstruct.from_provider("openrouter/openai/gpt-4o-mini", api_key=api_key) as client:
        user = client.extract(
            response_model=User,
            messages=[{"role": "user", "content": "Create user Charlie, age 40"}],
        )
        print(f"Result: {user}")
        assert user.name.lower() == "charlie"
    
    print("Client auto-closed after context")
    print("✓ PASSED")


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*60)
    print("OpenInstruct Comprehensive Test Suite")
    print("="*60)
    
    # Get API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    
    passed = 0
    failed = 0
    skipped = 0
    
    # Unit tests (no API key needed)
    print("\n" + "="*60)
    print("UNIT TESTS (No API key required)")
    print("="*60)
    
    tests_no_api = [
        test_list_providers,
        test_invalid_provider,
        test_invalid_format,
        test_missing_api_key,
        test_custom_base_url,
    ]
    
    for test in tests_no_api:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    # Integration tests (API key required)
    print("\n" + "="*60)
    print("INTEGRATION TESTS (API key required)")
    print("="*60)
    
    if not api_key:
        print("\n⚠ OPENROUTER_API_KEY not set - skipping integration tests")
        print("Set it with: set OPENROUTER_API_KEY=your-key-here")
        skipped = 7
    else:
        tests_with_api = [
            test_simple_extraction,
            test_json_mode,
            test_list_extraction,
            test_nested_model,
            test_context_injection,
            test_sentiment_analysis,
            test_context_manager,
        ]
        
        for test in tests_with_api:
            try:
                test(api_key)
                passed += 1
            except Exception as e:
                print(f"✗ FAILED: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"  ✓ Passed:  {passed}")
    print(f"  ✗ Failed:  {failed}")
    print(f"  ⊘ Skipped: {skipped}")
    print("="*60)
    
    if failed == 0:
        print("ALL TESTS PASSED! ✓")
    else:
        print(f"SOME TESTS FAILED ({failed})")
        sys.exit(1)


if __name__ == "__main__":
    main()
