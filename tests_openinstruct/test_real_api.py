"""
Real API integration test for OpenInstruct.

Run with: python tests_openinstruct/test_real_api.py
Requires: OPENAI_API_KEY environment variable
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel
from openinstruct import OpenInstruct


class User(BaseModel):
    name: str
    age: int


class Analysis(BaseModel):
    total: float
    average: float
    trend: str


def test_simple_extraction():
    """Test basic extraction with a simple model."""
    print("\n=== Test 1: Simple Extraction ===")
    
    client = OpenInstruct.from_provider("openai/gpt-4o-mini")
    
    user = client.extract(
        response_model=User,
        messages=[{"role": "user", "content": "Create a user named Alice who is 30 years old"}],
    )
    
    print(f"Result: {user}")
    print(f"Name: {user.name}, Age: {user.age}")
    assert user.name.lower() == "alice"
    assert user.age == 30
    print("✓ PASSED")
    
    client.close()


def test_list_extraction():
    """Test extracting a list of objects."""
    print("\n=== Test 2: List Extraction ===")
    
    client = OpenInstruct.from_provider("openai/gpt-4o-mini")
    
    users = client.extract(
        response_model=list[User],
        messages=[{"role": "user", "content": "Create 3 users: Alice (25), Bob (30), Carol (35)"}],
    )
    
    print(f"Result: {users}")
    assert len(users) == 3
    assert users[0].name.lower() == "alice"
    print("✓ PASSED")
    
    client.close()


def test_context_injection():
    """Test with TSON context injection."""
    print("\n=== Test 3: Context Injection ===")
    
    client = OpenInstruct.from_provider("openai/gpt-4o-mini")
    
    sales_data = [
        {"month": "Jan", "revenue": 1000},
        {"month": "Feb", "revenue": 1500},
        {"month": "Mar", "revenue": 1200},
    ]
    
    result = client.extract(
        response_model=Analysis,
        messages=[{"role": "user", "content": "Analyze this sales data: {data}"}],
        context={"data": sales_data},
    )
    
    print(f"Result: {result}")
    print(f"Total: {result.total}, Average: {result.average}, Trend: {result.trend}")
    assert result.total == 3700 or result.total == 3700.0
    print("✓ PASSED")
    
    client.close()


def test_json_fallback():
    """Test that JSON fallback works when TSON fails."""
    print("\n=== Test 4: JSON Mode ===")
    
    client = OpenInstruct.from_provider("openai/gpt-4o-mini")
    
    user = client.extract(
        response_model=User,
        messages=[{"role": "user", "content": "Create a user named Bob, age 25"}],
        optimize=False,  # Force JSON mode
    )
    
    print(f"Result: {user}")
    assert user.name.lower() == "bob"
    print("✓ PASSED")
    
    client.close()


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    print("=" * 50)
    print("OpenInstruct Real API Tests")
    print("=" * 50)
    
    try:
        test_simple_extraction()
        test_list_extraction()
        test_context_injection()
        test_json_fallback()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
