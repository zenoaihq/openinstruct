"""
Retry logic for validation failures.

Formats Pydantic validation errors and creates retry messages.
"""

from typing import Any
from pydantic import ValidationError


def create_retry_message(
    error: Exception = None,
    optimize: bool = True,
    previous_response: str = None
) -> dict:
    """
    Create a retry message from an error.
    
    Args:
        error: The exception (ValidationError or other)
        optimize: Whether TSON format is being used
        previous_response: The previous LLM response that failed
    
    Returns:
        Message dict with 'role' and 'content'
    """
    format_name = "TSON" if optimize else "JSON"
    
    # Format error details based on error type
    if error is None:
        error_details = "Parse error: Could not parse the response format."
    elif hasattr(error, 'errors'):
        # Pydantic ValidationError
        error_details = format_validation_error(error)
    else:
        # Other exceptions (ValueError, KeyError, etc.)
        error_details = f"{type(error).__name__}: {str(error)}"
    
    content = f"""Your previous response had errors:

{error_details}

Please fix these errors and return valid {format_name} data matching the schema exactly."""
    
    if previous_response:
        # Include truncated previous response for context
        truncated = previous_response[:300] + "..." if len(previous_response) > 300 else previous_response
        content = f"""Your previous response:
```
{truncated}
```

Had errors:

{error_details}

Please fix these errors and return valid {format_name} data matching the schema exactly."""
    
    return {
        "role": "user",
        "content": content
    }


def format_validation_error(error: ValidationError) -> str:
    """
    Format a Pydantic ValidationError into a human-readable string.
    
    Args:
        error: The Pydantic ValidationError
    
    Returns:
        Formatted error string
    """
    lines = []
    
    for err in error.errors():
        # Get field path
        loc = err.get("loc", ())
        field_path = " → ".join(str(x) for x in loc) if loc else "root"
        
        # Get error message
        msg = err.get("msg", "Unknown error")
        err_type = err.get("type", "")
        
        # Get expected vs received if available
        ctx = err.get("ctx", {})
        
        error_line = f"• Field '{field_path}': {msg}"
        
        if "expected" in ctx:
            error_line += f" (expected: {ctx['expected']})"
        
        lines.append(error_line)
    
    return "\n".join(lines)


def should_retry(error: Exception, attempt: int, max_retries: int) -> bool:
    """
    Determine if a retry should be attempted.
    
    Args:
        error: The exception that occurred
        attempt: Current attempt number (0-indexed)
        max_retries: Maximum number of retries allowed
    
    Returns:
        True if should retry, False otherwise
    """
    if attempt >= max_retries:
        return False
    
    # Always retry on validation errors
    if isinstance(error, ValidationError):
        return True
    
    # Retry on parse errors
    if isinstance(error, (ValueError, KeyError)):
        return True
    
    return False
