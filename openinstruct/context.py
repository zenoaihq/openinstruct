"""
Context injection for input optimization.

Converts context data to TSON format before injecting into prompts.
"""

import json
from typing import Any
from copy import deepcopy

import tson


def inject_context(
    messages: list[dict],
    context: dict[str, Any],
    optimize: bool = True
) -> list[dict]:
    """
    Replace {placeholder} in messages with context data.
    
    If optimize=True, converts list/dict data to TSON format for token efficiency.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        context: Dict mapping placeholder names to data values
        optimize: If True, convert structured data to TSON format
    
    Returns:
        New list of messages with placeholders replaced
    
    Example:
        >>> messages = [{"role": "user", "content": "Analyze: {data}"}]
        >>> context = {"data": [{"id": 1, "name": "Alice"}]}
        >>> inject_context(messages, context, optimize=True)
        [{"role": "user", "content": "Analyze: {@id,name#1|1,Alice}"}]
    """
    # Deep copy to avoid mutating original
    messages = deepcopy(messages)
    
    for key, value in context.items():
        placeholder = f"{{{key}}}"
        
        # Format the value
        if optimize and isinstance(value, (list, dict)):
            # Convert to TSON for token efficiency
            formatted = tson.dumps(value)
        elif isinstance(value, (list, dict)):
            # Use compact JSON
            formatted = json.dumps(value, separators=(',', ':'))
        else:
            # Primitives: use string representation
            formatted = str(value) if value is not None else "null"
        
        # Replace placeholder in all messages
        for msg in messages:
            if "content" in msg and isinstance(msg["content"], str):
                msg["content"] = msg["content"].replace(placeholder, formatted)
    
    return messages


def format_data(data: Any, optimize: bool = True) -> str:
    """
    Format data as TSON (if optimize) or JSON string.
    
    Args:
        data: Any JSON-serializable data
        optimize: If True, use TSON format
    
    Returns:
        Formatted string representation
    """
    if optimize and isinstance(data, (list, dict)):
        return tson.dumps(data)
    elif isinstance(data, (list, dict)):
        return json.dumps(data, separators=(',', ':'))
    else:
        return json.dumps(data)
