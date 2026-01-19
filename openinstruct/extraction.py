"""
Core extraction logic.

Handles the full pipeline: input prep → LLM call → parsing → validation.
"""

import json
import re
from typing import Any, Type, TypeVar, get_origin, get_args

from pydantic import BaseModel, ValidationError

import tson

from .prompts import (
    TSON_SYSTEM_PROMPT, 
    JSON_SYSTEM_PROMPT,
    TSON_INPUT_PROMPT,
    TSON_SCHEMA_INSTRUCTION,
    JSON_SCHEMA_INSTRUCTION
)
from .schema import pydantic_to_json_schema
from .retry import create_retry_message, should_retry

T = TypeVar('T', bound=BaseModel)


def build_system_prompt(
    response_model: Type[T],
    optimize: bool = True,
    has_tson_context: bool = False
) -> str:
    """
    Build the system prompt for the LLM.
    
    Uses JSON Schema to describe the expected structure, but requests
    TSON output format when optimize=True for token efficiency.
    
    Args:
        response_model: Pydantic model for response
        optimize: Whether to use TSON format for output
        has_tson_context: Whether input contains TSON data
    
    Returns:
        Complete system prompt string
    """
    parts = []
    
    # Add format instruction (TSON syntax or JSON)
    if optimize:
        parts.append(TSON_SYSTEM_PROMPT.strip())
    else:
        parts.append(JSON_SYSTEM_PROMPT.strip())
    
    # Add input format explanation if needed
    if has_tson_context:
        parts.append(TSON_INPUT_PROMPT.strip())
    
    # Get the actual model and check if it's a list type
    is_list = _is_list_type(response_model)
    actual_model = _get_inner_model(response_model)
    
    # Get JSON Schema for the model
    json_schema = pydantic_to_json_schema(actual_model)
    schema_str = json.dumps(json_schema, indent=2)
    
    # Add schema instruction
    if optimize:
        list_hint = "Return an array of objects in TSON tabular format: {@key1,key2#N|val1,val2|...}" if is_list else ""
        schema_instruction = TSON_SCHEMA_INSTRUCTION.format(
            schema=schema_str,
            list_hint=list_hint
        )
    else:
        schema_instruction = JSON_SCHEMA_INSTRUCTION.format(schema=schema_str)
    
    parts.append(schema_instruction.strip())
    
    return "\n\n".join(parts)


def parse_response(
    response_text: str,
    optimize: bool = True
) -> Any:
    """
    Parse LLM response text to extract structured data.
    
    Args:
        response_text: Raw response from LLM
        optimize: Whether TSON format was requested
    
    Returns:
        Parsed Python object (dict or list)
    """
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```"):
        # Extract content between code blocks
        lines = text.split("\n")
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end]).strip()
    
    if optimize:
        # Try TSON first
        # Look for TSON pattern: {@...}
        tson_match = re.search(r'(\{@[^}]*\})', text, re.DOTALL)
        if tson_match:
            tson_str = tson_match.group(1)
            try:
                return tson.loads(tson_str)
            except Exception:
                pass
        
        # Try full text as TSON
        if '{@' in text:
            try:
                return tson.loads(text)
            except Exception:
                pass
        
        # Fallback: try JSON (LLM might return JSON anyway)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON object
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse response as TSON or JSON: {text[:300]}")
    else:
        # JSON mode - try to extract JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from response
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse response as JSON: {text[:300]}")


def validate_response(
    data: Any,
    response_model: Type[T]
) -> T:
    """
    Validate parsed data against Pydantic model.
    
    Args:
        data: Parsed dict or list
        response_model: Pydantic model or list[Model] type
    
    Returns:
        Validated Pydantic model instance(s)
    """
    is_list = _is_list_type(response_model)
    actual_model = _get_inner_model(response_model)
    
    if is_list:
        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data).__name__}")
        return [actual_model.model_validate(item) for item in data]
    else:
        return actual_model.model_validate(data)


def _is_list_type(type_hint) -> bool:
    """Check if type hint is a list type like list[User]."""
    origin = get_origin(type_hint)
    return origin is list


def _get_inner_model(type_hint) -> Type[BaseModel]:
    """Extract the Pydantic model from a type hint."""
    if _is_list_type(type_hint):
        args = get_args(type_hint)
        if args:
            return args[0]
    return type_hint
