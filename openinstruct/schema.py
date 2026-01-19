"""
Pydantic to TSON/JSON schema conversion.

Converts Pydantic models to compact TSON schema format.
"""

import json
from typing import Any, Type, get_origin, get_args, Union
from pydantic import BaseModel


def pydantic_to_tson_schema(model: Type[BaseModel], is_list: bool = False) -> str:
    """
    Convert a Pydantic model to a TSON schema string.
    
    Args:
        model: Pydantic BaseModel class
        is_list: If True, format as array schema with #N marker
    
    Returns:
        TSON schema string like "@name,age,address(@city,zip)"
    
    Example:
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> pydantic_to_tson_schema(User)
        '@name,age'
        >>> pydantic_to_tson_schema(User, is_list=True)
        '@name,age#N'
    """
    fields = _extract_fields(model)
    schema_parts = []
    
    for field_name, field_info in fields.items():
        field_type = field_info.get("type")
        
        # Check if field is a nested Pydantic model
        if field_info.get("is_model"):
            nested_schema = pydantic_to_tson_schema(field_type, is_list=False)
            # Remove leading @ from nested schema
            nested_inner = nested_schema.lstrip('@').rstrip('#N')
            schema_parts.append(f"{field_name}({nested_inner})")
        else:
            schema_parts.append(field_name)
    
    schema = "@" + ",".join(schema_parts)
    
    if is_list:
        schema += "#N"
    
    return schema


def pydantic_to_json_schema(model: Type[BaseModel]) -> dict:
    """
    Convert a Pydantic model to JSON schema.
    
    Args:
        model: Pydantic BaseModel class
    
    Returns:
        JSON schema dict
    """
    return model.model_json_schema()


def _extract_fields(model: Type[BaseModel]) -> dict[str, dict[str, Any]]:
    """
    Extract field information from a Pydantic model.
    
    Returns dict mapping field names to info dicts containing:
    - type: The Python type
    - is_model: Whether it's a nested Pydantic model
    - is_list: Whether it's a list type
    """
    fields = {}
    
    for field_name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        
        # Handle Optional types
        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            # Filter out NoneType for Optional
            non_none_args = [a for a in args if a is not type(None)]
            if non_none_args:
                annotation = non_none_args[0]
                origin = get_origin(annotation)
        
        # Check for list types
        is_list = False
        inner_type = annotation
        if origin is list:
            is_list = True
            args = get_args(annotation)
            if args:
                inner_type = args[0]
        
        # Check if it's a Pydantic model
        is_model = False
        try:
            if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                is_model = True
        except TypeError:
            pass
        
        fields[field_name] = {
            "type": inner_type,
            "is_model": is_model,
            "is_list": is_list,
            "annotation": annotation,
        }
    
    return fields


def get_type_hint(model: Type[BaseModel]) -> str:
    """
    Get a human-readable type hint for a Pydantic model.
    
    Example:
        >>> get_type_hint(User)
        'User{name:str, age:int}'
    """
    fields = _extract_fields(model)
    parts = []
    
    for field_name, info in fields.items():
        type_name = _get_type_name(info["annotation"])
        parts.append(f"{field_name}:{type_name}")
    
    return f"{model.__name__}{{{', '.join(parts)}}}"


def _get_type_name(annotation) -> str:
    """Get simple type name from annotation."""
    origin = get_origin(annotation)
    
    if origin is list:
        args = get_args(annotation)
        if args:
            inner = _get_type_name(args[0])
            return f"list[{inner}]"
        return "list"
    
    if origin is Union:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return f"Optional[{_get_type_name(non_none[0])}]"
        return f"Union[{', '.join(_get_type_name(a) for a in args)}]"
    
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    
    return str(annotation)


def schema_to_example(model: Type[BaseModel], is_list: bool = False) -> str:
    """
    Generate an example TSON output for a model.
    
    Useful for few-shot prompting.
    """
    tson_schema = pydantic_to_tson_schema(model, is_list=is_list)
    
    # Generate placeholder values
    fields = _extract_fields(model)
    placeholders = []
    
    for field_name, info in fields.items():
        if info["is_model"]:
            # Nested model - use {...}
            placeholders.append("{...}")
        else:
            # Simple type - use placeholder based on type
            type_name = _get_type_name(info["annotation"])
            if "str" in type_name:
                placeholders.append(f"<{field_name}>")
            elif "int" in type_name:
                placeholders.append("0")
            elif "float" in type_name:
                placeholders.append("0.0")
            elif "bool" in type_name:
                placeholders.append("true")
            else:
                placeholders.append(f"<{field_name}>")
    
    if is_list:
        return f"{{{tson_schema}|{','.join(placeholders)}|...}}"
    else:
        return f"{{{tson_schema}|{','.join(placeholders)}}}"
