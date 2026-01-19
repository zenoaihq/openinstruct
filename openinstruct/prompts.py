"""
TSON v1.1 System Prompts for LLM communication.

Uses JSON Schema to describe expected structure, but requests TSON output format for token efficiency.
"""

# Full TSON v1.1 system prompt
TSON_SYSTEM_PROMPT = """You are working with TSON (Token-efficient Structured Object Notation), a compact alternative to JSON.

TSON Syntax:
• Objects: {@key1,key2|value1,value2}
• Arrays: [value1,value2,value3]
• Tabular (array of objects): {@key1,key2#N|val1,val2|val1,val2}
• Nested objects: {@field,nested|value,{@subkey1,subkey2|subval1,subval2}}

Delimiters:
• @ = object marker (start of keys)
• , = field/value separator
• | = row separator (between key declaration and values)
• # = row count for arrays of objects

Primitives:
• Strings: Alice or "quoted string" (quote if contains special chars)
• Numbers: 42, 3.14
• Booleans: true, false
• Null: null

Examples:

1. Simple object:
   JSON: {"name": "Alice", "age": 30}
   TSON: {@name,age|Alice,30}

2. Array of objects (tabular):
   JSON: [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
   TSON: {@id,name#2|1,Alice|2,Bob}

3. Nested object:
   JSON: {"user": "John", "address": {"city": "NYC", "country": "USA"}}
   TSON: {@user,address|John,{@city,country|NYC,USA}}

Key Rules:
• Keys written ONCE in header after @, then only values after |
• For nested objects, use full {@keys|values} syntax inside
• Quote strings containing: , | @ # { } [ ]
"""

# System prompt explaining TSON input data
TSON_INPUT_PROMPT = """The input data is in TSON format. Parse it using the same syntax rules."""

# JSON system prompt (fallback)
JSON_SYSTEM_PROMPT = """Return data in valid JSON format matching the provided schema exactly."""

# Schema instruction - uses JSON Schema to describe structure, requests TSON output
TSON_SCHEMA_INSTRUCTION = """
Your response must match this JSON Schema structure:
```json
{schema}
```

{list_hint}

Return your response in TSON format (NOT JSON). Return ONLY the TSON data, no explanations or markdown."""

# JSON schema instruction
JSON_SCHEMA_INSTRUCTION = """
Return your response as valid JSON matching this schema:
```json
{schema}
```
Return ONLY the JSON data, no explanations or markdown."""
