"""
Tests for validation utilities.
"""

import pytest
from pydantic import BaseModel
from probablyprofit.api.exceptions import SchemaValidationError
from probablyprofit.utils.validation_utils import validate_and_parse_decision, clean_json_string

class MockModel(BaseModel):
    name: str
    value: int
    confidence: float

def test_clean_json_string_markdown():
    """Test cleaning markdown code blocks."""
    raw = "```json\n{\"name\": \"test\"}\n```"
    assert clean_json_string(raw) == "{\"name\": \"test\"}"

    raw = "```\n{\"name\": \"test\"}\n```"
    assert clean_json_string(raw) == "{\"name\": \"test\"}"

def test_clean_json_string_plain():
    """Test string that is already clean."""
    raw = "{\"name\": \"test\"}"
    assert clean_json_string(raw) == raw

def test_validate_and_parse_valid():
    """Test valid JSON parsing."""
    json_str = '{"name": "test", "value": 42, "confidence": 0.9}'
    obj = validate_and_parse_decision(json_str, MockModel)
    assert obj.name == "test"
    assert obj.value == 42
    assert obj.confidence == 0.9

def test_validate_and_parse_invalid_json():
    """Test invalid JSON syntax."""
    json_str = '{"name": "test", "value": 42'  # Missing closing brace
    with pytest.raises(SchemaValidationError) as exc:
        validate_and_parse_decision(json_str, MockModel)
    assert "Invalid JSON format" in str(exc.value)

def test_validate_and_parse_schema_mismatch():
    """Test valid JSON but invalid schema."""
    json_str = '{"name": "test", "value": "not_an_int", "confidence": 0.9}'
    with pytest.raises(SchemaValidationError) as exc:
        validate_and_parse_decision(json_str, MockModel)
    assert "Schema validation failed" in str(exc.value)
    assert "value" in str(exc.value)

def test_validate_and_parse_embedded_json():
    """Test extracting JSON from surrounding text."""
    json_str = 'Here is the response: {"name": "test", "value": 42, "confidence": 0.9} Hope this helps.'
    obj = validate_and_parse_decision(json_str, MockModel)
    assert obj.name == "test"
    assert obj.value == 42

def test_validate_and_parse_markdown_embedded():
    """Test extracting JSON from markdown block with surrounding text."""
    json_str = 'Thought process...\n```json\n{"name": "test", "value": 42, "confidence": 0.9}\n```\nReasoning...'
    obj = validate_and_parse_decision(json_str, MockModel)
    assert obj.name == "test"
    assert obj.value == 42
