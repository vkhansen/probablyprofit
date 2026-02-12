"""
Validation utilities for LLM responses.
"""

import json
import re
from typing import TypeVar

from loguru import logger
from pydantic import BaseModel, ValidationError

from probablyprofit.agent.base import Decision
from probablyprofit.api.exceptions import SchemaValidationError

T = TypeVar("T", bound=BaseModel)


def clean_json_string(text: str) -> str:
    """
    Clean markdown formatting from JSON string.

    Args:
        text: Raw text string potentially containing markdown code blocks

    Returns:
        Cleaned JSON string
    """
    # Remove markdown code blocks if present
    if "```" in text:
        # Try to find JSON block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1).strip()

    return text.strip()


def validate_and_parse_decision(response_text: str, model_class: type[T] = Decision) -> T:
    """
    Validate and parse a JSON response into a Pydantic model.

    Args:
        response_text: Raw response text from LLM
        model_class: Pydantic model class to validate against (default: Decision)

    Returns:
        Instance of model_class

    Raises:
        SchemaValidationError: If JSON is invalid or schema validation fails
    """
    try:
        # Clean markdown formatting
        cleaned_text = clean_json_string(response_text)
        data = None

        # Parse JSON
        try:
            data = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            # Try to find a JSON-like object if strict parsing fails
            # This helps when LLM adds text before/after the JSON
            match = re.search(r"(\{[\s\S]*\})", cleaned_text)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError as inner_e:
                    raise SchemaValidationError(f"Invalid JSON format: {e}") from inner_e

            if data is None:
                raise SchemaValidationError(f"Invalid JSON format: {e}") from e

        # Handle case where LLM returns an array of decisions - take the first one
        if isinstance(data, list):
            if not data:
                raise SchemaValidationError("Empty decision array returned")
            logger.warning("LLM returned array of decisions, using first one")
            data = data[0]

        # Validate against Pydantic model
        return model_class.model_validate(data)

    except ValidationError as e:
        # Format validation errors nicely
        errors = e.errors()
        error_msgs = []
        for err in errors:
            loc = ".".join(str(x) for x in err["loc"])
            msg = err["msg"]
            error_msgs.append(f"{loc}: {msg}")

        error_str = "; ".join(error_msgs)
        logger.warning(f"Schema validation failed: {error_str}")
        raise SchemaValidationError(f"Schema validation failed: {error_str}") from e

    except Exception as e:
        if isinstance(e, SchemaValidationError):
            raise
        logger.error(f"Unexpected validation error: {e}")
        raise SchemaValidationError(f"Unexpected validation error: {e}") from e
