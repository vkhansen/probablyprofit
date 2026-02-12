from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from probablyprofit.agent.base import Decision
from probablyprofit.agent.openai_agent import OpenAIAgent


@pytest.mark.asyncio
async def test_openai_agent_retry_on_schema_error():
    """Test that OpenAIAgent retries on schema validation errors."""
    mock_client = MagicMock()
    mock_risk = MagicMock()

    agent = OpenAIAgent(
        client=mock_client,
        risk_manager=mock_risk,
        openai_api_key="fake-key",
        strategy_prompt="test strategy",
        dry_run=True,
    )

    # Mock the API call to return invalid JSON first, then valid JSON
    # First call: Invalid JSON (triggers SchemaValidationError -> Retry)
    # Second call: Valid JSON (Success)
    invalid_response = MagicMock()
    invalid_response.choices = [MagicMock(message=MagicMock(content="Invalid JSON"))]

    valid_response = MagicMock()
    valid_response.choices = [
        MagicMock(message=MagicMock(content='{"action": "hold", "confidence": 0.8}'))
    ]

    # Mock openai.chat.completions.create
    with patch.object(
        agent.openai.chat.completions, "create", side_effect=[invalid_response, valid_response]
    ) as mock_create:

        observation = MagicMock()
        observation.timestamp = datetime.fromisoformat("2024-01-01")
        observation.markets = []
        observation.positions = []
        observation.balance = 1000.0

        decision = await agent.decide(observation)

        assert isinstance(decision, Decision)
        assert decision.action == "hold"
        assert decision.confidence == 0.8

        # Verify it was called twice (initial + 1 retry)
        assert mock_create.call_count == 2
