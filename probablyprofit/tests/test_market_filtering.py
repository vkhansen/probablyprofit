"""
Tests for market filtering functionality.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from probablyprofit.api.client import PolymarketClient
from datetime import datetime
from probablyprofit.config import get_config, APIConfig
from probablyprofit.agent.base import BaseAgent, Decision, Action
from probablyprofit.api.client import Market

class MockRiskManager:
    def can_open_position(self, size, price):
        return True

@pytest.fixture
def mock_client():
    """Fixture for a mocked PolymarketClient."""
    client = PolymarketClient()
    client.get_markets = AsyncMock(return_value=[])
    client.get_tags = AsyncMock(return_value=[])
    return client

class ConcreteTestAgent(BaseAgent):
    """A concrete implementation of BaseAgent for testing."""
    async def decide(self, observation) -> Decision:
        return Decision(action=Action.HOLD, reasoning="Test decision")

@pytest.fixture
def mock_agent(mock_client):
    """Fixture for a BaseAgent with a mocked client."""
    risk_manager = MockRiskManager()
    agent = ConcreteTestAgent(client=mock_client, risk_manager=risk_manager, name="TestAgent")
    return agent

@pytest.mark.asyncio
async def test_tag_resolution(mock_agent):
    """Test that tag slug is correctly resolved to tag ID."""
    mock_agent.client.get_tags.return_value = [
        {"id": 1, "name": "Crypto", "slug": "cryptocurrency"},
        {"id": 2, "name": "Politics", "slug": "politics"},
    ]

    tag_id = await mock_agent._resolve_tag_id("cryptocurrency")
    assert tag_id == 1

    tag_id_none = await mock_agent._resolve_tag_id("non-existent-tag")
    assert tag_id_none is None

@pytest.mark.asyncio
async def test_observe_with_filters(mock_agent):
    """Test the observe method with tag, keyword, and duration filters."""
    # Setup config overrides
    config = get_config()
    config.api = APIConfig(
        market_tag_slug="cryptocurrency",
        market_whitelist_keywords=["15M"],
        market_blacklist_keywords=["daily"],
        market_duration_max_minutes=30,
    )

    mock_agent.client.get_tags.return_value = [{"id": 1, "slug": "cryptocurrency"}]

    # Mock markets
    markets = [
        Market(condition_id="1", question="BTC to hit $100k in 15M?", end_date=datetime.now(), outcomes=["Yes", "No"], outcome_prices=[0.5, 0.5], volume=1000, liquidity=1000),
        Market(condition_id="2", question="ETH daily price movement", end_date=datetime.now(), outcomes=["Yes", "No"], outcome_prices=[0.5, 0.5], volume=1000, liquidity=1000),
        Market(condition_id="3", question="SOL to rally in 15M?", end_date=datetime.now(), outcomes=["Yes", "No"], outcome_prices=[0.5, 0.5], volume=1000, liquidity=1000),
        Market(condition_id="4", question="XRP price in 1 hour", end_date=datetime.now(), outcomes=["Yes", "No"], outcome_prices=[0.5, 0.5], volume=1000, liquidity=1000),
    ]
    mock_agent.client.get_markets.return_value = markets

    # Mock agent._resolve_tag_id to return a value
    mock_agent._resolve_tag_id = AsyncMock(return_value=1)


    observation = await mock_agent.observe()

    # Check that get_markets was called with the right params
    mock_agent.client.get_markets.assert_called_once()
    call_args = mock_agent.client.get_markets.call_args
    assert call_args.kwargs["tag_id"] == 1
    assert "end_date_max" in call_args.kwargs
    assert call_args.kwargs["closed"] is False

    # Check that the markets were filtered correctly
    assert len(observation.markets) == 2
    assert observation.markets[0].question == "BTC to hit $100k in 15M?"
    assert observation.markets[1].question == "SOL to rally in 15M?"
