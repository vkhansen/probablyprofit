"""
poly16z - AI-Powered Polymarket Trading Framework

A modular framework for building AI-powered trading bots for Polymarket.
Inspired by a16z's approach to crypto innovation.
"""

__version__ = "0.1.0"

from poly16z.api.client import PolymarketClient
from poly16z.agent.base import BaseAgent
from poly16z.agent.anthropic_agent import AnthropicAgent
from poly16z.risk.manager import RiskManager
from poly16z.backtesting.engine import BacktestEngine

# Optional AI providers
try:
    from poly16z.agent.gemini_agent import GeminiAgent
except ImportError:
    GeminiAgent = None

try:
    from poly16z.agent.openai_agent import OpenAIAgent
except ImportError:
    OpenAIAgent = None

__all__ = [
    "PolymarketClient",
    "BaseAgent",
    "AnthropicAgent",
    "RiskManager",
    "BacktestEngine",
    "GeminiAgent",
    "OpenAIAgent",
]

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
