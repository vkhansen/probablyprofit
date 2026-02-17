# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProbablyProfit is an AI-powered trading framework for Polymarket prediction markets. Users define strategies in plain English; AI models (Claude, GPT-4, Gemini) execute an observe-decide-act loop to trade automatically. Python 3.10+, MIT licensed.

## Common Commands

### Install
```bash
pip install -e ".[full]"         # Everything
pip install -e ".[dev,anthropic]" # Dev + Claude only
```

### Test
```bash
pytest probablyprofit/tests/ -v                    # All tests
pytest probablyprofit/tests/test_agent.py -v       # Single file
pytest probablyprofit/tests/test_agent.py::test_name -v  # Single test
pytest --cov=probablyprofit probablyprofit/tests/  # With coverage
pytest -m "not slow" probablyprofit/tests/         # Skip slow tests
```

Tests use `asyncio_mode = "auto"` (no need for `@pytest.mark.asyncio`). Timeout is 30s per test. Markers: `slow`, `integration`.

### Lint & Format
```bash
ruff check probablyprofit/ --fix   # Lint (primary linter)
black probablyprofit/              # Format
isort probablyprofit/              # Sort imports
mypy probablyprofit/ --ignore-missing-imports  # Type check
bandit -r probablyprofit/ -ll -ii  # Security scan
pre-commit run --all-files         # All hooks at once
```

Line length is 100. Black profile for isort.

### CLI
```bash
probablyprofit run "strategy text" --paper   # Paper trading
probablyprofit run -s strategy.txt --live --confirm-live  # Live
pp markets -q "bitcoin"                      # Search markets (pp is alias)
pp preflight                                 # Health checks
pp emergency-stop                            # Kill switch
```

### CI Pipeline
CI runs on Python 3.10/3.11/3.12: pytest (excluding `test_agent_comprehensive.py`), ruff check, mypy (non-blocking), and bandit.

## Architecture

### Core Agent Loop (probablyprofit/agent/)
All agents inherit `BaseAgent` and implement the **observe → decide → act** cycle:
- **Observe**: Fetch markets, positions, balance, intelligence signals
- **Decide**: AI analyzes observation context and returns BUY/SELL/HOLD with reasoning
- **Act**: Execute orders through the client, respecting risk limits

Agent variants: `AnthropicAgent`, `OpenAIAgent`, `GeminiAgent`, `EnsembleAgent` (multi-AI voting with majority/weighted/unanimous modes), `FallbackAgent` (auto-failover).

### API Client (probablyprofit/api/)
`PolymarketClient` wraps the Polymarket CLOB API. Handles order lifecycle (`order_manager.py`), real-time price feeds (`websocket.py`), and transaction signing (`signer.py`).

### Risk Management (probablyprofit/risk/manager.py)
`RiskManager` enforces Kelly criterion sizing, position/exposure limits, stop-loss/take-profit, daily loss limits, max drawdown halting, and liquidity checks. All trades pass through risk validation before execution.

### Storage (probablyprofit/storage/)
SQLModel ORM with async SQLite (aiosqlite). Repository pattern in `repositories.py`. Alembic migrations in `alembic/`. Models: trades, positions, orders, market history.

### Resilience (probablyprofit/utils/)
- Circuit breakers for API endpoints (`resilience.py`)
- Exponential backoff retries via tenacity
- AI API rate limiting (`ai_rate_limiter.py`)
- Kill switch: file-based, signal-based, or HTTP (`killswitch.py`)
- Log redaction for secrets (`logging.py`)

### Configuration (probablyprofit/config.py)
Loads from `~/.probablyprofit/config.yaml`, `.env`, and environment variables. Covers AI providers, wallet, API settings, risk limits, market filtering, and agent behavior.

### Plugin System (probablyprofit/plugins/)
Event hook architecture with pre/post hooks. Community plugins for Discord and Twitter notifications.

### Backtesting (probablyprofit/backtesting/)
Historical market simulation engine with performance metrics and strategy optimization.

### Frontend (frontend/)
React + Vite + TypeScript web dashboard for monitoring trades and P&L.

## Key Patterns

- **Async-first**: Most I/O operations are async. Tests use `asyncio_mode = "auto"`.
- **Lazy loading**: Package `__init__.py` uses `__getattr__` for fast CLI startup.
- **Factory fixtures**: Tests use `create_mock_market()`, `create_mock_position()`, etc. from `conftest.py` and `mock_exchange.py`.
- **Composition over inheritance**: Strategy objects compose into agents; plugins attach via hooks.
