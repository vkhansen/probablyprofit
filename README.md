# [±] ProbablyProfit

<div align="center">

### AI-powered prediction market bots in plain English.

[![Website](https://img.shields.io/badge/Website-probablyprofit-22c55e?style=flat-square)](https://randomness11.github.io/probablyprofit/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/badge/Twitter-@ankitkr0-1da1f2?style=flat-square&logo=twitter&logoColor=white)](https://twitter.com/ankitkr0)

**Write strategy in English → Pick your AI → Let it trade**

[Website](https://randomness11.github.io/probablyprofit/) · [Examples](examples/) · [Issues](https://github.com/randomness11/probablyprofit/issues)

</div>

---

## What is this?

ProbablyProfit lets you create trading bots for prediction markets (Polymarket, Kalshi) using plain English. No coding required.

```
You are a value investor for prediction markets.

BUY YES when market price is 15%+ below your estimated probability.
BUY NO when market price is 15%+ above your estimated probability.
Bet $10-20 per trade. Maximum 5 open positions.

AVOID markets you don't understand and prices between 40-60%.
```

That's your entire trading bot. The AI reads markets, applies your strategy, and executes trades.

---

## Quick Start

```bash
pip install probablyprofit

probablyprofit setup          # Configure API keys
probablyprofit run -s strategy.txt --paper   # Paper trade
```

Or use the Python API:

```python
from probablyprofit import AnthropicAgent, PolymarketClient

client = PolymarketClient(private_key="0x...")
agent = AnthropicAgent(
    client=client,
    strategy_prompt="Buy undervalued YES positions on political markets"
)

await agent.run_loop()
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Plain English Strategies** | No code. Write like you'd explain to a friend. |
| **Multi-AI** | GPT-4, Claude, Gemini. Or ensemble mode for consensus. |
| **Multi-Platform** | Polymarket (crypto) + Kalshi (regulated US) |
| **Risk Management** | Kelly sizing, position limits, stop-loss, take-profit |
| **Paper Trading** | Test with virtual money before going live |
| **Backtesting** | Simulate on historical data |
| **Real-time Streaming** | WebSocket price feeds |
| **Web Dashboard** | Monitor positions and P&L |

---

## Installation

```bash
# Full install (recommended)
pip install probablyprofit[full]

# Or minimal + what you need
pip install probablyprofit
pip install probablyprofit[anthropic]   # Claude
pip install probablyprofit[openai]      # GPT-4
pip install probablyprofit[polymarket]  # Polymarket
```

---

## Usage

### CLI

```bash
# Interactive setup
probablyprofit setup

# Run with inline strategy
probablyprofit run "Buy underpriced markets with volume > $5k"

# Run from file
probablyprofit run -s my_strategy.txt

# Modes
probablyprofit run --dry-run "..."   # Analyze only (default)
probablyprofit run --paper "..."     # Virtual money
probablyprofit run --live "..."      # Real money (careful!)

# Other commands
probablyprofit markets               # List markets
probablyprofit markets -q "bitcoin"  # Search
probablyprofit positions             # View positions
probablyprofit balance               # Check wallet
probablyprofit backtest -s strat.txt # Backtest
probablyprofit dashboard             # Web UI
```

### Python API

```python
import asyncio
from probablyprofit import PolymarketClient, AnthropicAgent, RiskManager

async def main():
    client = PolymarketClient(private_key="0x...")
    risk = RiskManager(initial_capital=1000.0, max_position_size=50.0)

    agent = AnthropicAgent(
        client=client,
        risk_manager=risk,
        api_key="sk-ant-...",
        strategy_prompt="""
        Buy YES on markets where price < 0.20 and you estimate
        probability > 0.40. Bet $10-25 based on confidence.
        """
    )

    await agent.run_loop()

asyncio.run(main())
```

---

## Writing Strategies

Strategies are plain text files. Here's the structure:

```
[Role] - Who is the AI?
[Goal] - What are you trying to achieve?
[Rules] - When to buy/sell
[Avoid] - What to stay away from
[Sizing] - How much to bet
```

### Example: Value Investor

```
You are a value investor for prediction markets.

GOAL: Find mispriced markets where the crowd is wrong.

BUY when:
- Market price is 15%+ below your estimated probability
- Volume > $5,000
- Clear resolution criteria

AVOID:
- Markets you don't understand
- Prices between 40-60%
- Low liquidity (< $1k volume)

SIZING: $10-25 per trade, max 5 positions
```

### Example: News Trader

```
You are a news-based trader.

GOAL: React to breaking news before markets adjust.

TRADE when:
- News directly impacts outcome
- Price hasn't moved yet
- You can verify the source

SPEED: Enter within 5 minutes of news
EXIT: Take profit at 50% of expected move
SIZING: $20-50 on high conviction only
```

More examples in [`examples/strategies/`](examples/strategies/)

---

## Configuration

### Environment Variables

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export PRIVATE_KEY=0x...  # Ethereum wallet for Polymarket
```

### Config Files

```
~/.probablyprofit/
├── config.yaml        # Settings
└── credentials.yaml   # API keys (encrypted)
```

Run `probablyprofit setup` for interactive configuration.

---

## Supported Platforms

| Platform | Type | Region | Auth |
|----------|------|--------|------|
| [Polymarket](https://polymarket.com) | Crypto | Global (VPN in US) | Ethereum wallet |
| [Kalshi](https://kalshi.com) | Regulated | US only | Email/password |

---

## Disclaimer

**This is for educational purposes only. Not financial advice.**

- Trading prediction markets involves risk of loss
- Past performance doesn't guarantee future results
- Only trade money you can afford to lose
- Authors are not responsible for losses
- Do your own research

---

## Contributing

PRs welcome! See [CONTRIBUTING.md](probablyprofit/CONTRIBUTING.md)

---

## License

MIT License - see [LICENSE](LICENSE)

---

<div align="center">

Made by [@ankitkr0](https://twitter.com/ankitkr0)

**[± probablyprofit](https://randomness11.github.io/probablyprofit/)**

</div>
