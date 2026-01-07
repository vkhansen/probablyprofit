# 🤖 poly16z

<div align="center">

**The "Hedge Fund in a Box" for Polymarket**

Write trading strategies in plain English. Let AI agents execute them 24/7.

*Inspired by [ai16z](https://github.com/ai16z) — autonomous AI agents for prediction markets*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Polymarket](https://img.shields.io/badge/Polymarket-Enabled-purple.svg)](https://polymarket.com)

</div>

---

## ⚡ Quick Start (5 minutes)

```bash
# 1. Clone the repo
git clone https://github.com/randomness11/poly16z.git
cd poly16z

# 2. Install dependencies
pip install -e .

# 3. Run setup wizard (creates your .env file)
python scripts/setup_wizard.py

# 4. Write your strategy in strategy.txt
echo "Buy YES on any market about AI if price is below 25 cents" > strategy.txt

# 5. Test in dry-run mode (no real money)
python main.py --strategy custom --dry-run

# 6. Go live!
python main.py --strategy custom
```

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| 🧠 **Multiple AI Agents** | GPT-4, Claude, Gemini - pick your model |
| 📝 **Plain English Strategies** | Write strategies in `strategy.txt`, not code |
| 🔒 **Risk Management** | Built-in daily loss limits, position sizing |
| 🧪 **Dry Run Mode** | Test without risking real money |
| 🐳 **Docker Ready** | One-command deployment |
| 📊 **Real-time Data** | Live prices & volume from Polymarket Gamma API |

---

## 📋 How It Works

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Your Strategy  │ ──▶ │   AI Agent   │ ──▶ │   Polymarket    │
│  (strategy.txt) │     │ (GPT-4/etc)  │     │   (Real Trades) │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

1. **You write** a strategy in plain English
2. **The AI analyzes** live Polymarket data
3. **Trades execute** based on AI decisions

---

## 🛠️ CLI Commands

### Run the Bot
```bash
# Custom strategy with GPT-4
python main.py --strategy custom --agent openai

# News trading for specific keywords
python main.py --strategy news --keywords "Bitcoin,ETH" --agent openai

# Use Gemini (cheaper, faster)
python main.py --strategy custom --agent gemini
```

### Dry Run (Safe Testing)
```bash
python main.py --strategy custom --dry-run
```
Watch the bot analyze markets and log what it *would* trade — without spending money.

### Interactive CLI
```bash
python scripts/cli.py get-markets --limit 10   # View active markets
python scripts/cli.py check-balance            # Check your USDC
python scripts/cli.py get-positions            # See open positions
```

---

## 🧠 Writing Your Strategy

Open `strategy.txt` and write instructions in plain English:

```
You are an aggressive trader looking for undervalued opportunities.

Rules:
1. Focus on markets with over $100k volume
2. If YES price is below 15¢ and you believe it should win, BUY YES
3. If YES price is above 85¢ and seems overpriced, BUY NO
4. Bet $25 per trade
5. Don't be afraid to take positions
```

Then run:
```bash
python main.py --strategy custom --dry-run
```

---

## 🐳 Docker Deployment

```bash
docker compose up --build -d
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

| Variable | Description |
|----------|-------------|
| `PRIVATE_KEY` | Your Polygon wallet private key |
| `POLYMARKET_API_KEY` | Polymarket CLOB API key (auto-derived) |
| `POLYMARKET_API_SECRET` | Polymarket CLOB API secret |
| `POLYMARKET_API_PASSPHRASE` | Polymarket CLOB passphrase |
| `OPENAI_API_KEY` | OpenAI API key |
| `GOOGLE_API_KEY` | Google Gemini API key |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key |
| `INITIAL_CAPITAL` | Your starting capital (for risk calc) |

Run `python scripts/setup_wizard.py` to generate this automatically.

---

## 🔒 Risk Management

The built-in `RiskManager` protects you from blowing up:

- **Max Daily Loss**: Stops trading if you hit your loss limit
- **Position Sizing**: Limits max bet per trade
- **Max Exposure**: Caps total capital at risk

Configure in code:
```python
risk = RiskManager(initial_capital=1000)
risk.limits.max_daily_loss = 50.0    # Stop at -$50
risk.limits.max_position_size = 25.0  # Max $25 per bet
```

---

## 📊 Supported Agents

| Agent | Model | Best For |
|-------|-------|----------|
| OpenAI | `gpt-4o`, `o1-preview` | Complex reasoning |
| Gemini | `gemini-1.5-pro` | Long context, cheaper |
| Anthropic | `claude-3-opus` | Nuanced analysis |

---

## ⚠️ Requirements

- **Python 3.10+** recommended (3.9 works but limited)
- **USDC on Polygon** for trading
- **MATIC** for gas (~$1 worth)

---

## 🚨 Disclaimer

**This is not financial advice.** Prediction markets are risky. You can lose your entire investment. Only trade with money you can afford to lose.

---

## 📄 License

MIT License - do whatever you want with it.

---

<div align="center">

**Built for degens, by degens 🎲**

</div>
