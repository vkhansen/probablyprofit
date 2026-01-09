# Quick Start

Get your first trading bot running in 5 minutes.

## 1. Initialize Your Project

```bash
probablyprofit init
```

This creates:
- `.env` — Configuration file for API keys
- `strategy.txt` — Your trading strategy

## 2. Configure API Keys

Edit `.env` with your credentials:

```bash
# Required: At least one AI provider
OPENAI_API_KEY=sk-your-openai-key

# Required for live trading
POLYMARKET_API_KEY=your-key
POLYMARKET_API_SECRET=your-secret
POLYMARKET_API_PASSPHRASE=your-passphrase
```

## 3. Write Your Strategy

Edit `strategy.txt` or use an example:

```bash
# Use a built-in example
probablyprofit run -s examples/conservative.txt --dry-run
```

Or write your own in `strategy.txt`:

```
You are a value trader looking for mispriced markets.

Rules:
1. Buy YES when price is below 0.20 and fundamentals support it
2. Buy NO when YES price is above 0.85 and seems overconfident
3. Bet $20 per trade
4. Focus on high-volume markets (>$100k)
```

## 4. Test with Dry Run

```bash
probablyprofit run --dry-run
```

Watch the bot analyze markets and see what it would trade.

## 5. Go Live

When you're ready:

```bash
probablyprofit run
```

!!! warning "Risk Warning"
    Live trading uses real money. Start small and monitor closely.

## 6. Monitor with Dashboard

```bash
probablyprofit dashboard
```

Opens a web UI at `http://localhost:8000` showing:
- Portfolio value
- Open positions
- Trade history
- Real-time P&L

## Next Steps

- [Configuration Reference](configuration.md) — All available settings
- [Writing Strategies](../strategies/writing.md) — Strategy best practices
- [Using Plugins](../plugins/overview.md) — Extend functionality
