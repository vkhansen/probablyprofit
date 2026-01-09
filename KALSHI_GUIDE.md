# Kalshi Integration Guide 🎉

ProbablyProfit now supports **Kalshi** in addition to Polymarket! This guide will help you get started.

## What is Kalshi?

Kalshi is a regulated prediction market platform in the US (CFTC-regulated). Unlike Polymarket which uses crypto, Kalshi operates with traditional USD banking.

## Setup

### 1. Get Kalshi API Credentials

1. Sign up at [kalshi.com](https://kalshi.com)
2. Go to API settings in your account
3. Generate an API key pair (you'll get an API Key ID and need to create/upload an RSA private key)
4. Download your private key file (`.pem` format)

**Important:** Store your private key securely! It's like your wallet's private key.

### 2. Configure Environment Variables

Add to your `.env` file:

```bash
# Kalshi Configuration
KALSHI_API_KEY_ID=your_api_key_id_here
KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi_private_key.pem
KALSHI_DEMO=false  # Set to true for testing with demo environment
```

### 3. Install Dependencies

```bash
pip install kalshi-python-async
```

Or if you're installing from scratch:

```bash
pip install -e .
```

## Usage

### Running with Kalshi

Use the `--platform kalshi` flag:

```bash
# Basic usage
python probablyprofit/main.py --platform kalshi --strategy custom --agent openai

# With custom strategy
python probablyprofit/main.py --platform kalshi --strategy custom --prompt-file my_strategy.txt --agent anthropic

# Dry run mode (highly recommended for testing)
python probablyprofit/main.py --platform kalshi --strategy custom --agent openai --dry-run

# With news intelligence
python probablyprofit/main.py --platform kalshi --strategy custom --agent openai --news --top-markets 5
```

### Example Strategy for Kalshi

Create a `kalshi_strategy.txt` file:

```
You are trading on Kalshi, a regulated US prediction market.

Your goal: Find mispriced election and economic event markets.

Entry rules:
- Look for markets with clear fundamental drivers
- Price should differ >10% from your analysis
- Minimum $500 liquidity
- Use 5% of capital per trade

Exit rules:
- Take profit at 25% gain
- Stop loss at 15% loss
- Exit 24 hours before market close

Risk management:
- Maximum 5 positions at once
- No more than 20% total exposure
- Diversify across different event types
```

## Differences from Polymarket

### Pricing
- **Kalshi**: Prices in **cents** (1-99¢)
- **Polymarket**: Prices as probabilities (0.01-0.99)

The bot handles this conversion automatically.

### Market Structure
- **Kalshi**: All markets are binary (Yes/No)
- **Polymarket**: Can have multiple outcomes

### Settlement
- **Kalshi**: USD settled to your bank account
- **Polymarket**: USDC settled on Polygon

### Fees
- **Kalshi**: Maker/taker fees (check their fee schedule)
- **Polymarket**: Gas fees + exchange fees

## API Differences

### Order Placement

**Kalshi:**
```python
await client.place_order(
    ticker="PRES-2024-DEM",
    side="yes",  # or "no"
    action="buy",  # or "sell"
    count=10,  # number of contracts
    price=55,  # price in cents (55¢ = 55% probability)
)
```

**Polymarket:**
```python
await client.place_order(
    market_id="0x123...",
    outcome="Trump",
    side="BUY",  # or "SELL"
    size=10.0,  # size in USDC
    price=0.55,  # probability (0.55 = 55%)
)
```

## Best Practices

### 1. Start with Demo Mode

Set `KALSHI_DEMO=true` in your `.env` to use Kalshi's demo environment for testing.

### 2. Use Dry Run First

Always test your strategy with `--dry-run` before live trading:

```bash
python probablyprofit/main.py --platform kalshi --strategy custom --dry-run
```

### 3. Monitor API Limits

Kalshi has rate limits. The bot respects these automatically, but be aware:
- Per-second request limits
- Maximum 200,000 open orders per user

### 4. Understand Regulation

Kalshi is CFTC-regulated and has restrictions:
- US residents only
- KYC/AML requirements
- Certain markets may have position limits

## Troubleshooting

### "kalshi_python_async not installed"

Install the SDK:
```bash
pip install kalshi-python-async
```

### "Failed to initialize Kalshi SDK"

Check your private key path:
```bash
# Verify the file exists
ls -la /path/to/kalshi_private_key.pem

# Check permissions
chmod 600 /path/to/kalshi_private_key.pem
```

### "Authentication required for trading"

Make sure your `.env` has:
- `KALSHI_API_KEY_ID`
- `KALSHI_PRIVATE_KEY_PATH` pointing to a valid `.pem` file

### "Market not found"

Kalshi uses tickers like `PRES-2024-DEM`. Check the Kalshi website for exact ticker symbols.

## Example: Running Both Platforms

You can run multiple bots simultaneously (different terminals):

```bash
# Terminal 1: Polymarket bot
python probablyprofit/main.py --platform polymarket --strategy custom --agent openai

# Terminal 2: Kalshi bot
python probablyprofit/main.py --platform kalshi --strategy custom --agent anthropic
```

## Multi-Platform Arbitrage (Coming Soon)

Future versions will support:
- Cross-platform arbitrage detection
- Unified position tracking
- Correlation trading between Polymarket and Kalshi

## Resources

- [Kalshi API Docs](https://docs.kalshi.com)
- [Kalshi Trading Guide](https://kalshi.com/learn)
- [Python SDK Repo](https://github.com/kalshi/kalshi-python)

---

**Questions?** Open an issue on GitHub or check the [main README](README.md).

**WAGMI** 🚀
