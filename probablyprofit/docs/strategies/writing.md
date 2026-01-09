# Writing Strategies

Strategies are text files that tell the AI how to trade.

## Basic Structure

A good strategy includes:

1. **Trading persona** — Your risk appetite and style
2. **Entry rules** — When to buy
3. **Exit rules** — When to sell (optional)
4. **Position sizing** — How much to bet

## Example

```
You are a conservative value trader.

Entry Rules:
1. Buy YES when price is below 0.25 and fundamentals are strong
2. Buy NO when YES price is above 0.80 and market seems overconfident
3. Only trade markets with volume > $100,000

Position Sizing:
- Bet $15 per trade
- Maximum 5 positions at once

Exit Rules:
- Hold until resolution unless price moves 50%+ against you
```

## Tips

### Be Specific

❌ "Buy good markets"  
✅ "Buy YES when price is below 0.20 on markets about technology"

### Include Context

The AI receives:
- Market question and description
- Current prices (YES/NO)
- Volume and liquidity
- Time until resolution

Reference these in your strategy.

### Use Price Thresholds

```
If YES price < 0.15: Strong buy signal
If YES price > 0.85: Consider buying NO
```

### Set Risk Limits

```
Never bet more than $25 on a single market.
Stop trading if daily loss exceeds $100.
```

## Built-in Examples

Try these starter strategies:

| File | Style |
|------|-------|
| `examples/conservative.txt` | Low risk, high volume only |
| `examples/aggressive.txt` | Force trades, bet on cheap YES |
| `examples/value_hunting.txt` | Find mispriced markets |
| `examples/mean_reversion.txt` | Fade extreme prices |
| `examples/news_driven.txt` | React to breaking news |
