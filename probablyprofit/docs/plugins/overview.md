# Plugins Overview

Extend probablyprofit with custom functionality.

## Plugin Types

| Type | Purpose | Example |
|------|---------|---------|
| `DATA_SOURCE` | Custom data feeds | Twitter sentiment |
| `STRATEGY` | Trading strategies | Momentum trading |
| `AGENT` | Custom AI agents | Rule-based agent |
| `RISK` | Risk rules | Correlation limits |
| `OUTPUT` | Notifications | Discord alerts |

## Using Plugins

Plugins in `plugins/community/` are auto-discovered:

```bash
probablyprofit plugins  # List all plugins
```

## Built-in Examples

### Slack Notifications

```python
from probablyprofit.plugins.examples import SlackNotificationPlugin

plugin = SlackNotificationPlugin(webhook_url="https://hooks.slack.com/...")
await plugin.send("trade", {"market": "...", "action": "buy"})
```

### Whale Tracker

```python
from probablyprofit.plugins.examples import WhaleTrackerPlugin

plugin = WhaleTrackerPlugin(min_bet_size=1000)
data = await plugin.fetch("market_id_here")
# Returns large bets and net whale flow
```

### Momentum Strategy

```python
from probablyprofit.plugins.examples import MomentumStrategyPlugin

strategy = MomentumStrategyPlugin(lookback_hours=24)
print(strategy.get_prompt())
```

## Community Plugins

See `plugins/community/` for more examples:

- `discord_plugin.py` — Discord webhook notifications
- `twitter_sentiment.py` — Twitter/X sentiment data source

[Create Your Own →](creating.md)
