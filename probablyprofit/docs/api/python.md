# Python API

Use probablyprofit programmatically.

## Quick Example

```python
import asyncio
from probablyprofit.api.client import PolymarketClient
from probablyprofit.agent.openai_agent import OpenAIAgent
from probablyprofit.risk.manager import RiskManager

async def main():
    # Initialize client
    client = PolymarketClient(
        api_key="your-key",
        secret="your-secret",
        passphrase="your-passphrase"
    )
    
    # Get markets
    markets = await client.get_markets(limit=10)
    for m in markets:
        print(f"{m.question}: YES={m.outcome_prices[0]:.0%}")
    
    # Check balance
    balance = await client.get_balance()
    print(f"Balance: ${balance:.2f}")
    
    await client.close()

asyncio.run(main())
```

## Core Classes

### PolymarketClient

```python
from probablyprofit.api.client import PolymarketClient

client = PolymarketClient(api_key, secret, passphrase)

# Markets
markets = await client.get_markets(active=True, limit=100)
market = await client.get_market(condition_id)

# Trading
order = await client.place_order(
    market_id="...",
    outcome="Yes",
    side="BUY",
    size=100,
    price=0.5
)

# Portfolio
positions = await client.get_positions()
balance = await client.get_balance()
```

### BaseAgent

```python
from probablyprofit.agent.base import BaseAgent, Decision

class MyAgent(BaseAgent):
    async def decide(self, observation):
        # Your trading logic
        return Decision(
            action="buy",
            market_id="...",
            outcome="Yes",
            size=10,
            price=0.5,
            reasoning="Why I'm making this trade"
        )

# Run the agent
agent = MyAgent(client, risk_manager)
await agent.run_loop()
```

### RiskManager

```python
from probablyprofit.risk.manager import RiskManager

risk = RiskManager(initial_capital=1000.0)

# Check if trade is allowed
can_trade = risk.can_open_position(size=100, price=0.5)

# Calculate position size
kelly_size = risk.calculate_position_size(
    price=0.5,
    confidence=0.7,
    method="kelly"
)
```

## Data Models

### Market

```python
@dataclass
class Market:
    condition_id: str
    question: str
    outcomes: List[str]
    outcome_prices: List[float]
    volume: float
    liquidity: float
    end_date: datetime
```

### Decision

```python
@dataclass
class Decision:
    action: str  # "buy", "sell", "hold"
    market_id: Optional[str]
    outcome: Optional[str]
    size: float
    price: Optional[float]
    reasoning: str
    confidence: float
```
