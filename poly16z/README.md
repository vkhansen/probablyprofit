# poly16z

**poly16z** is the ultimate "hedge fund in a box" for Polymarket. It lets you write trading strategies in plain English and have AI agents (Claude, GPT-4, Gemini) execute them for you 24/7.

## 🚀 The Noob's Guide to Setting Up

Okay, so you want to trade on Polymarket with AI but don't know where to start? Follow these steps exactly.

### Step 1: Get Your Money Ready
1. **Create a Polymarket Account**: Go to [polymarket.com](https://polymarket.com) and sign up with email (easier) or a wallet (advanced).
2. **Deposit USDC**: You need USDC (on Polygon network) to trade. You can buy it directly on Polymarket or send it from an exchange like Coinbase/Binance.
3. **Get MATIC**: You need a tiny amount of MATIC (like $1 worth) to pay for transaction fees on Polygon.

### Step 2: Get Your Keys (The Scary Part)
> ⚠️ **WARNING**: Your Private Key gives FULL access to your money. NEVER share it. If you paste it in a Discord or send it to someone, consider your money gone.

1. **Export Private Key**:
   - If you used Email Login: You need to reveal your wallet's private key. Go to your wallet settings on Polymarket (or the Magic wallet dashboard) and look for "Export Private Key".
   - If you used MetaMask/etc: You can export it from the extension.
2. **Get API Keys**:
   - For OpenAI: Go to [platform.openai.com](https://platform.openai.com) -> API Keys.
   - For Anthropic: Go to [console.anthropic.com](https://console.anthropic.com).
   - For Gemini: Go to [aistudio.google.com](https://aistudio.google.com).

### Step 3: Install the Bot
You need Python installed. If you don't have it, download it from [python.org](https://python.org).

1. **Download this folder**:
   ```bash
   git clone https://github.com/your-repo/poly16z.git
   cd poly16z
   ```

2. **Install the magic**:
   ```bash
   # This installs all the dependencies
   pip install -e .
   ```

3. **Configure your secrets**:
   Create a file named `.env` in this folder. Open it with standard text editor (Notepad, TextEdit) and paste this:

   ```env
   # Your Wallet (starts with 0x...)
   PRIVATE_KEY=0x123456789abcdef...
   
   # AI Keys (fill the one you want to use)
   ANTHROPIC_API_KEY=sk-ant-...
   OPENAI_API_KEY=sk-...
   GOOGLE_API_KEY=...
   ```

### Step 4: Run It!

Create a file called `my_bot.py`:

```python
import asyncio
import os
from poly16z.api.client import PolymarketClient
from poly16z.risk.manager import RiskManager
from poly16z.agent.openai_agent import OpenAIAgent

async def main():
    # 1. Connect to Polymarket
    # Note: For real trading, you need to generate specific API keys from your 
    # private key. The client handles some of this, but check docs for details.
    client = PolymarketClient(
        key=os.getenv("POLYMARKET_KEY"),    # You might need to generate these
        passphrase=os.getenv("POLYMARKET_PASS"), # using the get_api_key helper
        secret=os.getenv("POLYMARKET_SECRET")
    )
    
    # 2. Safety First
    risk = RiskManager(initial_capital=500.0) # Start small!
    
    # 3. Create the Agent
    agent = OpenAIAgent(
        client=client,
        risk_manager=risk,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        strategy_prompt="Find markets about Tech Stocks. Buy Yes if recent news is positive."
    )
    
    # 4. Launch
    print("🚀 Blasting off...")
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python my_bot.py
```

## Advanced Stuff

### Agents
We support:
- `AnthropicAgent` (Claude) - Great for complex reasoning.
- `OpenAIAgent` (GPT-4) - Reliable and fast.
- `GeminiAgent` (Google) - Good value and huge context window.

### Risk Management
The bot has a built-in `RiskManager` that prevents you from blowing up your account. It checks:
- Daily loss limits
- Position size limits
- Stop losses

To change limits:
```python
risk = RiskManager()
risk.limits.max_daily_loss = 50.0  # Only lose $50 max per day
```

---
*Disclaimer: This is not financial advice. Crypto trading is risky. You can lose everything. Don't trade money you can't afford to burn.*
