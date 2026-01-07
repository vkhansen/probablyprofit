import asyncio
import os
import argparse
import sys
from dotenv import load_dotenv
from loguru import logger


# Add parent directory to path to allow importing this folder as 'poly16z' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from poly16z.api.client import PolymarketClient
from poly16z.risk.manager import RiskManager
from poly16z.agent.openai_agent import OpenAIAgent
from poly16z.agent.gemini_agent import GeminiAgent
from poly16z.agent.strategy import MeanReversionStrategy, NewsTradingStrategy, CustomStrategy

def parse_args():
    parser = argparse.ArgumentParser(description="Poly16z: AI Trading Bot for Polymarket")
    
    parser.add_argument("--strategy", type=str, choices=["mean-reversion", "news", "custom"], default="mean-reversion",
                        help="Trading strategy to employ")
    parser.add_argument("--keywords", type=str, default="",
                        help="Comma-separated keywords for News/Custom strategy")
    parser.add_argument("--prompt-file", type=str, default="strategy.txt",
                        help="Path to custom strategy prompt file (for --strategy custom)")
    parser.add_argument("--agent", type=str, choices=["openai", "gemini"], default="openai",
                        help="AI provider to use")
    parser.add_argument("--model", type=str, default="",
                        help="Specific model name (e.g. 'gpt-4o', 'o1-preview', 'gemini-1.5-pro')")
    parser.add_argument("--interval", type=int, default=60,
                        help="Loop interval in seconds")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without placing real trades (simulation mode)")
    
    return parser.parse_args()

async def main():
    # 0. Load Config
    load_dotenv()
    args = parse_args()
    
    logger.info(f"🚀 Starting Poly16z Bot [Strategy: {args.strategy}] [Agent: {args.agent}]")

    # 1. Initialize Client
    client = PolymarketClient(
        api_key=os.getenv("POLYMARKET_API_KEY"),
        secret=os.getenv("POLYMARKET_API_SECRET"),
        passphrase=os.getenv("POLYMARKET_API_PASSPHRASE")
    )

    # 2. Risk Manager
    risk = RiskManager(initial_capital=float(os.getenv("INITIAL_CAPITAL", 1000.0)))
    
    # 3. Strategy Setup
    strategy = None
    if args.strategy == "mean-reversion":
        strategy = MeanReversionStrategy()
    elif args.strategy == "news":
        if not args.keywords:
            logger.error("❌ You must provide --keywords for news strategy")
            return
        keywords = [k.strip() for k in args.keywords.split(",")]
        strategy = NewsTradingStrategy(keywords=keywords)
    elif args.strategy == "custom":
        if not os.path.exists(args.prompt_file):
            logger.error(f"❌ Custom prompt file not found: {args.prompt_file}")
            print(f"💡 Tip: Create a {args.prompt_file} file with your strategy instructions.")
            return
        
        with open(args.prompt_file, "r") as f:
            prompt_text = f.read()
            
        keywords = [k.strip() for k in args.keywords.split(",")] if args.keywords else []
        strategy = CustomStrategy(prompt_text=prompt_text, keywords=keywords)
        logger.info(f"Loaded Custom Strategy from {args.prompt_file}")
    else:
        logger.error("Unknown strategy.")
        return

    # 4. Agent Setup
    agent = None
    strategy_prompt = strategy.get_prompt()
    
    if args.agent == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("❌ Missing OPENAI_API_KEY in .env")
            return
        
        # Default model logic
        model = args.model if args.model else "gpt-4o"
        agent = OpenAIAgent(client, risk, api_key, strategy_prompt, model=model, loop_interval=args.interval, strategy=strategy, dry_run=args.dry_run)
        
    elif args.agent == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("❌ Missing GOOGLE_API_KEY in .env")
            return
            
        model = args.model if args.model else "gemini-1.5-pro"
        agent = GeminiAgent(client, risk, api_key, strategy_prompt, model=model, loop_interval=args.interval, strategy=strategy, dry_run=args.dry_run)

    # 5. Run
    try:
        if agent:
            await agent.run()
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user.")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
