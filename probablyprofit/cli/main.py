#!/usr/bin/env python3
"""
probablyprofit CLI

A modern CLI for the AI-powered prediction market trading framework.

Usage:
    probablyprofit run --strategy examples/aggressive.txt
    probablyprofit markets --limit 10
    probablyprofit balance
    probablyprofit dashboard
"""

import asyncio
import os
import sys
import click
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger


# Ensure probablyprofit package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Rich output helpers
def success(msg): click.echo(click.style(f"✅ {msg}", fg="green"))
def error(msg): click.echo(click.style(f"❌ {msg}", fg="red"))
def info(msg): click.echo(click.style(f"ℹ️  {msg}", fg="blue"))
def warn(msg): click.echo(click.style(f"⚠️  {msg}", fg="yellow"))


BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║                     🤖 probablyprofit                          ║
║         AI-Powered Prediction Market Trading Framework        ║
╚═══════════════════════════════════════════════════════════════╝
"""


@click.group()
@click.version_option(version="0.1.0", prog_name="probablyprofit")
def cli():
    """
    probablyprofit - The "Hedge Fund in a Box" for Polymarket.
    
    Write trading strategies in plain English. Let AI agents execute them 24/7.
    
    Quick Start:
    
        probablyprofit init                    # Set up your project
        probablyprofit markets                 # See active markets
        probablyprofit run --dry-run           # Test without real money
        probablyprofit run                     # Go live!
    """
    pass


@cli.command()
@click.option("--strategy", "-s", type=click.Path(exists=True), 
              help="Path to strategy file (e.g., examples/aggressive.txt)")
@click.option("--agent", "-a", type=click.Choice(["openai", "gemini", "anthropic", "ensemble"]),
              default="openai", help="AI agent to use")
@click.option("--dry-run", is_flag=True, help="Simulate trades without real money")
@click.option("--interval", "-i", type=int, default=60, help="Loop interval in seconds")
@click.option("--news", is_flag=True, help="Enable news context via Perplexity")
def run(strategy, agent, dry_run, interval, news):
    """
    Start the trading bot.
    
    Examples:
    
        probablyprofit run --dry-run
        probablyprofit run -s examples/aggressive.txt --agent gemini
        probablyprofit run --news --interval 120
    """
    load_dotenv()
    
    click.echo(BANNER)
    
    mode = "🧪 DRY RUN" if dry_run else "🔴 LIVE TRADING"
    info(f"Mode: {mode}")
    info(f"Agent: {agent}")
    info(f"Interval: {interval}s")
    
    if strategy:
        info(f"Strategy: {strategy}")
    else:
        strategy = "strategy.txt"
        if not os.path.exists(strategy):
            error("No strategy file found. Create strategy.txt or use --strategy")
            return
        info(f"Strategy: {strategy} (default)")
    
    click.echo()
    
    # Build command args
    cmd_args = [
        sys.executable, "main.py",
        "--strategy", "custom",
        "--prompt-file", strategy,
        "--agent", agent,
        "--interval", str(interval),
    ]
    if dry_run:
        cmd_args.append("--dry-run")
    if news:
        cmd_args.append("--news")
    
    # Run the main script
    os.chdir(Path(__file__).parent.parent)
    os.execv(sys.executable, cmd_args)


@cli.command()
@click.option("--limit", "-l", type=int, default=10, help="Number of markets to show")
@click.option("--sort", type=click.Choice(["volume", "end_date"]), default="volume",
              help="Sort order")
def markets(limit, sort):
    """
    List active prediction markets.
    
    Examples:
    
        probablyprofit markets
        probablyprofit markets --limit 20
    """
    async def _markets():
        from probablyprofit.api.client import PolymarketClient
        
        client = PolymarketClient()
        try:
            markets_list = await client.get_markets(active=True, limit=limit)
            
            if not markets_list:
                warn("No active markets found.")
                return
            
            click.echo(f"\n📊 Top {len(markets_list)} Active Markets:\n")
            click.echo("─" * 80)
            
            for i, m in enumerate(markets_list, 1):
                # Format prices
                if len(m.outcome_prices) >= 2:
                    yes_price = m.outcome_prices[0]
                    no_price = m.outcome_prices[1]
                    price_str = f"Yes: {yes_price:.0%} | No: {no_price:.0%}"
                else:
                    price_str = "N/A"
                
                # Truncate question
                q = m.question[:55] + "..." if len(m.question) > 58 else m.question
                
                click.echo(f"{i:2}. {q}")
                click.echo(f"    💰 Volume: ${m.volume:,.0f}  |  {price_str}")
                click.echo()
        finally:
            await client.close()
    
    asyncio.run(_markets())


@cli.command()
def balance():
    """
    Check your wallet balance.
    """
    async def _balance():
        load_dotenv()
        
        from probablyprofit.api.client import PolymarketClient
        
        api_key = os.getenv("POLYMARKET_API_KEY")
        if not api_key:
            error("POLYMARKET_API_KEY not found in .env")
            info("Run 'probablyprofit init' to set up your credentials.")
            return
        
        client = PolymarketClient(
            api_key=api_key,
            secret=os.getenv("POLYMARKET_API_SECRET"),
            passphrase=os.getenv("POLYMARKET_API_PASSPHRASE")
        )
        
        try:
            bal = await client.get_balance()
            click.echo(f"\n💰 Balance: {click.style(f'${bal:,.2f} USDC', fg='green', bold=True)}\n")
        finally:
            await client.close()
    
    asyncio.run(_balance())


@cli.command()
def positions():
    """
    Show your open positions.
    """
    async def _positions():
        load_dotenv()
        
        from probablyprofit.api.client import PolymarketClient
        
        api_key = os.getenv("POLYMARKET_API_KEY")
        if not api_key:
            error("POLYMARKET_API_KEY not found in .env")
            return
        
        client = PolymarketClient(
            api_key=api_key,
            secret=os.getenv("POLYMARKET_API_SECRET"),
            passphrase=os.getenv("POLYMARKET_API_PASSPHRASE")
        )
        
        try:
            pos = await client.get_positions()
            
            if not pos:
                info("No open positions.")
                return
            
            click.echo(f"\n📈 Open Positions ({len(pos)}):\n")
            for p in pos:
                pnl_color = "green" if p.pnl >= 0 else "red"
                click.echo(f"  • {p.outcome} on {p.market_id[:20]}...")
                click.echo(f"    Size: {p.size} | Entry: ${p.avg_price:.2f} | P&L: " +
                          click.style(f"${p.pnl:+.2f}", fg=pnl_color))
        finally:
            await client.close()
    
    asyncio.run(_positions())


@cli.command()
@click.option("--port", "-p", type=int, default=8000, help="Port to run dashboard on")
def dashboard(port):
    """
    Launch the web dashboard.
    
    Opens a real-time monitoring UI in your browser.
    """
    info(f"Starting dashboard on http://localhost:{port}")
    info("Press Ctrl+C to stop")
    
    # Import and run the web server
    try:
        import uvicorn
        from probablyprofit.web.app import create_app
        
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except ImportError:
        error("uvicorn not installed. Run: pip install uvicorn")


@cli.command()
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def init(force):
    """
    Initialize a new probablyprofit project.
    
    Creates .env file and example strategy.
    """
    click.echo(BANNER)
    
    env_file = Path(".env")
    strategy_file = Path("strategy.txt")
    
    # Create .env
    if env_file.exists() and not force:
        warn(".env already exists. Use --force to overwrite.")
    else:
        env_template = Path(__file__).parent.parent / ".env.example"
        if env_template.exists():
            env_file.write_text(env_template.read_text())
            success("Created .env from template")
        else:
            # Fallback
            env_content = """# probablyprofit Configuration
OPENAI_API_KEY=sk-your-key-here
# GOOGLE_API_KEY=your-google-api-key
# ANTHROPIC_API_KEY=sk-ant-your-key

POLYMARKET_API_KEY=
POLYMARKET_API_SECRET=
POLYMARKET_API_PASSPHRASE=

INITIAL_CAPITAL=1000.0
"""
            env_file.write_text(env_content)
            success("Created .env template")
    
    # Create strategy.txt
    if strategy_file.exists() and not force:
        warn("strategy.txt already exists.")
    else:
        strategy_content = """You are a balanced trader seeking good risk-adjusted returns.

Rules:
1. Look for YES prices below 0.25 with strong fundamentals
2. Look for YES prices above 0.80 that seem overconfident
3. Bet $15 per trade
4. Focus on markets with high volume (>$100k)

Make thoughtful trades, not reckless bets.
"""
        strategy_file.write_text(strategy_content)
        success("Created strategy.txt")
    
    click.echo()
    info("Next steps:")
    click.echo("  1. Edit .env with your API keys")
    click.echo("  2. Edit strategy.txt with your trading rules")
    click.echo("  3. Run: probablyprofit run --dry-run")


@cli.command()
@click.option("--strategy", "-s", type=click.Path(exists=True), required=True,
              help="Strategy file to backtest")
@click.option("--days", "-d", type=int, default=30, help="Days of history to simulate")
def backtest(strategy, days):
    """
    Backtest a strategy on historical data.
    
    Example:
    
        probablyprofit backtest -s examples/aggressive.txt --days 60
    """
    async def _backtest():
        from probablyprofit.backtesting.engine import BacktestEngine
        from probablyprofit.backtesting.data import MockDataGenerator
        from probablyprofit.risk.manager import RiskManager
        from probablyprofit.agent.mock_agent import MockAgent
        
        info(f"Running backtest for {days} days...")
        info(f"Strategy: {strategy}")
        click.echo()
        
        # Load strategy
        with open(strategy) as f:
            strategy_text = f.read()
        
        # Generate mock data
        generator = MockDataGenerator()
        markets, timestamps = generator.generate_market_scenario(num_markets=5, days=days)
        
        # Create mock agent
        risk = RiskManager(initial_capital=1000.0)
        agent = MockAgent(None, risk)
        
        # Run backtest
        engine = BacktestEngine(initial_capital=1000.0)
        result = await engine.run_backtest(agent, markets, timestamps)
        
        # Print results
        click.echo("═" * 50)
        click.echo(f"📊 BACKTEST RESULTS")
        click.echo("═" * 50)
        
        ret_color = "green" if result.total_return >= 0 else "red"
        click.echo(f"Return:       " + click.style(f"{result.total_return_pct:+.2%}", fg=ret_color))
        click.echo(f"Sharpe:       {result.sharpe_ratio:.2f}")
        click.echo(f"Max Drawdown: {click.style(f'{result.max_drawdown:.2%}', fg='red')}")
        click.echo(f"Win Rate:     {result.win_rate:.1%}")
        click.echo(f"Total Trades: {result.total_trades}")
        click.echo("═" * 50)
    
    asyncio.run(_backtest())


@cli.command()
def plugins():
    """
    List installed plugins.
    """
    from probablyprofit.plugins import registry
    
    plugins_dict = registry.list_plugins()
    
    if not any(plugins_dict.values()):
        info("No plugins registered.")
        info("See plugins/community/ for examples.")
        return
    
    click.echo("\n🔌 Installed Plugins:\n")
    
    for plugin_type, names in plugins_dict.items():
        if names:
            click.echo(f"  {plugin_type}:")
            for name in names:
                click.echo(f"    • {name}")
    click.echo()


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
