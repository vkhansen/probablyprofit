# CLI Reference

The `probablyprofit` command-line interface.

## Global Options

```bash
probablyprofit --version  # Show version
probablyprofit --help     # Show help
```

## Commands

### `run`

Start the trading bot.

```bash
probablyprofit run [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-s, --strategy PATH` | Strategy file (default: strategy.txt) |
| `-a, --agent TYPE` | AI agent: openai, gemini, anthropic, ensemble |
| `--dry-run` | Simulate without real trades |
| `-i, --interval SEC` | Loop interval in seconds (default: 60) |
| `--news` | Enable news context via Perplexity |

Examples:

```bash
probablyprofit run --dry-run
probablyprofit run -s examples/aggressive.txt --agent gemini
probablyprofit run --news --interval 120
```

### `init`

Initialize a new project.

```bash
probablyprofit init [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-f, --force` | Overwrite existing files |

### `markets`

List active prediction markets.

```bash
probablyprofit markets [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-l, --limit N` | Number of markets (default: 10) |

### `balance`

Check wallet balance.

```bash
probablyprofit balance
```

### `positions`

Show open positions.

```bash
probablyprofit positions
```

### `dashboard`

Launch web monitoring UI.

```bash
probablyprofit dashboard [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-p, --port N` | Port number (default: 8000) |

### `backtest`

Run strategy backtest.

```bash
probablyprofit backtest [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-s, --strategy PATH` | Strategy file (required) |
| `-d, --days N` | Days of history (default: 30) |

### `plugins`

List installed plugins.

```bash
probablyprofit plugins
```
