# Creating Plugins

Build your own plugins to extend probablyprofit.

## Quick Start

1. Create a file in `plugins/community/`
2. Import the registry and base class
3. Use the `@registry.register()` decorator

## Data Source Plugin

Fetch custom data for trading decisions.

```python
from probablyprofit.plugins import registry, PluginType
from probablyprofit.plugins.base import DataSourcePlugin

@registry.register("my_data", PluginType.DATA_SOURCE)
class MyDataSource(DataSourcePlugin):
    async def fetch(self, query: str):
        # Your data fetching logic
        return {"query": query, "data": "..."}
```

## Strategy Plugin

Custom trading strategy.

```python
from probablyprofit.plugins import registry, PluginType
from probablyprofit.plugins.base import StrategyPlugin

@registry.register("my_strategy", PluginType.STRATEGY)
class MyStrategy(StrategyPlugin):
    def get_prompt(self):
        return "Your trading instructions..."
    
    def filter_markets(self, markets):
        return [m for m in markets if m.volume > 10000]
```

## Output Plugin

Send notifications for events.

```python
from probablyprofit.plugins import registry, PluginType
from probablyprofit.plugins.base import OutputPlugin

@registry.register("my_alerts", PluginType.OUTPUT)
class MyAlerts(OutputPlugin):
    async def send(self, event_type: str, data: dict):
        if event_type == "trade":
            # Send notification
            pass
```

## Plugin Lifecycle

```python
class MyPlugin(BasePlugin):
    async def initialize(self):
        # Called when plugin starts
        await super().initialize()
        self.client = SomeClient()
    
    async def cleanup(self):
        # Called when plugin stops
        await self.client.close()
        await super().cleanup()
```

## Configuration

Use `PluginConfig` for settings:

```python
from probablyprofit.plugins.base import PluginConfig

config = PluginConfig(
    enabled=True,
    priority=10,
    options={"api_key": "..."}
)

plugin = MyPlugin(config=config)
```

## Hooks

React to system events:

```python
from probablyprofit.plugins.hooks import hooks, Hook

@hooks.on(Hook.AFTER_TRADE, priority=100)
async def on_trade(data):
    print(f"Trade executed: {data}")

@hooks.on(Hook.ON_RISK_BREACH)
async def on_risk(data):
    print(f"Risk breach: {data}")
```
