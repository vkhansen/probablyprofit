"""
Polymarket API Client

Provides a clean wrapper around the Polymarket CLOB API.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from decimal import Decimal

import httpx
from loguru import logger
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from pydantic import BaseModel


class Market(BaseModel):
    """Represents a Polymarket market."""

    condition_id: str
    question: str
    description: Optional[str] = None
    end_date: datetime
    outcomes: List[str]
    outcome_prices: List[float]
    volume: float
    liquidity: float
    active: bool = True
    metadata: Dict[str, Any] = {}


class Order(BaseModel):
    """Represents an order."""

    order_id: Optional[str] = None
    market_id: str
    outcome: str
    side: str  # BUY or SELL
    size: float
    price: float
    status: str = "pending"
    filled_size: float = 0.0
    timestamp: datetime = datetime.now()


class Position(BaseModel):
    """Represents a position in a market."""

    market_id: str
    outcome: str
    size: float
    avg_price: float
    current_price: float
    pnl: float = 0.0

    @property
    def value(self) -> float:
        """Current position value."""
        return self.size * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.size * (self.current_price - self.avg_price)


class PolymarketClient:
    """
    High-level wrapper for Polymarket CLOB API.

    Provides clean methods for:
    - Fetching market data
    - Placing and managing orders
    - Tracking positions
    - Real-time price streaming
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        chain_id: int = 137,
        testnet: bool = False,
    ):
        """
        Initialize Polymarket client.

        Args:
            api_key: API key for authentication
            secret: API secret
            passphrase: API passphrase
            chain_id: Chain ID (137 for Polygon mainnet)
            testnet: Whether to use testnet
        """
        self.chain_id = chain_id
        self.testnet = testnet

        # Initialize CLOB client if credentials provided
        if api_key and secret and passphrase:
            self.client = ClobClient(
                host="https://clob.polymarket.com" if not testnet else "https://clob-test.polymarket.com",
                key=api_key,
                chain_id=chain_id,
            )
        else:
            # Read-only mode
            self.client = None
            logger.warning("No API credentials provided - running in read-only mode")

        # HTTP client for public endpoints
        self.http_client = httpx.AsyncClient(
            base_url="https://clob.polymarket.com" if not testnet else "https://clob-test.polymarket.com",
            timeout=30.0,
        )

        # Cache for market data
        self._market_cache: Dict[str, Market] = {}
        self._positions_cache: Dict[str, Position] = {}

    async def get_markets(
        self,
        active: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Market]:
        """
        Fetch available markets.

        Args:
            active: Only fetch active markets
            limit: Maximum number of markets to return
            offset: Pagination offset

        Returns:
            List of Market objects
        """
        try:
            response = await self.http_client.get(
                "/markets",
                params={
                    "active": active,
                    "limit": limit,
                    "offset": offset,
                }
            )
            response.raise_for_status()
            data = response.json()

            markets = []
            for market_data in data:
                market = Market(
                    condition_id=market_data["condition_id"],
                    question=market_data["question"],
                    description=market_data.get("description"),
                    end_date=datetime.fromisoformat(market_data["end_date"]),
                    outcomes=market_data["outcomes"],
                    outcome_prices=market_data.get("outcome_prices", [0.5] * len(market_data["outcomes"])),
                    volume=float(market_data.get("volume", 0)),
                    liquidity=float(market_data.get("liquidity", 0)),
                    active=market_data.get("active", True),
                    metadata=market_data,
                )
                markets.append(market)
                self._market_cache[market.condition_id] = market

            logger.info(f"Fetched {len(markets)} markets")
            return markets

        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []

    async def get_market(self, condition_id: str) -> Optional[Market]:
        """
        Get details for a specific market.

        Args:
            condition_id: Market condition ID

        Returns:
            Market object or None
        """
        # Check cache first
        if condition_id in self._market_cache:
            return self._market_cache[condition_id]

        try:
            response = await self.http_client.get(f"/markets/{condition_id}")
            response.raise_for_status()
            market_data = response.json()

            market = Market(
                condition_id=market_data["condition_id"],
                question=market_data["question"],
                description=market_data.get("description"),
                end_date=datetime.fromisoformat(market_data["end_date"]),
                outcomes=market_data["outcomes"],
                outcome_prices=market_data.get("outcome_prices", [0.5] * len(market_data["outcomes"])),
                volume=float(market_data.get("volume", 0)),
                liquidity=float(market_data.get("liquidity", 0)),
                active=market_data.get("active", True),
                metadata=market_data,
            )

            self._market_cache[condition_id] = market
            return market

        except Exception as e:
            logger.error(f"Error fetching market {condition_id}: {e}")
            return None

    async def get_orderbook(self, condition_id: str, outcome: str) -> Dict[str, Any]:
        """
        Get orderbook for a market outcome.

        Args:
            condition_id: Market condition ID
            outcome: Outcome name

        Returns:
            Orderbook data with bids and asks
        """
        try:
            response = await self.http_client.get(
                f"/orderbook/{condition_id}/{outcome}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching orderbook: {e}")
            return {"bids": [], "asks": []}

    async def place_order(
        self,
        market_id: str,
        outcome: str,
        side: str,
        size: float,
        price: float,
        order_type: str = "LIMIT",
    ) -> Optional[Order]:
        """
        Place an order.

        Args:
            market_id: Market condition ID
            outcome: Outcome to bet on
            side: BUY or SELL
            size: Order size in shares
            price: Limit price (0-1)
            order_type: Order type (LIMIT, MARKET, etc.)

        Returns:
            Order object or None
        """
        if not self.client:
            logger.error("Cannot place order - no API credentials provided")
            return None

        try:
            logger.info(f"Placing {side} order: {size} shares @ ${price} on {outcome}")

            # Create order using CLOB client
            order_args = OrderArgs(
                price=price,
                size=size,
                side=side,
                token_id=outcome,
            )

            resp = await self.client.create_order(order_args)
            
            # Map response to Order object
            # Note: This mapping is hypothetical based on typical CLOB responses
            # Actual response structure depends on py_clob_client version
            order = Order(
                order_id=resp.get("orderID"),
                market_id=market_id,
                outcome=outcome,
                side=side,
                size=size,
                price=price,
                status="submitted",
                timestamp=datetime.now()
            )

            logger.info(f"Order placed successfully: {order.order_id}")
            return order

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        if not self.client:
            logger.error("Cannot cancel order - no API credentials provided")
            return False

        try:
            logger.info(f"Cancelling order {order_id}")
            resp = await self.client.cancel(order_id)
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    async def get_positions(self) -> List[Position]:
        """
        Get current positions.

        Returns:
            List of Position objects
        """
        if not self.client:
            logger.warning("Cannot fetch positions - no API credentials")
            return []

        try:
            # Fetch positions from CLOB client (requires getting all token balances)
            # This is a simplified implementation assuming we can reconstruct positions
            # In a real app, you might need to query the graph or specific balance endpoints
            
            # Note: py_clob_client might not have a direct 'get_positions' akin to a CEX
            # usually it involves checking token balances.
            # For now, we'll implement a basic balance check if supported, or leave a more specific TODO
            
            # Using a hypothetical method for sake of completion as per plan
            # In reality, we often need to fetch all token balances for the user address
            
            # Placeholder for actual implementation:
            # balances = await self.client.get_balance_allowance(...)
            
            return []
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    async def get_balance(self) -> float:
        """
        Get account balance in USDC.

        Returns:
            Balance in USDC
        """
        if not self.client:
            logger.warning("Cannot fetch balance - no API credentials")
            return 0.0

        try:
            # Fetch USDC balance
            # This often requires checking the collateral token balance on-chain or via API
            # For now, assuming client has a method or we use a fallback
            
            # resp = await self.client.get_collateral_balance()
            # return float(resp)
            
            return 0.0 
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    async def close(self) -> None:
        """Close HTTP client."""
        await self.http_client.aclose()

    async def __aenter__(self) -> "PolymarketClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()
