"""
Kalshi API Client

Provides a clean wrapper around the Kalshi API.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from decimal import Decimal

import httpx
from loguru import logger
from pydantic import BaseModel

try:
    from kalshi_python_async import Configuration, KalshiClient as KalshiSDK
    sdk_available = True
except ImportError:
    Configuration = None
    KalshiSDK = None
    sdk_available = False
    logger.warning("kalshi_python_async not installed. Kalshi trading functionality will be limited.")

from probablyprofit.api.exceptions import (
    APIException,
    NetworkException,
    ValidationException,
    OrderException,
)
from probablyprofit.utils.validators import (
    validate_price,
    validate_positive,
    validate_non_negative,
    validate_side,
)


class KalshiMarket(BaseModel):
    """Represents a Kalshi market."""

    ticker: str  # Kalshi uses ticker instead of condition_id
    question: str
    description: Optional[str] = None
    end_date: datetime
    outcomes: List[str] = ["Yes", "No"]  # Kalshi markets are binary
    outcome_prices: List[float] = [0.5, 0.5]
    volume: float = 0.0
    liquidity: float = 0.0
    active: bool = True
    metadata: Dict[str, Any] = {}

    # Kalshi-specific fields
    status: str = "open"  # unopened, open, closed, settled
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None


class KalshiOrder(BaseModel):
    """Represents a Kalshi order."""

    order_id: Optional[str] = None
    ticker: str  # Kalshi uses ticker
    side: str  # yes or no
    action: str  # buy or sell
    count: int  # number of contracts
    price: int  # price in cents
    status: str = "pending"  # resting, canceled, executed
    filled_count: int = 0
    timestamp: datetime = datetime.now()


class KalshiPosition(BaseModel):
    """Represents a position in a Kalshi market."""

    ticker: str
    side: str  # yes or no
    position: int  # number of contracts held
    avg_price: float
    current_price: float
    pnl: float = 0.0

    @property
    def value(self) -> float:
        """Current position value in dollars."""
        return self.position * self.current_price / 100

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss in dollars."""
        return self.position * (self.current_price - self.avg_price) / 100


class KalshiClient:
    """
    High-level wrapper for Kalshi API.

    Provides clean methods for:
    - Fetching market data
    - Placing and managing orders
    - Tracking positions
    - Account balance
    """

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        demo: bool = False,
    ):
        """
        Initialize Kalshi client.

        Args:
            api_key_id: API key ID from Kalshi
            private_key_path: Path to RSA private key file
            demo: Whether to use demo environment
        """
        self.demo = demo
        self.api_key_id = api_key_id

        # Base URL
        self.base_url = "https://api.elections.kalshi.com" if not demo else "https://demo-api.kalshi.com"

        # Initialize SDK client if credentials provided
        if api_key_id and private_key_path and sdk_available:
            try:
                config = Configuration(host=f"{self.base_url}/trade-api/v2")
                config.api_key_id = api_key_id

                # Load private key
                with open(private_key_path, "r") as f:
                    config.private_key_pem = f.read()

                self.client = KalshiSDK(config)
                logger.info("✅ Kalshi client initialized with authentication")
            except Exception as e:
                logger.error(f"Failed to initialize Kalshi SDK: {e}")
                self.client = None
        else:
            # Read-only mode
            self.client = None
            if not sdk_available:
                logger.warning("kalshi_python_async not installed - install with: pip install kalshi_python_async")
            else:
                logger.warning("No API credentials provided - running in read-only mode")

        # HTTP client for public endpoints
        self.http_client = httpx.AsyncClient(
            base_url=f"{self.base_url}/trade-api/v2",
            timeout=30.0,
        )

        # Cache for market data
        self._market_cache: Dict[str, KalshiMarket] = {}
        self._cache_ttl = 60  # Cache for 60 seconds

    async def get_markets(
        self,
        status: str = "open",
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> List[KalshiMarket]:
        """
        Get active Kalshi markets.

        Args:
            status: Market status filter (unopened, open, closed, settled)
            limit: Maximum number of markets to return
            cursor: Pagination cursor

        Returns:
            List of Kalshi markets
        """
        try:
            params = {
                "status": status,
                "limit": limit,
            }
            if cursor:
                params["cursor"] = cursor

            response = await self.http_client.get("/markets", params=params)
            response.raise_for_status()
            data = response.json()

            markets = []
            for market_data in data.get("markets", []):
                market = self._parse_market(market_data)
                markets.append(market)
                self._market_cache[market.ticker] = market

            logger.info(f"Fetched {len(markets)} Kalshi markets")
            return markets

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching Kalshi markets: {e}")
            raise APIException(f"Failed to fetch markets: {e}")
        except Exception as e:
            logger.error(f"Error fetching Kalshi markets: {e}")
            raise NetworkException(f"Network error: {e}")

    async def get_market(self, ticker: str) -> Optional[KalshiMarket]:
        """
        Get a specific market by ticker.

        Args:
            ticker: Market ticker symbol

        Returns:
            Market object or None if not found
        """
        # Check cache first
        if ticker in self._market_cache:
            return self._market_cache[ticker]

        try:
            response = await self.http_client.get(f"/markets/{ticker}")
            response.raise_for_status()
            data = response.json()

            market = self._parse_market(data.get("market", {}))
            self._market_cache[ticker] = market
            return market

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Market {ticker} not found")
                return None
            logger.error(f"HTTP error fetching market {ticker}: {e}")
            raise APIException(f"Failed to fetch market: {e}")
        except Exception as e:
            logger.error(f"Error fetching market {ticker}: {e}")
            raise NetworkException(f"Network error: {e}")

    async def get_orderbook(self, ticker: str) -> Dict[str, Any]:
        """
        Get orderbook for a market.

        Args:
            ticker: Market ticker

        Returns:
            Orderbook data with yes/no bids and asks
        """
        try:
            response = await self.http_client.get(f"/markets/{ticker}/orderbook")
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Error fetching orderbook for {ticker}: {e}")
            raise NetworkException(f"Network error: {e}")

    async def place_order(
        self,
        ticker: str,
        side: str,  # "yes" or "no"
        action: str,  # "buy" or "sell"
        count: int,  # number of contracts
        price: int,  # price in cents (1-99)
    ) -> Optional[KalshiOrder]:
        """
        Place an order on Kalshi.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            price: Price in cents (1-99)

        Returns:
            Order object or None if failed
        """
        if not self.client:
            logger.error("Cannot place order - no authentication configured")
            raise OrderException("Authentication required for trading")

        # Validate inputs
        if side.lower() not in ["yes", "no"]:
            raise ValidationException(f"Invalid side: {side}. Must be 'yes' or 'no'")
        if action.lower() not in ["buy", "sell"]:
            raise ValidationException(f"Invalid action: {action}. Must be 'buy' or 'sell'")
        if not 1 <= price <= 99:
            raise ValidationException(f"Invalid price: {price}. Must be between 1 and 99 cents")
        if count <= 0:
            raise ValidationException(f"Invalid count: {count}. Must be positive")

        try:
            # Use SDK to place order
            order_response = self.client.create_order(
                ticker=ticker,
                client_order_id=f"pp_{int(datetime.now().timestamp())}",  # ProbablyProfit prefix
                side=side.lower(),
                action=action.lower(),
                count=count,
                type="limit",
                yes_price=price if side.lower() == "yes" else None,
                no_price=price if side.lower() == "no" else None,
            )

            order = KalshiOrder(
                order_id=order_response.get("order_id"),
                ticker=ticker,
                side=side,
                action=action,
                count=count,
                price=price,
                status="resting",
                timestamp=datetime.now(),
            )

            logger.info(f"✅ Placed Kalshi order: {action} {count} contracts of {ticker} @ {price}¢")
            return order

        except Exception as e:
            logger.error(f"Failed to place Kalshi order: {e}")
            raise OrderException(f"Order placement failed: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        if not self.client:
            raise OrderException("Authentication required")

        try:
            self.client.cancel_order(order_id)
            logger.info(f"✅ Canceled Kalshi order {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_positions(self) -> List[KalshiPosition]:
        """
        Get current positions.

        Returns:
            List of positions
        """
        if not self.client:
            logger.warning("Cannot get positions - no authentication")
            return []

        try:
            positions_data = self.client.get_positions()
            positions = []

            for pos_data in positions_data.get("positions", []):
                position = KalshiPosition(
                    ticker=pos_data.get("ticker"),
                    side=pos_data.get("side"),
                    position=pos_data.get("position", 0),
                    avg_price=pos_data.get("avg_price", 0),
                    current_price=pos_data.get("market_price", 0),
                )
                positions.append(position)

            return positions

        except Exception as e:
            logger.error(f"Error fetching Kalshi positions: {e}")
            return []

    async def get_balance(self) -> float:
        """
        Get account balance in USD.

        Returns:
            Balance in dollars
        """
        if not self.client:
            logger.warning("Cannot get balance - no authentication")
            return 0.0

        try:
            balance_data = self.client.get_balance()
            # Kalshi returns balance in cents
            balance_cents = balance_data.get("balance", 0)
            balance_usd = balance_cents / 100

            logger.info(f"Kalshi balance: ${balance_usd:,.2f}")
            return balance_usd

        except Exception as e:
            logger.error(f"Error fetching Kalshi balance: {e}")
            return 0.0

    async def get_exchange_status(self) -> Dict[str, Any]:
        """
        Get current exchange operational status.

        Returns:
            Status dict with exchange_active, trading_active, etc.
        """
        try:
            response = await self.http_client.get("/exchange/status")
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.warning(f"Could not fetch exchange status: {e}")
            # Return unknown status rather than assuming inactive
            # This prevents false negatives on temporary network issues
            return {
                "exchange_active": None,  # Unknown
                "trading_active": None,   # Unknown
                "status_unknown": True,
                "error": str(e),
            }

    def _parse_market(self, data: Dict[str, Any]) -> KalshiMarket:
        """Parse Kalshi market data into Market object."""
        # Extract key fields
        ticker = data.get("ticker", "")
        question = data.get("title", "") or data.get("subtitle", "")

        # Parse dates
        close_time = data.get("close_time") or data.get("expiration_time")
        end_date = datetime.fromisoformat(close_time.replace("Z", "+00:00")) if close_time else datetime.now()

        # Get current prices from orderbook if available
        yes_bid = data.get("yes_bid")
        yes_ask = data.get("yes_ask")
        no_bid = data.get("no_bid")
        no_ask = data.get("no_ask")

        # Calculate mid prices
        yes_price = 0.5
        if yes_bid and yes_ask:
            yes_price = (yes_bid + yes_ask) / 200  # Convert cents to 0-1
        elif yes_bid:
            yes_price = yes_bid / 100
        elif yes_ask:
            yes_price = yes_ask / 100

        no_price = 1.0 - yes_price

        return KalshiMarket(
            ticker=ticker,
            question=question,
            description=data.get("rules", ""),
            end_date=end_date,
            outcomes=["Yes", "No"],
            outcome_prices=[yes_price, no_price],
            volume=data.get("volume", 0),
            liquidity=data.get("open_interest", 0),
            active=data.get("status") == "open",
            status=data.get("status", "open"),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            metadata=data,
        )

    async def close(self):
        """Close HTTP clients."""
        await self.http_client.aclose()
        logger.info("Kalshi client closed")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
