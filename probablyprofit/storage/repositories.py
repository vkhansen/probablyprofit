"""
Data Repositories

Repository pattern for data access layer - handles all database queries.
"""

from datetime import datetime, timedelta

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from probablyprofit.storage.models import (
    BalanceSnapshot,
    DecisionRecord,
    ObservationRecord,
    TradeRecord,
)


class TradeRepository:
    """Repository for trade records."""

    @staticmethod
    async def create(
        session: AsyncSession,
        order_id: str | None,
        market_id: str,
        outcome: str,
        side: str,
        size: float,
        price: float,
        status: str,
        filled_size: float = 0.0,
        timestamp: datetime | None = None,
        observation_id: int | None = None,
        decision_id: int | None = None,
        realized_pnl: float | None = None,
        fees: float = 0.0,
        market_question: str | None = None,
    ) -> TradeRecord:
        """Create trade record."""
        trade = TradeRecord(
            order_id=order_id,
            market_id=market_id,
            market_question=market_question,
            outcome=outcome,
            side=side,
            size=size,
            price=price,
            status=status,
            filled_size=filled_size,
            timestamp=timestamp or datetime.now(),
            observation_id=observation_id,
            decision_id=decision_id,
            realized_pnl=realized_pnl,
            fees=fees,
        )
        session.add(trade)
        await session.commit()
        await session.refresh(trade)
        logger.debug(f"Saved trade record: {trade.id}")
        return trade

    @staticmethod
    async def get_recent(session: AsyncSession, limit: int = 100) -> list[TradeRecord]:
        """Get recent trades."""
        stmt = select(TradeRecord).order_by(TradeRecord.timestamp.desc()).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_by_market(session: AsyncSession, market_id: str) -> list[TradeRecord]:
        """Get trades for a specific market."""
        stmt = select(TradeRecord).where(TradeRecord.market_id == market_id)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_date_range(
        session: AsyncSession, start: datetime, end: datetime
    ) -> list[TradeRecord]:
        """Get trades within date range."""
        stmt = (
            select(TradeRecord)
            .where(TradeRecord.timestamp >= start)
            .where(TradeRecord.timestamp <= end)
            .order_by(TradeRecord.timestamp)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def search_by_question(
        session: AsyncSession, search_text: str, limit: int = 100
    ) -> list[TradeRecord]:
        """
        Search trades by market question text.

        Allows users to search like "show me election trades" or "politics".

        Args:
            session: Database session
            search_text: Text to search for in market questions
            limit: Maximum results to return

        Returns:
            List of matching trades, sorted by timestamp descending
        """
        # Case-insensitive search using LIKE
        search_pattern = f"%{search_text.lower()}%"
        stmt = (
            select(TradeRecord)
            .where(TradeRecord.market_question.isnot(None))
            .where(TradeRecord.market_question.ilike(search_pattern))
            .order_by(TradeRecord.timestamp.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())


class ObservationRepository:
    """Repository for observations."""

    @staticmethod
    async def create(
        session: AsyncSession,
        timestamp: datetime,
        balance: float,
        num_markets: int,
        num_positions: int,
        markets_json: str,
        positions_json: str = "{}",
        signals_json: str = "{}",
        metadata_json: str = "{}",
        news_context: str | None = None,
        sentiment_summary: str | None = None,
    ) -> ObservationRecord:
        """Create observation record."""
        obs_record = ObservationRecord(
            timestamp=timestamp,
            balance=balance,
            num_markets=num_markets,
            num_positions=num_positions,
            markets_json=markets_json,
            positions_json=positions_json,
            signals_json=signals_json,
            metadata_json=metadata_json,
            news_context=news_context,
            sentiment_summary=sentiment_summary,
        )
        session.add(obs_record)
        await session.commit()
        await session.refresh(obs_record)
        logger.debug(f"Saved observation record: {obs_record.id}")
        return obs_record

    @staticmethod
    async def get_recent(session: AsyncSession, limit: int = 100) -> list[ObservationRecord]:
        """Get recent observations."""
        stmt = select(ObservationRecord).order_by(ObservationRecord.timestamp.desc()).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())


class DecisionRepository:
    """Repository for decisions."""

    @staticmethod
    async def create(
        session: AsyncSession,
        action: str,
        market_id: str | None,
        outcome: str | None,
        size: float,
        price: float | None,
        reasoning: str,
        confidence: float,
        metadata_json: str = "{}",
        observation_id: int | None = None,
        agent_name: str = "unknown",
        agent_type: str = "unknown",
        timestamp: datetime | None = None,
    ) -> DecisionRecord:
        """Create decision record."""
        dec_record = DecisionRecord(
            timestamp=timestamp or datetime.now(),
            action=action,
            market_id=market_id,
            outcome=outcome,
            size=size,
            price=price,
            reasoning=reasoning,
            confidence=confidence,
            metadata_json=metadata_json,
            observation_id=observation_id,
            agent_name=agent_name,
            agent_type=agent_type,
        )
        session.add(dec_record)
        await session.commit()
        await session.refresh(dec_record)
        logger.debug(f"Saved decision record: {dec_record.id}")
        return dec_record

    @staticmethod
    async def get_recent(session: AsyncSession, limit: int = 100) -> list[DecisionRecord]:
        """Get recent decisions."""
        stmt = select(DecisionRecord).order_by(DecisionRecord.timestamp.desc()).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())


class PerformanceRepository:
    """Repository for performance metrics."""

    @staticmethod
    async def create_daily_snapshot(
        session: AsyncSession,
        balance: float,
        exposure: float,
        positions: int,
        daily_pnl: float,
        total_pnl: float,
        timestamp: datetime | None = None,
    ) -> BalanceSnapshot:
        """Create daily balance snapshot."""
        snapshot = BalanceSnapshot(
            timestamp=timestamp or datetime.now(),
            balance=balance,
            total_exposure=exposure,
            num_positions=positions,
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
        )
        session.add(snapshot)
        await session.commit()
        await session.refresh(snapshot)
        logger.debug(f"Saved balance snapshot: {snapshot.id}")
        return snapshot

    @staticmethod
    async def get_equity_curve(session: AsyncSession, days: int = 30) -> list[BalanceSnapshot]:
        """Get equity curve for last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        stmt = (
            select(BalanceSnapshot)
            .where(BalanceSnapshot.timestamp >= cutoff)
            .order_by(BalanceSnapshot.timestamp)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())
