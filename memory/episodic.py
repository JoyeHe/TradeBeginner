"""Episodic memory backed by SQLAlchemy async storage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
import json
import uuid

from sqlalchemy import DateTime, Float, String, Text, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

import structlog

from memory.base import BaseMemory
from schemas.memory import EpisodicTrace

logger = structlog.get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base for episodic tables."""


class EpisodicTraceORM(Base):
    """SQL table for episodic traces."""

    __tablename__ = "episodic_traces"

    trace_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    market_regime: Mapped[str] = mapped_column(String(32), index=True, default="uncertain")
    reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)
    context_json: Mapped[str] = mapped_column(Text)
    strategy_json: Mapped[str] = mapped_column(Text)
    user_modification_json: Mapped[str] = mapped_column(Text)
    execution_result_json: Mapped[str] = mapped_column(Text)


class EpisodicMemory(BaseMemory):
    """Append-only episodic memory with optional DB fallback."""

    def __init__(self, postgres_url: str) -> None:
        self._postgres_url = postgres_url
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._fallback: dict[str, EpisodicTrace] = {}
        self._lock = __import__("asyncio").Lock()

    async def initialize(self) -> None:
        """Initialize async engine and create tables."""
        try:
            self._engine = create_async_engine(self._postgres_url, pool_pre_ping=True)
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
        except Exception as exc:
            logger.warning("episodic_db_unavailable", error=str(exc))
            self._engine = None
            self._session_factory = None

    async def shutdown(self) -> None:
        """Dispose DB engine."""
        if self._engine is not None:
            await self._engine.dispose()

    async def store_episode(self, episode: EpisodicTrace) -> str:
        """Store a completed episode and return its ID."""
        if self._session_factory is None:
            async with self._lock:
                self._fallback[episode.trace_id] = episode
            return episode.trace_id

        ts = episode.timestamp
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)

        model = EpisodicTraceORM(
            trace_id=episode.trace_id,
            timestamp=ts,
            market_regime=str(episode.context.get("market_regime", "uncertain")),
            reward=episode.reward,
            context_json=json.dumps(episode.context, default=str),
            strategy_json=json.dumps(episode.strategy, default=str),
            user_modification_json=json.dumps(episode.user_modification, default=str),
            execution_result_json=json.dumps(episode.execution_result, default=str),
        )
        async with self._session_factory() as session:
            session.add(model)
            await session.commit()
        return episode.trace_id

    def _to_trace(self, row: EpisodicTraceORM) -> EpisodicTrace:
        return EpisodicTrace(
            trace_id=row.trace_id,
            strategy=json.loads(row.strategy_json),
            user_modification=json.loads(row.user_modification_json),
            execution_result=json.loads(row.execution_result_json),
            reward=row.reward,
            timestamp=row.timestamp,
            context=json.loads(row.context_json),
        )

    async def retrieve_recent(self, n: int = 10) -> list[EpisodicTrace]:
        """Retrieve most recent episodes."""
        if self._session_factory is None:
            traces = sorted(self._fallback.values(), key=lambda x: x.timestamp, reverse=True)
            return traces[:n]
        async with self._session_factory() as session:
            rows = (await session.scalars(select(EpisodicTraceORM).order_by(EpisodicTraceORM.timestamp.desc()).limit(n))).all()
            return [self._to_trace(row) for row in rows]

    async def retrieve_by_regime(self, regime: str, n: int = 10) -> list[EpisodicTrace]:
        """Retrieve episodes by market regime."""
        if self._session_factory is None:
            filtered = [x for x in self._fallback.values() if x.context.get("market_regime") == regime]
            return sorted(filtered, key=lambda x: x.timestamp, reverse=True)[:n]
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(EpisodicTraceORM)
                    .where(EpisodicTraceORM.market_regime == regime)
                    .order_by(EpisodicTraceORM.timestamp.desc())
                    .limit(n)
                )
            ).all()
            return [self._to_trace(row) for row in rows]

    async def retrieve_by_reward_range(self, min_r: float, max_r: float, n: int = 10) -> list[EpisodicTrace]:
        """Retrieve episodes by reward bounds."""
        if self._session_factory is None:
            filtered = [x for x in self._fallback.values() if x.reward is not None and min_r <= x.reward <= max_r]
            return sorted(filtered, key=lambda x: x.timestamp, reverse=True)[:n]
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(EpisodicTraceORM)
                    .where(EpisodicTraceORM.reward >= min_r)
                    .where(EpisodicTraceORM.reward <= max_r)
                    .order_by(EpisodicTraceORM.timestamp.desc())
                    .limit(n)
                )
            ).all()
            return [self._to_trace(row) for row in rows]

    async def retrieve_similar(self, context: dict, n: int = 5) -> list[EpisodicTrace]:
        """Retrieve similar episodes using regime and reward proxy matching."""
        regime = str(context.get("market_regime", "uncertain"))
        traces = await self.retrieve_by_regime(regime, n=max(n * 2, 10))
        target_reward = context.get("target_reward")
        if target_reward is None:
            return traces[:n]
        return sorted(traces, key=lambda t: abs((t.reward or 0.0) - float(target_reward)))[:n]

    async def get_episode(self, episode_id: str) -> Optional[EpisodicTrace]:
        """Get one episode by id."""
        if self._session_factory is None:
            return self._fallback.get(episode_id)
        async with self._session_factory() as session:
            row = await session.get(EpisodicTraceORM, episode_id)
            return None if row is None else self._to_trace(row)

    async def get_statistics(self) -> dict[str, Any]:
        """Return aggregate episodic stats."""
        episodes = await self.retrieve_recent(n=5000)
        rewards = [e.reward for e in episodes if e.reward is not None]
        avg = sum(rewards) / len(rewards) if rewards else 0.0
        return {"total_episodes": len(episodes), "avg_reward": avg, "reward_count": len(rewards)}

    async def health(self) -> dict[str, Any]:
        """Return health metadata."""
        return {"backend": "postgres" if self._session_factory else "in_memory", "id": str(uuid.uuid4())}

