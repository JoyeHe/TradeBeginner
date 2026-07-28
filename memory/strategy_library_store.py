"""Strategy library store: seed JSON + Postgres with in-memory fallback."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import uuid

import structlog
from sqlalchemy import DateTime, Float, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from schemas.strategy_library import LibraryOrigin, StrategyLibraryEntry

logger = structlog.get_logger(__name__)


class Base(DeclarativeBase):
    pass


class StrategyLibraryORM(Base):
    __tablename__ = "strategy_library_entries"

    entry_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    template_id: Mapped[Optional[str]] = mapped_column(String(128), index=True, nullable=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, default="default")
    origin: Mapped[str] = mapped_column(String(32), default="baseline")
    quality_score: Mapped[float] = mapped_column(Float, default=0.5, index=True)
    backtest_reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    payload_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    use_count: Mapped[int] = mapped_column(Integer, default=0)
    promote_count: Mapped[int] = mapped_column(Integer, default=0)


class StrategyLibraryStore:
    """In-memory catalog with optional Postgres persistence and JSON seed bootstrap."""

    def __init__(self, postgres_url: str, seed_path: str, learned_path: str) -> None:
        self._postgres_url = postgres_url
        self._seed_path = Path(seed_path)
        self._learned_path = Path(learned_path)
        self._entries: dict[str, StrategyLibraryEntry] = {}
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    async def initialize(self) -> None:
        self._load_json_file(self._seed_path, default_origin=LibraryOrigin.SEED)
        self._load_json_file(self._learned_path, default_origin=LibraryOrigin.PROMOTED)
        try:
            self._engine = create_async_engine(self._postgres_url, pool_pre_ping=True)
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
            await self._load_from_db()
            # Ensure seeds are upserted into DB
            for entry in list(self._entries.values()):
                if entry.origin == LibraryOrigin.SEED:
                    await self._persist(entry)
        except Exception as exc:
            logger.warning("strategy_library_db_unavailable", error=str(exc))
            self._engine = None
            self._session_factory = None
        logger.info("strategy_library_initialized", entries=len(self._entries), seeds=sum(1 for e in self._entries.values() if e.origin == LibraryOrigin.SEED))

    async def shutdown(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()

    def _load_json_file(self, path: Path, default_origin: LibraryOrigin) -> None:
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("strategy_library_json_load_failed", path=str(path), error=str(exc))
            return
        if not isinstance(raw, list):
            return
        for item in raw:
            try:
                if "origin" not in item:
                    item = {**item, "origin": default_origin.value}
                if "entry_id" not in item:
                    item = {**item, "entry_id": str(uuid.uuid4())}
                entry = StrategyLibraryEntry.model_validate(item)
                self._index_entry(entry)
            except Exception as exc:
                logger.warning("strategy_library_seed_row_invalid", error=str(exc))

    def _index_entry(self, entry: StrategyLibraryEntry) -> None:
        # Prefer higher quality when colliding on template_id + user_id for seeds/global
        if entry.template_id:
            for existing in list(self._entries.values()):
                if (
                    existing.template_id == entry.template_id
                    and existing.user_id == entry.user_id
                    and existing.entry_id != entry.entry_id
                ):
                    if (existing.quality_score or 0) > (entry.quality_score or 0):
                        return
                    self._entries.pop(existing.entry_id, None)
        self._entries[entry.entry_id] = entry

    async def _load_from_db(self) -> None:
        if self._session_factory is None:
            return
        async with self._session_factory() as session:
            rows = (await session.scalars(select(StrategyLibraryORM))).all()
            for row in rows:
                try:
                    entry = StrategyLibraryEntry.model_validate(json.loads(row.payload_json))
                    self._index_entry(entry)
                except Exception as exc:
                    logger.warning("strategy_library_db_row_invalid", error=str(exc))

    async def _persist(self, entry: StrategyLibraryEntry) -> None:
        if self._session_factory is None:
            return
        ts = entry.created_at
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        uts = entry.updated_at
        if uts.tzinfo is not None:
            uts = uts.astimezone(timezone.utc).replace(tzinfo=None)
        model = StrategyLibraryORM(
            entry_id=entry.entry_id,
            template_id=entry.template_id,
            user_id=entry.user_id,
            origin=entry.origin.value if isinstance(entry.origin, LibraryOrigin) else str(entry.origin),
            quality_score=float(entry.quality_score or 0),
            backtest_reward=entry.backtest_reward,
            payload_json=entry.model_dump_json(),
            created_at=ts,
            updated_at=uts,
            use_count=entry.use_count,
            promote_count=entry.promote_count,
        )
        async with self._session_factory() as session:
            await session.merge(model)
            await session.commit()

    def _save_learned_json(self) -> None:
        learned = [
            e.model_dump(mode="json")
            for e in self._entries.values()
            if e.origin in (LibraryOrigin.PROMOTED, LibraryOrigin.BASELINE, LibraryOrigin.POST_FEEDBACK)
            and e.user_id != "*"
        ]
        try:
            self._learned_path.parent.mkdir(parents=True, exist_ok=True)
            self._learned_path.write_text(json.dumps(learned, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.warning("strategy_library_learned_save_failed", error=str(exc))

    def add_entry(self, entry: StrategyLibraryEntry) -> StrategyLibraryEntry:
        entry.updated_at = datetime.now(timezone.utc)
        self._index_entry(entry)
        self._save_learned_json()
        return entry

    async def upsert(self, entry: StrategyLibraryEntry) -> StrategyLibraryEntry:
        self.add_entry(entry)
        await self._persist(entry)
        return entry

    def get(self, entry_id: str) -> Optional[StrategyLibraryEntry]:
        return self._entries.get(entry_id)

    def list_entries(
        self,
        user_id: str = "default",
        limit: int = 50,
        tag: Optional[str] = None,
        min_reward: Optional[float] = None,
        include_global: bool = True,
    ) -> list[StrategyLibraryEntry]:
        items = []
        for e in self._entries.values():
            if e.user_id == user_id or (include_global and e.user_id == "*"):
                if tag and tag not in (e.tags or []):
                    continue
                if min_reward is not None and (e.backtest_reward is None or e.backtest_reward < min_reward):
                    if e.origin != LibraryOrigin.SEED:
                        continue
                items.append(e)
        items.sort(key=lambda x: (x.quality_score or 0, x.updated_at), reverse=True)
        return items[:limit]

    def search(self, query: str, user_id: str = "default", limit: int = 20) -> list[StrategyLibraryEntry]:
        tokens = set(query.lower().split())
        scored: list[tuple[float, StrategyLibraryEntry]] = []
        for e in self.list_entries(user_id=user_id, limit=500, include_global=True):
            blob = " ".join(
                [
                    e.name or "",
                    e.description or "",
                    e.template_id or "",
                    " ".join(e.tags or []),
                    " ".join(a.expression for a in (e.alphas or [])),
                ]
            ).lower()
            score = sum(1.0 for t in tokens if t in blob) + (e.quality_score or 0) * 0.1
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:limit]]

    def retrieve_candidates(
        self,
        regime: Optional[str] = None,
        tags: Optional[list[str]] = None,
        user_id: str = "default",
        k: int = 5,
    ) -> list[StrategyLibraryEntry]:
        pool = self.list_entries(user_id=user_id, limit=200, include_global=True)
        scored: list[tuple[float, StrategyLibraryEntry]] = []
        tagset = set(tags or [])
        for e in pool:
            score = float(e.quality_score or 0.5)
            if regime and e.applicable_regimes and regime in e.applicable_regimes:
                score += 0.2
            if tagset and e.tags:
                score += 0.1 * len(tagset.intersection(e.tags))
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]

    async def promote_or_merge(
        self,
        entry: StrategyLibraryEntry,
        *,
        merge_by_template: bool = True,
    ) -> StrategyLibraryEntry:
        """Insert or merge into existing template_id row (sliding quality)."""
        entry.origin = LibraryOrigin.PROMOTED if entry.origin != LibraryOrigin.SEED else entry.origin
        entry.updated_at = datetime.now(timezone.utc)
        if merge_by_template and entry.template_id:
            for existing in list(self._entries.values()):
                if existing.template_id == entry.template_id and existing.user_id in (entry.user_id, "*"):
                    # Update quality as EMA
                    old_q = float(existing.quality_score or 0.5)
                    new_q = float(entry.quality_score if entry.quality_score is not None else entry.backtest_reward or old_q)
                    existing.quality_score = 0.7 * old_q + 0.3 * new_q
                    if entry.backtest_reward is not None:
                        existing.backtest_reward = entry.backtest_reward
                        existing.backtest_metrics = entry.backtest_metrics
                    if entry.strategy is not None:
                        existing.strategy = entry.strategy
                    existing.promote_count = int(existing.promote_count or 0) + 1
                    existing.updated_at = datetime.now(timezone.utc)
                    existing.tags = sorted(set((existing.tags or []) + (entry.tags or [])))
                    await self.upsert(existing)
                    return existing
        entry.promote_count = int(entry.promote_count or 0) + 1
        await self.upsert(entry)
        return entry

    def count(self) -> int:
        return len(self._entries)
