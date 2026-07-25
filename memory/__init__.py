"""Unified memory manager for all four tiers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
import uuid

import structlog

from config.settings import Settings
from memory.episodic import EpisodicMemory
from memory.perceptual import PerceptualMemory
from memory.semantic import SemanticMemory
from memory.working import WorkingMemory
from schemas.memory import EpisodicTrace
from schemas.strategy import Strategy, UserModifiedStrategy

logger = structlog.get_logger(__name__)


class MemoryManager:
    """Coordinates working, episodic, semantic, and perceptual memory."""

    def __init__(self, settings: Settings):
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(settings.postgres_url)
        self.semantic = SemanticMemory(settings.chroma_persist_dir)
        self.perceptual = PerceptualMemory()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize persistent memory backends and cleanup task."""
        await self.episodic.initialize()
        await self.semantic.initialize()
        self._cleanup_task = asyncio.create_task(self._perceptual_cleanup_loop())

    async def shutdown(self) -> None:
        """Cleanly shutdown memory backends."""
        self._shutdown.set()
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        await self.episodic.shutdown()

    async def _perceptual_cleanup_loop(self) -> None:
        while not self._shutdown.is_set():
            await self.perceptual.clear_expired()
            await asyncio.sleep(10)

    async def record_episode(
        self,
        strategy: Strategy,
        user_modification: Optional[UserModifiedStrategy],
        execution_result: Any = None,
        reward: Optional[float] = None,
        context: Optional[dict] = None,
    ) -> str:
        """Persist a complete decision episode for replay and training."""
        trace = EpisodicTrace(
            trace_id=str(uuid.uuid4()),
            strategy=strategy.model_dump(mode="json"),
            user_modification=None if user_modification is None else user_modification.model_dump(mode="json"),
            execution_result=execution_result,
            reward=reward,
            timestamp=datetime.now(timezone.utc),
            context=context or {},
        )
        trace_id = await self.episodic.store_episode(trace)
        try:
            summary = strategy.rationale or f"Strategy in {strategy.market_regime.value} regime"
            await self.semantic.store_knowledge(
                content=summary,
                metadata={
                    "strategy_id": strategy.strategy_id,
                    "market_regime": strategy.market_regime.value,
                    "reward": reward,
                },
                category="strategy",
            )
        except Exception as exc:
            logger.warning("semantic_store_failed", error=str(exc))
        return trace_id

    async def get_strategy_context(self) -> dict:
        """Return merged strategy context for Agent 3."""
        snapshot = await self.working.snapshot()
        regime = "uncertain"
        market_snapshot = snapshot.get("market_snapshot")
        if market_snapshot is not None:
            if hasattr(market_snapshot, "market_breadth") and isinstance(market_snapshot.market_breadth, dict):
                regime = market_snapshot.market_breadth.get("regime", "uncertain")
            elif isinstance(market_snapshot, dict):
                breadth = market_snapshot.get("market_breadth") or {}
                regime = breadth.get("regime", "uncertain") if isinstance(breadth, dict) else "uncertain"
        episodes = await self.episodic.retrieve_by_regime(regime=regime, n=5)
        knowledge = await self.semantic.query(query=f"market regime {regime} strategy", n=5)
        return {
            "working_memory": snapshot,
            "recent_episodes": [e.model_dump(mode="json") for e in episodes],
            "semantic_knowledge": knowledge,
        }

    async def consolidate(
        self,
        source: str = "working",
        target: str = "episodic",
        importance_threshold: float = 0.7,
    ) -> int:
        """Promote high-importance memories from source to target tier."""
        promoted = 0
        if source == "working" and target == "episodic":
            snapshot = await self.working.snapshot()
            for key, value in snapshot.items():
                importance = 0.5
                if isinstance(value, dict):
                    importance = value.get("importance", 0.5)
                if importance >= importance_threshold:
                    trace = EpisodicTrace(
                        trace_id=str(uuid.uuid4()),
                        strategy=value,
                        user_modification=None,
                        execution_result=None,
                        reward=None,
                        timestamp=datetime.now(timezone.utc),
                        context={"source": "consolidation", "key": key},
                    )
                    await self.episodic.store_episode(trace)
                    promoted += 1
        logger.info("memory_consolidation", source=source, target=target, promoted=promoted)
        return promoted

    async def forget(self, strategy: str = "importance", threshold: float = 0.3, max_age_days: int = 30) -> int:
        """Run forget strategy across applicable tiers."""
        removed = await self.perceptual.clear_expired()
        if strategy == "importance":
            snapshot = await self.working.snapshot()
            for key, value in snapshot.items():
                importance = 0.5
                if isinstance(value, dict):
                    importance = value.get("importance", 0.5)
                if importance < threshold:
                    await self.working.delete(key)
                    removed += 1
        return removed

