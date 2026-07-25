from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from config.settings import Settings
from memory import MemoryManager
from schemas.memory import EpisodicTrace


def _diag(function_tested: str, input_value, expected, actual, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


@pytest.mark.asyncio
async def test_initialize_and_shutdown():
    """FUNCTION TESTED: memory.__init__.MemoryManager.initialize/shutdown"""
    settings = Settings(_env_file=None)
    mm = MemoryManager(settings)
    mm.episodic.initialize = AsyncMock(return_value=None)
    mm.semantic.initialize = AsyncMock(return_value=None)
    mm.episodic.shutdown = AsyncMock(return_value=None)

    await mm.initialize()
    assert mm._cleanup_task is not None
    await mm.shutdown()
    await mm.shutdown()
    assert mm.episodic.shutdown.await_count == 2, _diag(
        "memory.__init__.MemoryManager.shutdown",
        "double shutdown",
        "idempotent",
        mm.episodic.shutdown.await_count,
        "MISSING_ERROR_HANDLING",
    )


@pytest.mark.asyncio
async def test_record_episode_convenience(valid_strategy):
    """FUNCTION TESTED: memory.__init__.MemoryManager.record_episode"""
    settings = Settings(_env_file=None)
    mm = MemoryManager(settings)
    mm.episodic.store_episode = AsyncMock(return_value="episode-123")

    episode_id = await mm.record_episode(
        strategy=valid_strategy,
        user_modification=None,
        execution_result={"orders": []},
        reward=0.5,
    )
    assert episode_id == "episode-123"
    assert mm.episodic.store_episode.await_count == 1
    stored_trace = mm.episodic.store_episode.await_args.args[0]
    assert stored_trace.reward == 0.5
    assert stored_trace.execution_result == {"orders": []}


@pytest.mark.asyncio
async def test_get_strategy_context_combines_tiers(valid_market_snapshot):
    """FUNCTION TESTED: memory.__init__.MemoryManager.get_strategy_context"""
    settings = Settings(_env_file=None)
    mm = MemoryManager(settings)
    await mm.working.set("market_snapshot", valid_market_snapshot, category="market")
    await mm.working.set("news_digest", {"overall_market_sentiment": 0.2}, category="news")

    mm.episodic.retrieve_by_regime = AsyncMock(
        return_value=[
            EpisodicTrace(
                trace_id="e1",
                strategy={"id": "s1"},
                user_modification=None,
                execution_result=None,
                reward=0.1,
            )
        ]
    )
    mm.semantic.query = AsyncMock(return_value=[{"content": "use lower risk in bull reversal"}])

    context = await mm.get_strategy_context()
    assert "working_memory" in context
    assert "recent_episodes" in context
    assert "semantic_knowledge" in context
    assert "market_snapshot" in context["working_memory"], _diag(
        "memory.__init__.MemoryManager.get_strategy_context",
        "working memory population",
        "market_snapshot key",
        list(context["working_memory"].keys()),
        "DATA_INTEGRITY",
    )
