from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memory.episodic import EpisodicMemory
from schemas.memory import EpisodicTrace


def _diag(function_tested: str, input_value, expected, actual, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


def _trace(trace_id: str, reward: float | None, regime: str, ts: datetime) -> EpisodicTrace:
    return EpisodicTrace(
        trace_id=trace_id,
        strategy={"asset": "AAPL", "id": trace_id},
        user_modification=None,
        execution_result={"filled": True},
        reward=reward,
        timestamp=ts,
        context={"market_regime": regime},
    )


@pytest.mark.asyncio
async def test_store_and_retrieve_episode():
    """FUNCTION TESTED: memory.episodic.EpisodicMemory.store_episode/get_episode"""
    mem = EpisodicMemory("postgresql+asyncpg://invalid:invalid@127.0.0.1:1/invalid")
    trace = _trace("t1", 0.75, "bull", datetime.now(timezone.utc))
    episode_id = await mem.store_episode(trace)
    loaded = await mem.get_episode(episode_id)
    assert loaded is not None, _diag(
        "memory.episodic.EpisodicMemory.get_episode",
        episode_id,
        "non-None trace",
        loaded,
        "DATA_INTEGRITY",
    )
    assert loaded.trace_id == trace.trace_id
    assert loaded.reward == trace.reward


@pytest.mark.asyncio
async def test_retrieve_recent_ordering():
    """FUNCTION TESTED: memory.episodic.EpisodicMemory.retrieve_recent"""
    mem = EpisodicMemory("postgresql+asyncpg://invalid:invalid@127.0.0.1:1/invalid")
    base = datetime.now(timezone.utc)
    for i in range(5):
        await mem.store_episode(_trace(f"r{i}", 0.1 * i, "bull", base + timedelta(hours=i)))
    recent = await mem.retrieve_recent(n=3)
    ids = [x.trace_id for x in recent]
    assert ids == ["r4", "r3", "r2"], _diag(
        "memory.episodic.EpisodicMemory.retrieve_recent",
        "5 episodes, n=3",
        ["r4", "r3", "r2"],
        ids,
        "WRONG_CALCULATION",
    )


@pytest.mark.asyncio
async def test_retrieve_by_regime():
    """FUNCTION TESTED: memory.episodic.EpisodicMemory.retrieve_by_regime"""
    mem = EpisodicMemory("postgresql+asyncpg://invalid:invalid@127.0.0.1:1/invalid")
    now = datetime.now(timezone.utc)
    for i in range(3):
        await mem.store_episode(_trace(f"b{i}", 0.1, "bull", now + timedelta(minutes=i)))
    for i in range(2):
        await mem.store_episode(_trace(f"br{i}", -0.1, "bear", now + timedelta(minutes=10 + i)))
    await mem.store_episode(_trace("s0", 0.0, "sideways", now + timedelta(minutes=20)))

    bull = await mem.retrieve_by_regime("bull", n=10)
    assert len(bull) == 3
    volatile = await mem.retrieve_by_regime("volatile", n=10)
    assert len(volatile) == 0


@pytest.mark.asyncio
async def test_retrieve_by_reward_range():
    """FUNCTION TESTED: memory.episodic.EpisodicMemory.retrieve_by_reward_range"""
    mem = EpisodicMemory("postgresql+asyncpg://invalid:invalid@127.0.0.1:1/invalid")
    now = datetime.now(timezone.utc)
    rewards = [-0.5, 0.0, 0.3, 0.7, 0.9]
    for i, r in enumerate(rewards):
        await mem.store_episode(_trace(f"rw{i}", r, "bull", now + timedelta(minutes=i)))

    hi = await mem.retrieve_by_reward_range(0.5, 1.0)
    assert sorted([x.reward for x in hi]) == [0.7, 0.9]
    lo = await mem.retrieve_by_reward_range(-1.0, -0.1)
    assert [x.reward for x in lo] == [-0.5]
    none = await mem.retrieve_by_reward_range(0.95, 1.0)
    assert none == []


@pytest.mark.asyncio
async def test_episode_append_only():
    """FUNCTION TESTED: memory.episodic.EpisodicMemory append-only invariant"""
    mem = EpisodicMemory("postgresql+asyncpg://invalid:invalid@127.0.0.1:1/invalid")
    original = _trace("immut", 0.2, "bull", datetime.now(timezone.utc))
    await mem.store_episode(original)
    assert not hasattr(mem, "update_episode"), _diag(
        "memory.episodic.EpisodicMemory",
        "update attempt",
        "no mutating update method",
        "method exists",
        "MISSING_FEATURE",
    )
    loaded = await mem.get_episode("immut")
    assert loaded is not None and loaded.reward == 0.2


@pytest.mark.asyncio
async def test_get_statistics():
    """FUNCTION TESTED: memory.episodic.EpisodicMemory.get_statistics"""
    mem = EpisodicMemory("postgresql+asyncpg://invalid:invalid@127.0.0.1:1/invalid")
    now = datetime.now(timezone.utc)
    rewards = [0.0, 0.1, 0.2, 0.3, 0.4]
    for i, r in enumerate(rewards):
        await mem.store_episode(_trace(f"st{i}", r, "bull", now + timedelta(minutes=i)))
    stats = await mem.get_statistics()
    assert stats["total_episodes"] == 5
    assert stats["reward_count"] == 5
    assert abs(stats["avg_reward"] - 0.2) < 1e-9, _diag(
        "memory.episodic.EpisodicMemory.get_statistics",
        rewards,
        0.2,
        stats["avg_reward"],
        "WRONG_CALCULATION",
    )

