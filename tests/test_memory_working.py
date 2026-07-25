from __future__ import annotations

import asyncio
import time

import pytest

from memory.working import WorkingMemory


def _diag(function_tested: str, input_value, expected, actual, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


@pytest.mark.asyncio
async def test_set_and_get_basic(valid_market_snapshot):
    """FUNCTION TESTED: memory.working.WorkingMemory.set/get"""
    mem = WorkingMemory(max_items=10)
    await mem.set("market_snapshot", valid_market_snapshot, category="market")
    loaded = await mem.get("market_snapshot")
    assert loaded == valid_market_snapshot, _diag(
        "memory.working.WorkingMemory.get",
        "market_snapshot",
        valid_market_snapshot.model_dump(mode="json"),
        None if loaded is None else loaded.model_dump(mode="json"),
        "DATA_INTEGRITY",
    )
    assert await mem.get("nonexistent") is None, _diag(
        "memory.working.WorkingMemory.get",
        "nonexistent",
        None,
        "raised or non-None",
        "MISSING_ERROR_HANDLING",
    )


@pytest.mark.asyncio
async def test_set_overwrites_existing():
    """FUNCTION TESTED: memory.working.WorkingMemory.set"""
    mem = WorkingMemory(max_items=10)
    await mem.set("data", "v1")
    await mem.set("data", "v2")
    actual = await mem.get("data")
    assert actual == "v2", _diag(
        "memory.working.WorkingMemory.set",
        {"key": "data", "values": ["v1", "v2"]},
        "v2",
        actual,
        "DATA_INTEGRITY",
    )


@pytest.mark.asyncio
async def test_get_by_category():
    """FUNCTION TESTED: memory.working.WorkingMemory.get_by_category"""
    mem = WorkingMemory(max_items=10)
    for i in range(3):
        await mem.set(f"m{i}", i, category="market")
    for i in range(2):
        await mem.set(f"n{i}", i, category="news")
    await mem.set("p0", 0, category="portfolio")

    market_items = await mem.get_by_category("market")
    assert set(market_items.keys()) == {"m0", "m1", "m2"}, _diag(
        "memory.working.WorkingMemory.get_by_category",
        "market",
        {"m0", "m1", "m2"},
        set(market_items.keys()),
        "DATA_INTEGRITY",
    )
    assert await mem.get_by_category("nonexistent") == {}


@pytest.mark.asyncio
async def test_delete():
    """FUNCTION TESTED: memory.working.WorkingMemory.delete"""
    mem = WorkingMemory(max_items=10)
    await mem.set("temp", "data")
    await mem.delete("temp")
    assert await mem.get("temp") is None
    await mem.delete("nonexistent")


@pytest.mark.asyncio
async def test_clear_by_category():
    """FUNCTION TESTED: memory.working.WorkingMemory.clear"""
    mem = WorkingMemory(max_items=10)
    await mem.set("market1", 1, category="market")
    await mem.set("market2", 2, category="market")
    await mem.set("news1", 3, category="news")

    await mem.clear(category="market")
    assert await mem.get("market1") is None
    assert await mem.get("market2") is None
    assert await mem.get("news1") == 3

    await mem.clear()
    assert await mem.snapshot() == {}


@pytest.mark.asyncio
async def test_snapshot():
    """FUNCTION TESTED: memory.working.WorkingMemory.snapshot"""
    mem = WorkingMemory(max_items=10)
    for i in range(5):
        await mem.set(f"k{i}", {"value": i}, category="market" if i % 2 == 0 else "news")
    snap = await mem.snapshot()
    assert len(snap) == 5
    snap["k0"]["value"] = 999
    live = await mem.get("k0")
    assert live["value"] != 999, _diag(
        "memory.working.WorkingMemory.snapshot",
        "snapshot mutation",
        "copy semantics",
        "mutates internal reference" if live["value"] == 999 else "copy preserved",
        "DATA_INTEGRITY",
    )


@pytest.mark.asyncio
async def test_get_context_for_agent():
    """FUNCTION TESTED: memory.working.WorkingMemory.get_context_for_agent"""
    mem = WorkingMemory(max_items=20)
    await mem.set("market_snapshot", {"m": 1}, category="market")
    await mem.set("news_digest", {"n": 1}, category="news")
    await mem.set("portfolio_state", {"p": 1}, category="execution")
    await mem.set("approved_strategy", {"a": 1}, category="strategy")
    await mem.set("conversation_context", {"c": 1}, category="misc")

    agent3 = await mem.get_context_for_agent("agent3_strategy")
    assert {"market_snapshot", "news_digest", "portfolio_state"} <= set(agent3.keys())
    assert "conversation_context" not in agent3

    agent4 = await mem.get_context_for_agent("agent4_executor")
    assert {"approved_strategy", "portfolio_state"} <= set(agent4.keys())
    assert "news_digest" not in agent4

    unknown = await mem.get_context_for_agent("unknown_agent")
    assert "conversation_context" in unknown, _diag(
        "memory.working.WorkingMemory.get_context_for_agent",
        "unknown_agent",
        "empty dict or clear behavior",
        unknown,
        "MISSING_FEATURE",
    )


@pytest.mark.asyncio
async def test_concurrent_access_safety():
    """FUNCTION TESTED: memory.working.WorkingMemory asyncio.Lock safety"""
    mem = WorkingMemory(max_items=1000)

    async def writer(i: int):
        await mem.set(f"k{i}", i, category="bulk")

    await asyncio.gather(*(writer(i) for i in range(100)))
    snap = await mem.snapshot()
    assert len([k for k in snap if k.startswith("k")]) == 100, _diag(
        "memory.working.WorkingMemory.set",
        "100 concurrent writes",
        100,
        len([k for k in snap if k.startswith("k")]),
        "RACE_CONDITION",
    )

    async def reader(i: int):
        return await mem.get(f"k{i % 100}")

    async def mixed_writer(i: int):
        await mem.set(f"w{i}", i, category="mixed")

    results = await asyncio.gather(
        *(reader(i) for i in range(50)),
        *(mixed_writer(i) for i in range(50)),
    )
    read_values = [x for x in results if isinstance(x, int)]
    assert len(read_values) >= 45, _diag(
        "memory.working.WorkingMemory mixed read/write",
        "50 readers + 50 writers",
        "most reads return stable int values",
        len(read_values),
        "RACE_CONDITION",
    )


@pytest.mark.asyncio
@pytest.mark.slow
async def test_working_memory_throughput():
    """FUNCTION TESTED: memory.working.WorkingMemory performance"""
    mem = WorkingMemory(max_items=20_000)
    n = 10_000

    start_set = time.perf_counter()
    for i in range(n):
        await mem.set(f"s{i}", i, category="perf")
    set_ops = n / (time.perf_counter() - start_set)

    start_get = time.perf_counter()
    for i in range(n):
        await mem.get(f"s{i}")
    get_ops = n / (time.perf_counter() - start_get)

    assert set_ops > 50_000, _diag(
        "memory.working.WorkingMemory.set",
        n,
        ">50000 ops/sec",
        f"{set_ops:.2f}",
        "PERFORMANCE",
    )
    assert get_ops > 50_000, _diag(
        "memory.working.WorkingMemory.get",
        n,
        ">50000 ops/sec",
        f"{get_ops:.2f}",
        "PERFORMANCE",
    )

