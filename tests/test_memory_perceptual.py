from __future__ import annotations

import asyncio
import time

import pytest

from memory.perceptual import PerceptualMemory


def _diag(function_tested: str, input_value, expected, actual, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


@pytest.mark.asyncio
async def test_ingest_and_consume():
    """FUNCTION TESTED: memory.perceptual.PerceptualMemory.ingest/consume"""
    mem = PerceptualMemory(max_items_per_source=10)
    for i in range(5):
        await mem.ingest("news_feed", {"id": i})
    consumed = await mem.consume("news_feed")
    assert consumed == [{"id": i} for i in range(5)], _diag(
        "memory.perceptual.PerceptualMemory.consume",
        "5 ingested items",
        "FIFO list of 5",
        consumed,
        "DATA_INTEGRITY",
    )
    assert await mem.consume("news_feed") == []


@pytest.mark.asyncio
async def test_peek_does_not_remove():
    """FUNCTION TESTED: memory.perceptual.PerceptualMemory.peek"""
    mem = PerceptualMemory(max_items_per_source=10)
    for i in range(3):
        await mem.ingest("ticks", {"id": i})
    first = await mem.peek("ticks", n=2)
    second = await mem.peek("ticks", n=2)
    assert first == second == [{"id": 0}, {"id": 1}]
    consumed = await mem.consume("ticks")
    assert consumed == [{"id": 0}, {"id": 1}, {"id": 2}]


@pytest.mark.asyncio
async def test_ttl_expiration():
    """FUNCTION TESTED: memory.perceptual.PerceptualMemory.clear_expired"""
    mem = PerceptualMemory(max_items_per_source=10)
    await mem.ingest("ttl", {"id": 1}, ttl_seconds=1)
    assert await mem.peek("ttl", 1) == [{"id": 1}]
    await asyncio.sleep(1.5)
    removed = await mem.clear_expired()
    assert removed == 1, _diag(
        "memory.perceptual.PerceptualMemory.clear_expired",
        "1 expired item",
        1,
        removed,
        "WRONG_CALCULATION",
    )
    assert await mem.peek("ttl", 1) == []

    await mem.ingest("ttl", {"id": 2}, ttl_seconds=3600)
    await asyncio.sleep(0.1)
    removed2 = await mem.clear_expired()
    assert removed2 == 0

    await mem.ingest("ttl", {"id": 3}, ttl_seconds=0)
    await asyncio.sleep(0.1)
    await mem.clear_expired()
    assert await mem.peek("ttl", 10) == [{"id": 2}]


@pytest.mark.asyncio
async def test_buffer_size_limit():
    """FUNCTION TESTED: memory.perceptual.PerceptualMemory bounded buffer"""
    mem = PerceptualMemory(max_items_per_source=10)
    for i in range(15):
        await mem.ingest("bounded", {"id": i})
    remaining = await mem.consume("bounded")
    expected = [{"id": i} for i in range(5, 15)]
    assert remaining == expected, _diag(
        "memory.perceptual.PerceptualMemory.ingest",
        "15 items with max 10",
        expected,
        remaining,
        "CONSTRAINT_NOT_ENFORCED",
    )


@pytest.mark.asyncio
async def test_get_sources_and_buffer_stats():
    """FUNCTION TESTED: memory.perceptual.PerceptualMemory.get_sources/get_buffer_stats"""
    mem = PerceptualMemory(max_items_per_source=10)
    await mem.ingest("news", {"id": 1})
    await mem.ingest("ticks", {"id": 2})
    await mem.ingest("api_raw", {"id": 3})
    sources = set(await mem.get_sources())
    assert sources == {"news", "ticks", "api_raw"}

    stats = await mem.get_buffer_stats()
    assert stats["news"]["count"] == 1.0
    assert stats["ticks"]["count"] == 1.0
    assert stats["api_raw"]["count"] == 1.0

    await mem.consume("news")
    sources2 = set(await mem.get_sources())
    assert "news" not in sources2


@pytest.mark.asyncio
@pytest.mark.slow
async def test_perceptual_memory_throughput():
    """FUNCTION TESTED: memory.perceptual.PerceptualMemory performance"""
    mem = PerceptualMemory(max_items_per_source=20_000)
    n = 10_000

    start_ingest = time.perf_counter()
    for i in range(n):
        await mem.ingest("perf", {"i": i})
    ingest_ops = n / (time.perf_counter() - start_ingest)

    start_consume = time.perf_counter()
    out = await mem.consume("perf", n=n)
    consume_ops = len(out) / (time.perf_counter() - start_consume)

    assert ingest_ops > 10_000, _diag(
        "memory.perceptual.PerceptualMemory.ingest",
        n,
        ">10000 ops/sec",
        f"{ingest_ops:.2f}",
        "PERFORMANCE",
    )
    assert consume_ops > 10_000, _diag(
        "memory.perceptual.PerceptualMemory.consume",
        n,
        ">10000 ops/sec",
        f"{consume_ops:.2f}",
        "PERFORMANCE",
    )

