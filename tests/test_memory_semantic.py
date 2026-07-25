from __future__ import annotations

import pytest

from memory.semantic import SemanticMemory


def _diag(function_tested: str, input_value, expected, actual, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


@pytest.mark.asyncio
async def test_store_and_query_knowledge(tmp_path):
    """FUNCTION TESTED: memory.semantic.SemanticMemory.store_knowledge/query"""
    mem = SemanticMemory(str(tmp_path / "chroma"))
    await mem.store_knowledge(
        "When RSI exceeds 70, the stock is overbought and likely to decline",
        category="technical_patterns",
    )
    await mem.store_knowledge(
        "Federal Reserve rate hikes tend to strengthen the dollar",
        category="macro",
    )

    rsi_rows = await mem.query("overbought", n=5)
    assert any("overbought" in row["content"].lower() for row in rsi_rows), _diag(
        "memory.semantic.SemanticMemory.query",
        "overbought",
        "RSI overbought content in top results",
        rsi_rows,
        "DATA_INTEGRITY",
    )
    macro_rows = await mem.query("rate hikes", n=5)
    assert any("rate hikes" in row["content"].lower() for row in macro_rows), _diag(
        "memory.semantic.SemanticMemory.query",
        "rate hikes",
        "rate hike content in top results",
        macro_rows,
        "DATA_INTEGRITY",
    )


@pytest.mark.asyncio
async def test_store_and_get_structured_fact(tmp_path):
    """FUNCTION TESTED: memory.semantic.SemanticMemory.store_structured_fact/get_structured_fact"""
    mem = SemanticMemory(str(tmp_path / "chroma"))
    await mem.store_structured_fact("spy_sma200", 450.25, category="indicators")
    value = await mem.get_structured_fact("spy_sma200")
    assert value == 450.25, _diag(
        "memory.semantic.SemanticMemory.get_structured_fact",
        "spy_sma200",
        450.25,
        value,
        "DATA_INTEGRITY",
    )
    missing = await mem.get_structured_fact("not_here")
    assert missing is None


@pytest.mark.asyncio
async def test_update_knowledge(tmp_path):
    """FUNCTION TESTED: memory.semantic.SemanticMemory.update_knowledge"""
    mem = SemanticMemory(str(tmp_path / "chroma"))
    kid = await mem.store_knowledge("Old thesis content", category="macro")
    await mem.update_knowledge(kid, "New thesis content")
    rows = await mem.query("new thesis", n=5)
    assert any("new thesis" in row["content"].lower() for row in rows), _diag(
        "memory.semantic.SemanticMemory.update_knowledge",
        kid,
        "new content present",
        rows,
        "DATA_INTEGRITY",
    )


@pytest.mark.asyncio
async def test_delete_knowledge(tmp_path):
    """FUNCTION TESTED: memory.semantic.SemanticMemory.delete_knowledge"""
    mem = SemanticMemory(str(tmp_path / "chroma"))
    kid = await mem.store_knowledge("Delete me", category="general")
    await mem.delete_knowledge(kid)
    rows = await mem.query("Delete me", n=5)
    assert rows == [], _diag(
        "memory.semantic.SemanticMemory.delete_knowledge",
        kid,
        [],
        rows,
        "DATA_INTEGRITY",
    )
    await mem.delete_knowledge("nonexistent-id")

