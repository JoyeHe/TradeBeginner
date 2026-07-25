"""Memory access helpers used by agents."""

from __future__ import annotations

from memory import MemoryManager



async def write_working(memory: MemoryManager, key: str, value, category: str = "general") -> None:
    """Write into working memory."""
    await memory.working.set(key, value, category=category)


async def read_working(memory: MemoryManager, key: str):
    """Read from working memory."""
    return await memory.working.get(key)


async def save_semantic(memory: MemoryManager, content: str, metadata: dict | None = None, category: str = "general") -> str:
    """Store semantic knowledge."""
    return await memory.semantic.store_knowledge(content=content, metadata=metadata or {}, category=category)

