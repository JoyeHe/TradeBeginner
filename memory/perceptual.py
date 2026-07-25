"""Perceptual memory as short-lived raw buffer."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Optional
import uuid

from memory.base import BaseMemory


class PerceptualMemory(BaseMemory):
    """High-throughput short TTL perceptual buffer."""

    def __init__(self, max_items_per_source: int = 500) -> None:
        self._buffers: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self._max_items_per_source = max_items_per_source
        self._lock = __import__("asyncio").Lock()

    async def ingest(self, source: str, data: Any, ttl_seconds: int = 300) -> str:
        """Ingest raw payload and return entry id."""
        entry_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        async with self._lock:
            self._buffers[source].append(
                {"entry_id": entry_id, "data": data, "ingested_at": datetime.utcnow(), "expires_at": expires_at}
            )
            while len(self._buffers[source]) > self._max_items_per_source:
                self._buffers[source].popleft()
        return entry_id

    async def consume(self, source: str, n: Optional[int] = None) -> list[Any]:
        """Consume buffered payloads."""
        async with self._lock:
            buf = self._buffers[source]
            out: list[Any] = []
            limit = len(buf) if n is None else min(n, len(buf))
            for _ in range(limit):
                out.append(buf.popleft()["data"])
            return out

    async def peek(self, source: str, n: int = 1) -> list[Any]:
        """Read buffered payloads without removing."""
        async with self._lock:
            return [entry["data"] for entry in list(self._buffers[source])[:n]]

    async def get_sources(self) -> list[str]:
        """List active sources."""
        async with self._lock:
            return [name for name, queue in self._buffers.items() if queue]

    async def clear_expired(self) -> int:
        """Clear expired entries and return removal count."""
        now = datetime.utcnow()
        removed = 0
        async with self._lock:
            for source in list(self._buffers.keys()):
                kept = deque([entry for entry in self._buffers[source] if entry["expires_at"] > now])
                removed += len(self._buffers[source]) - len(kept)
                self._buffers[source] = kept
        return removed

    async def get_buffer_stats(self) -> dict[str, dict[str, float]]:
        """Per-source count and age details."""
        now = datetime.utcnow()
        async with self._lock:
            stats: dict[str, dict[str, float]] = {}
            for source, queue in self._buffers.items():
                if not queue:
                    continue
                ages = [(now - entry["ingested_at"]).total_seconds() for entry in queue]
                stats[source] = {
                    "count": float(len(queue)),
                    "oldest_entry_age": max(ages),
                    "newest_entry_age": min(ages),
                }
            return stats

    async def health(self) -> dict[str, Any]:
        """Return health metadata."""
        async with self._lock:
            return {"sources": len(self._buffers), "max_items_per_source": self._max_items_per_source}

