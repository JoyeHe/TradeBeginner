"""In-process working memory with category indexing."""

from __future__ import annotations

import asyncio
import copy
from collections import deque
from time import monotonic
from typing import Any, Optional

from memory.base import BaseMemory


class WorkingMemory(BaseMemory):
    """Volatile, bounded, async-safe working memory."""

    def __init__(self, max_items: int = 200, ttl_minutes: int = 60) -> None:
        self._items: dict[str, dict[str, Any]] = {}
        self._order: deque[tuple[str, int]] = deque()
        self._max_items = max_items
        self._ttl_seconds = float(ttl_minutes * 60)
        self._lock = asyncio.Lock()
        self._next_ttl_cleanup_at = monotonic() + 1.0

    def _cleanup_expired(self, now: float) -> None:
        expired = [k for k, v in self._items.items() if v["expires_at"] <= now]
        for key in expired:
            self._items.pop(key, None)

    def _evict_overflow(self) -> None:
        while len(self._items) > self._max_items and self._order:
            oldest_key, oldest_version = self._order.popleft()
            entry = self._items.get(oldest_key)
            # Skip stale order records from overwrites.
            if entry is None or entry["version"] != oldest_version:
                continue
            self._items.pop(oldest_key, None)

    async def set(self, key: str, value: Any, category: str = "general") -> None:
        """Store a value under key and category."""
        async with self._lock:
            now = monotonic()
            previous = self._items.get(key)
            version = 1 if previous is None else int(previous["version"]) + 1
            self._items[key] = {
                "value": value,
                "category": category,
                "version": version,
                "expires_at": now + self._ttl_seconds,
            }
            self._order.append((key, version))
            self._evict_overflow()
            if now >= self._next_ttl_cleanup_at:
                self._cleanup_expired(now)
                self._next_ttl_cleanup_at = now + 1.0

    async def get(self, key: str) -> Any:
        """Read by key."""
        async with self._lock:
            entry = self._items.get(key)
            if entry is not None and entry["expires_at"] <= monotonic():
                self._items.pop(key, None)
                return None
            return None if entry is None else entry["value"]

    async def get_by_category(self, category: str) -> dict[str, Any]:
        """Read all entries in a category."""
        async with self._lock:
            now = monotonic()
            if now >= self._next_ttl_cleanup_at:
                self._cleanup_expired(now)
                self._next_ttl_cleanup_at = now + 1.0
            return {k: v["value"] for k, v in self._items.items() if v["category"] == category}

    async def delete(self, key: str) -> None:
        """Delete by key."""
        async with self._lock:
            self._items.pop(key, None)

    async def clear(self, category: Optional[str] = None) -> None:
        """Clear all or by category."""
        async with self._lock:
            if category is None:
                self._items.clear()
                self._order.clear()
                return

            to_delete = [k for k, v in self._items.items() if v["category"] == category]
            for key in to_delete:
                self._items.pop(key, None)

    async def snapshot(self) -> dict[str, Any]:
        """Return a plain snapshot of memory values."""
        async with self._lock:
            now = monotonic()
            if now >= self._next_ttl_cleanup_at:
                self._cleanup_expired(now)
                self._next_ttl_cleanup_at = now + 1.0
            return {k: copy.deepcopy(v["value"]) for k, v in self._items.items()}

    async def get_context_for_agent(self, agent_name: str) -> dict[str, Any]:
        """Return a minimal context view per agent."""
        mapping: dict[str, list[str]] = {
            "agent1_news": ["market_snapshot", "portfolio_state", "current_user_request"],
            "agent2_market": ["watchlist", "portfolio_state"],
            "agent3_strategy": ["market_snapshot", "news_digest", "portfolio_state", "active_strategy"],
            "agent4_executor": ["approved_strategy", "portfolio_state", "risk_assessment"],
            "agent6_backtest": ["active_strategy", "market_snapshot", "portfolio_state"],
        }
        async with self._lock:
            now = monotonic()
            if now >= self._next_ttl_cleanup_at:
                self._cleanup_expired(now)
                self._next_ttl_cleanup_at = now + 1.0
            keys = mapping.get(agent_name, list(self._items.keys()))
            return {k: self._items[k]["value"] for k in keys if k in self._items}

    async def health(self) -> dict[str, Any]:
        """Return basic health metrics."""
        async with self._lock:
            return {"items": len(self._items), "max_items": self._max_items}

