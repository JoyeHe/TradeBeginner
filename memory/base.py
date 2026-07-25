"""Base interfaces for memory tiers."""

from abc import ABC, abstractmethod
from typing import Any


class BaseMemory(ABC):
    """Abstract memory contract."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Return health metadata for this memory tier."""

