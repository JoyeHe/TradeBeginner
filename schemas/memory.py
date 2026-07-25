"""Schemas for memory layers."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    memory_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    memory_type: str
    content: Any
    metadata: dict = Field(default_factory=dict)
    ttl_seconds: Optional[int] = None
    importance: float = Field(default=0.5, ge=0, le=1)


class EpisodicTrace(BaseModel):
    trace_id: str
    strategy: Any
    user_modification: Any
    execution_result: Any
    reward: Optional[float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: dict = Field(default_factory=dict)

