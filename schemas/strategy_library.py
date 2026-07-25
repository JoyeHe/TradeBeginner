"""Persisted strategy library entries."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional
import uuid

from pydantic import BaseModel, Field

from schemas.strategy import Strategy


class LibraryOrigin(str, Enum):
    BASELINE = "baseline"
    POST_FEEDBACK = "post_feedback"


class StrategyLibraryEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default"
    source_analysis_id: str
    strategy: Strategy
    backtest_reward: Optional[float] = None
    backtest_metrics: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    origin: LibraryOrigin = LibraryOrigin.BASELINE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
