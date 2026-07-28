"""Persisted strategy library entries."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid

from pydantic import BaseModel, Field

from schemas.alpha_formula import AlphaFormula
from schemas.strategy import Strategy


class LibraryOrigin(str, Enum):
    SEED = "seed"
    BASELINE = "baseline"
    POST_FEEDBACK = "post_feedback"
    PROMOTED = "promoted"


class StrategySkeleton(BaseModel):
    """Default risk/size hints for a template."""

    default_action: str = "long"
    default_size_pct: float = 10.0
    max_size_pct: float = 12.0
    default_stop_pct: float = 4.0
    default_tp_pct: Optional[float] = 6.0
    default_horizon_days: int = 15
    forbidden_actions: list[str] = Field(default_factory=list)


class StrategyLibraryEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    template_id: Optional[str] = None
    name: Optional[str] = None
    description: str = ""
    user_id: str = "default"
    source_analysis_id: str = ""
    strategy: Optional[Strategy] = None
    skeleton: Optional[StrategySkeleton] = None
    alphas: list[AlphaFormula] = Field(default_factory=list)
    applicable_regimes: list[str] = Field(default_factory=list)
    backtest_reward: Optional[float] = None
    backtest_metrics: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    origin: LibraryOrigin = LibraryOrigin.BASELINE
    quality_score: float = 0.5
    parent_template_id: Optional[str] = None
    use_count: int = 0
    promote_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
