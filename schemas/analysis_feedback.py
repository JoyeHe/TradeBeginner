"""User feedback on market analysis (Flow 2 intuition loop)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional
import uuid

from pydantic import BaseModel, Field


class FeedbackDimension(str, Enum):
    NEWS_INTERPRETATION = "news_interpretation"
    SENTIMENT = "sentiment"
    MARKET_OUTLOOK = "market_outlook"
    AFFECTED_ASSETS = "affected_assets"
    TIME_HORIZON = "time_horizon"
    RISKS = "risks"


class DimensionFeedback(BaseModel):
    dimension: FeedbackDimension
    verdict: Literal["agree", "disagree", "partial"]
    correction: Optional[str] = None
    suggested_value: Optional[dict] = None


class AnalysisFeedback(BaseModel):
    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    analysis_id: str
    user_id: str = "default"
    overall_verdict: Literal["agree", "disagree", "partial"]
    dimension_feedbacks: list[DimensionFeedback] = Field(default_factory=list)
    free_text: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
