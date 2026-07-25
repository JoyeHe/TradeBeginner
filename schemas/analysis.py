"""Market analysis schemas for news-driven sentiment and outlook."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional
import uuid

from pydantic import BaseModel, Field


class OutlookDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class FlowType(str, Enum):
    BASELINE = "baseline"
    REVISED = "revised"


class SentimentView(BaseModel):
    overall_score: float = Field(..., ge=-1.0, le=1.0)
    sector_sentiments: dict[str, float] = Field(default_factory=dict)
    key_themes: list[str] = Field(default_factory=list)
    supporting_headline_ids: list[str] = Field(default_factory=list)


class MarketOutlook(BaseModel):
    direction: OutlookDirection
    horizon_days: int = Field(..., ge=1, le=365)
    confidence: float = Field(..., ge=0.0, le=1.0)
    affected_assets: list[str] = Field(default_factory=list)
    narrative: str = ""


class MarketAnalysis(BaseModel):
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str = "default"
    query_context: str = ""
    news_bundle: dict = Field(default_factory=dict)
    market_evidence: dict = Field(default_factory=dict)
    sentiment: SentimentView
    outlook: MarketOutlook
    risks: list[str] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)
    flow_type: FlowType = FlowType.BASELINE
    parent_analysis_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
