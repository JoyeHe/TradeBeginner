"""Schemas for financial news and sentiment digests."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    news_id: str
    source: str
    title: str
    summary: str
    full_text: Optional[str] = None
    url: Optional[str] = None
    published_at: datetime
    tickers_mentioned: list[str] = Field(default_factory=list)
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1)
    relevance_score: Optional[float] = Field(None, ge=0, le=1)
    category: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class SentimentDigest(BaseModel):
    timestamp: datetime
    overall_market_sentiment: float = Field(..., ge=-1, le=1)
    sector_sentiments: dict[str, float] = Field(default_factory=dict)
    top_positive_events: list[NewsItem] = Field(default_factory=list)
    top_negative_events: list[NewsItem] = Field(default_factory=list)
    trending_tickers: list[str] = Field(default_factory=list)
    macro_summary: str = ""

