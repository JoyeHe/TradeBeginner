"""Schemas for market data and technical indicator outputs."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class OHLCVBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    asset: str


class TechnicalIndicators(BaseModel):
    asset: str
    timestamp: datetime
    rsi_14: Optional[float] = None
    macd: Optional[dict] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    # Rate-of-change momentum: (close / close.shift(N) - 1) * 100
    momentum_roc_10: Optional[float] = None
    momentum_roc_20: Optional[float] = None
    bollinger_bands: Optional[dict] = None
    atr_14: Optional[float] = None
    volume_sma_20: Optional[float] = None
    obv: Optional[float] = None
    adx: Optional[float] = None


class MarketSnapshot(BaseModel):
    timestamp: datetime
    assets: dict[str, OHLCVBar]
    indicators: dict[str, TechnicalIndicators]
    market_breadth: Optional[dict] = None
    vix: Optional[float] = None
    sector_performance: Optional[dict[str, float]] = None

