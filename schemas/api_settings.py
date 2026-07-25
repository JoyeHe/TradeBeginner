"""Validated API payloads for settings endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RiskSettingsUpdate(BaseModel):
    max_position_size_pct: float | None = Field(default=None, ge=0.1, le=100.0)
    max_total_exposure_pct: float | None = Field(default=None, ge=1.0, le=500.0)
    max_stop_loss_pct: float | None = Field(default=None, ge=0.1, le=50.0)
    max_concurrent_positions: int | None = Field(default=None, ge=1, le=100)
    max_sector_concentration_pct: float | None = Field(default=None, ge=1.0, le=100.0)
    max_portfolio_drawdown_pct: float | None = Field(default=None, ge=1.0, le=100.0)
    min_avg_volume: int | None = Field(default=None, ge=0, le=1_000_000_000)
    reward_weight_sharpe: float | None = Field(default=None, ge=0.0, le=1.0)
    reward_weight_drawdown: float | None = Field(default=None, ge=0.0, le=1.0)
    reward_weight_winrate: float | None = Field(default=None, ge=0.0, le=1.0)


class WatchlistUpdate(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=50)
