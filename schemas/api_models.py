"""Request/response models for API validation."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RiskSettingsUpdate(BaseModel):
    """Validated risk settings payload for PUT /settings/risk."""

    max_position_size_pct: Optional[float] = Field(default=None, ge=0.1, le=100.0)
    max_total_exposure_pct: Optional[float] = Field(default=None, ge=1.0, le=500.0)
    max_stop_loss_pct: Optional[float] = Field(default=None, ge=0.1, le=50.0)
    max_concurrent_positions: Optional[int] = Field(default=None, ge=1, le=100)
    max_sector_concentration_pct: Optional[float] = Field(default=None, ge=1.0, le=100.0)
    max_portfolio_drawdown_pct: Optional[float] = Field(default=None, ge=1.0, le=100.0)
    min_avg_volume: Optional[int] = Field(default=None, ge=0, le=1_000_000_000)
    reward_weight_sharpe: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reward_weight_drawdown: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reward_weight_winrate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
