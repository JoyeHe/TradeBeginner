"""Backtest output and reward signal schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class BacktestResult(BaseModel):
    strategy_id: str
    backtest_start: datetime
    backtest_end: datetime
    total_return: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: float
    win_rate: float
    profit_factor: Optional[float] = None
    total_trades: int
    avg_trade_return: float
    volatility: float
    partial_data: bool = False


class RewardSignal(BaseModel):
    strategy_id: str
    terminal_reward: float
    component_rewards: dict[str, float] = Field(default_factory=dict)
    backtest_result: BacktestResult

