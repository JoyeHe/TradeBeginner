"""Deterministic backtesting and reward computation tools."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from config.settings import Settings
from schemas.rewards import BacktestResult, RewardSignal
from schemas.strategy import PositionAction, Strategy

logger = structlog.get_logger(__name__)


@dataclass
class TradeOutcome:
    returns: float
    won: bool


def _norm(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


async def run_backtest(
    strategy: Strategy,
    historical_data: dict[str, pd.DataFrame],
    initial_capital: float = 100000.0,
    commission_pct: float = 0.001,
) -> BacktestResult:
    """Run mechanical simulation from strategy against historical data."""
    outcomes: list[TradeOutcome] = []
    partial_data = False
    for pos in strategy.positions:
        df = historical_data.get(pos.asset)
        if df is None or df.empty:
            logger.warning("backtest_missing_data", asset=pos.asset)
            partial_data = True
            continue
        df = df.sort_index()
        if len(df) < 2:
            partial_data = True
            continue
        entry_price = pos.entry_price_target if pos.entry_price_target else float(df.iloc[1]["open"])
        side = 1.0 if pos.action in (PositionAction.LONG, PositionAction.INCREASE) else -1.0
        if pos.action in (PositionAction.CLOSE, PositionAction.REDUCE):
            continue

        stop = entry_price * (1 - (pos.stop_loss_pct / 100.0) * side)
        take = None if pos.take_profit_pct is None else entry_price * (1 + (pos.take_profit_pct / 100.0) * side)
        horizon = min(pos.time_horizon_days, len(df) - 1)

        exit_price = float(df.iloc[horizon]["close"])
        for i in range(1, horizon + 1):
            high = float(df.iloc[i]["high"])
            low = float(df.iloc[i]["low"])
            if take is not None and ((side > 0 and high >= take) or (side < 0 and low <= take)):
                exit_price = take
                break
            if (side > 0 and low <= stop) or (side < 0 and high >= stop):
                exit_price = stop
                break

        gross = side * ((exit_price - entry_price) / entry_price)
        net = gross - commission_pct * 2
        outcomes.append(TradeOutcome(returns=net, won=net > 0))

    returns = np.array([o.returns for o in outcomes], dtype=float) if outcomes else np.array([0.0])
    total_return = float(np.sum(returns))
    vol = float(np.std(returns)) if len(returns) > 1 else 0.0
    sharpe = float(np.mean(returns) / vol) if vol > 1e-9 else 0.0
    cum = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(cum)
    drawdown = ((cum - peaks) / peaks) if len(cum) else np.array([0.0])
    max_dd = abs(float(np.min(drawdown))) if len(drawdown) else 0.0
    win_rate = float(np.mean([o.won for o in outcomes])) if outcomes else 0.0
    profit_factor = None
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses > 0:
        profit_factor = float(profits / losses)

    all_dates = sorted({idx for df in historical_data.values() for idx in df.index})
    bt_start = all_dates[0] if all_dates else datetime.now(timezone.utc)
    bt_end = all_dates[-1] if all_dates else datetime.now(timezone.utc)
    return BacktestResult(
        strategy_id=strategy.strategy_id,
        backtest_start=bt_start,
        backtest_end=bt_end,
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=len(outcomes),
        avg_trade_return=float(np.mean(returns)) if len(returns) else 0.0,
        volatility=vol,
        partial_data=partial_data,
    )


async def compute_reward(result: BacktestResult, settings: Settings) -> RewardSignal:
    """Compute weighted reward signal from backtest metrics."""
    sharpe_term = _norm(result.sharpe_ratio or 0.0, -2.0, 4.0)
    drawdown_term = 1.0 - _norm(result.max_drawdown, 0.0, 0.5)
    win_term = _norm(result.win_rate, 0.0, 1.0)
    reward = (
        settings.reward_weight_sharpe * sharpe_term
        + settings.reward_weight_drawdown * drawdown_term
        + settings.reward_weight_winrate * win_term
    )
    reward = float(max(-1.0, min(1.0, reward)))
    return RewardSignal(
        strategy_id=result.strategy_id,
        terminal_reward=reward,
        component_rewards={"sharpe": sharpe_term, "drawdown_penalty": drawdown_term, "win_rate": win_term},
        backtest_result=result,
    )


async def rolling_window_backtest(
    strategy: Strategy,
    historical_data: dict[str, pd.DataFrame],
    settings: Settings,
) -> RewardSignal:
    """Primary rolling-window mode for orchestrator usage."""
    result = await run_backtest(strategy, historical_data)
    return await compute_reward(result, settings)


async def walk_forward_analysis(
    strategy: Strategy,
    historical_data: dict[str, pd.DataFrame],
    settings: Settings,
    total_lookback_days: int = 120,
    window_size_days: int = 20,
    step_days: int = 10,
) -> dict:
    """Run walk-forward windows over historical data, slicing different sub-periods."""
    all_dates = sorted({idx for df in historical_data.values() for idx in df.index})
    if not all_dates:
        empty = await run_backtest(strategy, historical_data)
        return {"results": [empty.model_dump(mode="json")], "avg_reward": 0.0}

    end_idx = len(all_dates) - 1
    results: list[BacktestResult] = []
    start_offset = 0
    while start_offset + window_size_days <= len(all_dates) and start_offset + window_size_days <= total_lookback_days:
        window_start = all_dates[start_offset]
        window_end_i = min(start_offset + window_size_days, end_idx)
        window_end = all_dates[window_end_i]
        sliced = {
            ticker: df.loc[(df.index >= window_start) & (df.index <= window_end)]
            for ticker, df in historical_data.items()
        }
        res = await run_backtest(strategy, sliced)
        results.append(res)
        start_offset += step_days
    if not results:
        results.append(await run_backtest(strategy, historical_data))
    rewards = [await compute_reward(r, settings) for r in results]
    avg_reward = sum(r.terminal_reward for r in rewards) / max(len(rewards), 1)
    return {"results": [x.model_dump(mode="json") for x in results], "avg_reward": avg_reward}


async def compare_strategies(items: list[tuple[Strategy, BacktestResult]]) -> dict:
    """Compare strategies with simple ranking metrics."""
    ranked = sorted(
        items,
        key=lambda p: ((p[1].sharpe_ratio or -99), p[1].total_return, -p[1].max_drawdown),
        reverse=True,
    )
    return {
        "rankings": [
            {
                "strategy_id": strategy.strategy_id,
                "sharpe": result.sharpe_ratio,
                "total_return": result.total_return,
                "max_drawdown": result.max_drawdown,
            }
            for strategy, result in ranked
        ]
    }

