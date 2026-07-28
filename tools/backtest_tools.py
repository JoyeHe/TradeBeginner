"""Deterministic backtesting and reward computation tools."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
    size_pct: float = 100.0


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
    """Run mechanical simulation from strategy against historical data.

    Position PnL is weighted by size_pct/100 so allocation changes affect total_return.
    """
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
        outcomes.append(TradeOutcome(returns=net, won=net > 0, size_pct=float(pos.size_pct)))

    if outcomes:
        raw = np.array([o.returns for o in outcomes], dtype=float)
        weights = np.array([o.size_pct / 100.0 for o in outcomes], dtype=float)
        weighted = raw * weights
        total_return = float(np.sum(weighted))
        # Equity path for drawdown: start 1.0, apply weighted period returns sequentially
        equity = np.cumprod(1.0 + weighted)
        peaks = np.maximum.accumulate(equity)
        drawdown = (equity - peaks) / peaks
        max_dd = abs(float(np.min(drawdown))) if len(drawdown) else 0.0
        # Floor: a single losing trade must show drawdown at least |loss|
        loss_floor = abs(float(np.min(np.minimum(weighted, 0.0)))) if len(weighted) else 0.0
        max_dd = max(max_dd, loss_floor)
        win_rate = float(np.mean([o.won for o in outcomes]))
        vol = float(np.std(weighted)) if len(weighted) > 1 else 0.0
        avg_trade = float(np.mean(weighted))
        profits = float(weighted[weighted > 0].sum())
        losses = abs(float(weighted[weighted < 0].sum()))
        profit_factor = float(profits / losses) if losses > 0 else None
        n = len(outcomes)
        if n >= 2 and vol > 1e-9:
            sharpe = float(np.mean(weighted) / vol)
            sharpe = float(max(-4.0, min(4.0, sharpe)))
        elif n < 2:
            sharpe = None
        else:
            sharpe = 0.0
    else:
        total_return = 0.0
        max_dd = 0.0
        win_rate = 0.0
        vol = 0.0
        avg_trade = 0.0
        profit_factor = None
        sharpe = None
        n = 0

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
        total_trades=n,
        avg_trade_return=avg_trade,
        volatility=vol,
        partial_data=partial_data,
    )


async def compute_reward(result: BacktestResult, settings: Settings) -> RewardSignal:
    """Compute weighted reward signal from backtest metrics."""
    min_trades = getattr(settings, "min_trades_for_sharpe", 5)
    low_sample = result.total_trades < min_trades
    if result.sharpe_ratio is None or low_sample:
        sharpe_term = 0.5  # neutral — avoid degenerate 0.467 lock-in from sharpe=0 + dd=0
    else:
        clipped = max(-2.0, min(4.0, float(result.sharpe_ratio)))
        sharpe_term = _norm(clipped, -2.0, 4.0)
    drawdown_term = 1.0 - _norm(result.max_drawdown, 0.0, 0.5)
    win_term = _norm(result.win_rate, 0.0, 1.0)
    reward = (
        settings.reward_weight_sharpe * sharpe_term
        + settings.reward_weight_drawdown * drawdown_term
        + settings.reward_weight_winrate * win_term
    )
    if low_sample:
        reward *= 0.85  # down-weight unreliable samples
    reward = float(max(-1.0, min(1.0, reward)))
    return RewardSignal(
        strategy_id=result.strategy_id,
        terminal_reward=reward,
        component_rewards={
            "sharpe": sharpe_term,
            "drawdown_penalty": drawdown_term,
            "win_rate": win_term,
            "low_sample": 1.0 if low_sample else 0.0,
        },
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
        key=lambda p: ((p[1].sharpe_ratio if p[1].sharpe_ratio is not None else -99), p[1].total_return, -p[1].max_drawdown),
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
