"""Agent 6: Backtesting and reward evaluation."""

from __future__ import annotations

from datetime import date, timedelta

import structlog
from agno.agent import Agent

from agents.common import build_agno_model
from config.settings import Settings
from memory import MemoryManager
from schemas.rewards import RewardSignal
from schemas.strategy import Strategy
from tools.backtest_tools import compute_reward, run_backtest, walk_forward_analysis

logger = structlog.get_logger(__name__)


class Agent6Backtest:
    """Evaluates strategy quality deterministically."""

    def __init__(self, settings: Settings, memory: MemoryManager, agent2):
        self.settings = settings
        self.memory = memory
        self.agent2 = agent2
        self.agent = Agent(
            name="agent6_backtest",
            model=build_agno_model(settings),
            instructions=[
                "Evaluate strategy performance using deterministic backtesting metrics.",
                "Provide objective quantitative reward signals.",
            ],
            markdown=True,
        )

    async def evaluate_strategy(self, strategy: Strategy) -> RewardSignal:
        """Multi-window evaluation; falls back to single-window reward."""
        tickers = sorted({p.asset for p in strategy.positions})
        end = date.today()
        start = end - timedelta(days=max(self.settings.backtest_window_days * 6, 90))
        data = await self.agent2.get_backtest_data(tickers, start, end)
        # Prefer walk-forward average when enough history; still return a RewardSignal
        try:
            wf = await walk_forward_analysis(
                strategy,
                data,
                self.settings,
                total_lookback_days=min(120, max(len(next(iter(data.values()))) if data else 0, 40)),
                window_size_days=min(20, self.settings.backtest_window_days),
                step_days=10,
            )
            # Recompute primary window for structured backtest_result
            primary = await run_backtest(strategy=strategy, historical_data=data)
            reward = await compute_reward(primary, self.settings)
            # Blend toward walk-forward average when available
            avg = float(wf.get("avg_reward") or reward.terminal_reward)
            reward.terminal_reward = float(0.5 * reward.terminal_reward + 0.5 * avg)
            reward.component_rewards = {
                **reward.component_rewards,
                "walk_forward_avg": avg,
            }
        except Exception as exc:
            logger.warning("walk_forward_eval_failed", error=str(exc))
            primary = await run_backtest(strategy=strategy, historical_data=data)
            reward = await compute_reward(primary, self.settings)
        await self.memory.working.set(f"backtest:{strategy.strategy_id}", reward, category="backtest")
        return reward

    async def deep_evaluation(self, strategy: Strategy) -> dict:
        """Run multi-window walk-forward style evaluation."""
        windows = [20, 40, 60]
        signals = []
        for w in windows:
            original = self.settings.backtest_window_days
            self.settings.backtest_window_days = w
            signals.append(await self.evaluate_strategy(strategy))
            self.settings.backtest_window_days = original
        avg_reward = sum(s.terminal_reward for s in signals) / max(len(signals), 1)
        return {"windows": windows, "signals": [s.model_dump(mode="json") for s in signals], "avg_reward": avg_reward}

    async def batch_evaluate(self, strategies: list[Strategy]) -> list[RewardSignal]:
        """Evaluate multiple strategies for RL-style rollouts."""
        out = []
        for strategy in strategies:
            try:
                out.append(await self.evaluate_strategy(strategy))
            except Exception as exc:
                logger.error("batch_backtest_failed", strategy_id=strategy.strategy_id, error=str(exc))
        return out
