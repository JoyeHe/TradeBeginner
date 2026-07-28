"""Main trading pipeline orchestrator."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog

from agents.agent1_news import Agent1News, news_monitoring_loop
from agents.agent2_market import Agent2Market, market_monitoring_loop
from agents.agent3_strategy import Agent3Strategy
from agents.agent4_executor import Agent4Executor
from agents.agent5_behavior import Agent5BehaviorStub
from agents.agent6_backtest import Agent6Backtest
from config.settings import Settings
from memory import MemoryManager
from memory.analysis_store import AnalysisStore
from memory.strategy_library_store import StrategyLibraryStore
from orchestrator.analysis_flow import AnalysisFlow
from risk.controller import RiskController
from schemas.execution import PortfolioState
from schemas.analysis_feedback import AnalysisFeedback
from schemas.risk import RiskAssessment, RiskCheckResult
from schemas.strategy import Strategy, UserModifiedStrategy

logger = structlog.get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ts_age(obj) -> timedelta:
    """Safely get age of an object that may have a .timestamp attribute."""
    ts = getattr(obj, "timestamp", None)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return _utcnow() - ts.astimezone(timezone.utc)
    return timedelta(hours=24)


def _asset_volumes_from_snapshot(market_snapshot) -> dict[str, int]:
    """Extract latest bar volumes from a market snapshot for risk checks."""
    volumes: dict[str, int] = {}
    if market_snapshot is None:
        return volumes
    assets = getattr(market_snapshot, "assets", None)
    if assets is None and isinstance(market_snapshot, dict):
        assets = market_snapshot.get("assets")
    if not isinstance(assets, dict):
        return volumes
    for ticker, payload in assets.items():
        vol = getattr(payload, "volume", None)
        if vol is None and isinstance(payload, dict):
            vol = payload.get("volume") or (payload.get("last_bar") or {}).get("volume")
        if vol is not None:
            volumes[ticker] = int(vol)
    return volumes


class TradingPipeline:
    """Coordinates all agents and user interaction flow."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.memory = MemoryManager(settings)
        self.agent1 = Agent1News(settings, self.memory)
        self.agent2 = Agent2Market(settings, self.memory)
        self.agent3 = Agent3Strategy(settings, self.memory, agent1=self.agent1, agent2=self.agent2)
        self.agent4 = Agent4Executor(settings, self.memory)
        self.agent5 = Agent5BehaviorStub(self.memory)
        self.agent6 = Agent6Backtest(settings, self.memory, agent2=self.agent2)
        self.risk = RiskController(settings)
        self.strategy_library = StrategyLibraryStore(
            postgres_url=settings.postgres_url,
            seed_path=settings.strategy_library_seed_path,
            learned_path=settings.strategy_library_learned_path,
        )
        self.analysis_store = AnalysisStore(library=self.strategy_library)
        self.agent3.library_store = self.strategy_library
        self.analysis_flow = AnalysisFlow(
            self.agent1,
            self.agent2,
            self.agent3,
            self.agent5,
            self.agent6,
            self.analysis_store,
            self.memory,
        )
        self._tasks: list[asyncio.Task] = []
        self._eval_tasks: dict[str, asyncio.Task] = {}
        self._watchlist = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "BTC-USD", "ETH-USD"]
        self._pending: dict[str, dict] = {}

    async def initialize(self) -> None:
        """Initialize memory and bootstrap data."""
        await self.memory.initialize()
        await self.strategy_library.initialize()
        await self.agent1.run_news_cycle()
        await self.agent2.run_market_update(self._watchlist)
        portfolio = await self.agent4.get_current_portfolio()
        await self.memory.working.set("portfolio_state", portfolio, category="execution")

    async def shutdown(self) -> None:
        """Stop background tasks and close resources."""
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        for task in list(self._eval_tasks.values()):
            task.cancel()
        await self.strategy_library.shutdown()
        await self.memory.shutdown()

    async def start_background_tasks(self) -> None:
        """Start periodic monitoring loops."""
        self._tasks.append(asyncio.create_task(news_monitoring_loop(self.agent1, interval_minutes=15)))
        self._tasks.append(asyncio.create_task(market_monitoring_loop(self.agent2, self._watchlist, interval_minutes=5)))
        self._tasks.append(asyncio.create_task(self._order_monitor_loop()))

    async def _order_monitor_loop(self) -> None:
        while True:
            try:
                await self.agent4.monitor_open_orders()
            except Exception as exc:
                logger.error("order_monitor_loop_error", error=str(exc))
            await asyncio.sleep(60)

    async def _ensure_fresh_context(self) -> None:
        digest = await self.memory.working.get("news_digest")
        if digest is None or _ts_age(digest) > timedelta(minutes=30):
            await self.agent1.run_news_cycle()
        snapshot = await self.memory.working.get("market_snapshot")
        if snapshot is None or _ts_age(snapshot) > timedelta(minutes=10):
            await self.agent2.run_market_update(self._watchlist)
        await self.memory.working.set("portfolio_state", await self.agent4.get_current_portfolio(), category="execution")

    def _schedule_evaluation(self, strategy: Strategy) -> None:
        """Run backtest evaluation in background with safe pending updates."""
        sid = strategy.strategy_id

        async def _run() -> None:
            try:
                reward = await self.agent6.evaluate_strategy(strategy)
                payload = self._pending.get(sid)
                if payload is not None:
                    payload["reward"] = reward
            except Exception as exc:
                logger.error("strategy_eval_failed", strategy_id=sid, error=str(exc))

        task = asyncio.create_task(_run())
        self._eval_tasks[sid] = task

        def _done(t: asyncio.Task) -> None:
            self._eval_tasks.pop(sid, None)
            if not t.cancelled() and t.exception() is not None:
                logger.error("strategy_eval_task_error", strategy_id=sid, error=str(t.exception()))

        task.add_done_callback(_done)

    async def request_strategy(self) -> dict:
        """Trigger strategy generation and asynchronous evaluation."""
        await self._ensure_fresh_context()
        strategy = await self.agent3.generate()
        self._pending[strategy.strategy_id] = {"strategy": strategy, "reward": None}
        self._schedule_evaluation(strategy)
        return {"strategy": strategy.model_dump(mode="json"), "status": "generated"}

    async def get_strategy_result(self, strategy_id: str) -> dict:
        """Get strategy and evaluation status."""
        payload = self._pending.get(strategy_id)
        if payload is None:
            return {"status": "not_found"}
        reward = payload.get("reward")
        return {
            "status": "ready" if reward is not None else "evaluating",
            "strategy": payload["strategy"].model_dump(mode="json"),
            "reward": None if reward is None else reward.model_dump(mode="json"),
        }

    async def user_approve(self, strategy_id: str, modifications: Optional[dict] = None) -> dict:
        """Approve a strategy, run risk checks, then execute."""
        payload = self._pending.get(strategy_id)
        if payload is None:
            return {"error": "strategy_not_found"}
        original: Strategy = payload["strategy"]
        if modifications:
            merged = original.model_dump(mode="json")
            merged.update(modifications)
            final = Strategy.model_validate(merged)
        else:
            final = original
        portfolio = await self.agent4.get_current_portfolio()
        market_snapshot = await self.memory.working.get("market_snapshot")
        volumes = _asset_volumes_from_snapshot(market_snapshot)
        risk_assessment: RiskAssessment = await self.risk.assess(final, portfolio, asset_volumes=volumes)
        if risk_assessment.result == RiskCheckResult.FAIL:
            return {"risk_check": risk_assessment.model_dump(mode="json"), "execution": None}
        exec_strategy = risk_assessment.adjusted_strategy if risk_assessment.adjusted_strategy else final
        executed = await self.agent4.execute_strategy(exec_strategy, risk_assessment)
        await self.agent5.on_strategy_approved(original=original, final=final)
        reward = payload.get("reward")
        user_mod = None
        if modifications:
            user_mod = UserModifiedStrategy(original_strategy=original, modified_strategy=final, approved=True)
        await self.memory.record_episode(
            strategy=exec_strategy,
            user_modification=user_mod,
            execution_result=[o.model_dump(mode="json") for o in executed],
            reward=None if reward is None else reward.terminal_reward,
            context={"market_regime": final.market_regime.value},
        )
        self._pending.pop(strategy_id, None)
        return {"risk_check": risk_assessment.model_dump(mode="json"), "execution": [x.model_dump(mode="json") for x in executed]}

    async def user_reject(self, strategy_id: str, reason: Optional[str] = None) -> dict:
        """Reject a generated strategy."""
        payload = self._pending.get(strategy_id)
        if payload is None:
            return {"error": "strategy_not_found"}
        await self.agent5.on_strategy_rejected(payload["strategy"], reason=reason)
        self._pending.pop(strategy_id, None)
        return {"status": "rejected"}

    async def user_request_refinement(self, strategy_id: str, feedback: str) -> dict:
        """Request strategy refinement with feedback."""
        payload = self._pending.get(strategy_id)
        if payload is None:
            return {"error": "strategy_not_found"}
        refined = await self.agent3.refine_strategy(payload["strategy"], feedback=feedback)
        self._pending[refined.strategy_id] = {"strategy": refined, "reward": None}
        self._schedule_evaluation(refined)
        return {"strategy": refined.model_dump(mode="json"), "status": "generated"}

    async def get_portfolio(self) -> PortfolioState:
        """Get current portfolio state."""
        return await self.agent4.get_current_portfolio()

    async def get_system_status(self) -> dict:
        """Return health and freshness status."""
        news = await self.memory.working.get("news_digest")
        market = await self.memory.working.get("market_snapshot")
        episodic_ok = self.memory.episodic._session_factory is not None
        return {
            "news_fresh": news is not None and _ts_age(news) < timedelta(minutes=30),
            "market_fresh": market is not None and _ts_age(market) < timedelta(minutes=10),
            "open_orders": len(self.agent4.open_orders),
            "pending_strategies": len(self._pending),
            "paper_trading": self.settings.paper_trading,
            "episodic_db_connected": episodic_ok,
            "watchlist": self._watchlist,
        }

    async def update_watchlist(self, tickers: list[str]) -> None:
        """Update market watchlist."""
        self._watchlist = tickers

    async def user_news_search(self, query: str) -> dict:
        """User-initiated news search — delegates to Agent 1."""
        return await self.agent1.user_search(query)

    async def user_market_search(
        self,
        query: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> dict:
        """User-initiated market data lookup — delegates to Agent 2."""
        return await self.agent2.user_lookup(query, start_date=start_date, end_date=end_date, interval=interval)

    async def user_directed_strategy(self, user_request: str) -> dict:
        """Generate a strategy from a free-form user instruction, with LLM summary."""
        await self._ensure_fresh_context()
        strategy = await self.agent3.generate_from_user_request(user_request)
        eval_result = await self.agent3.self_evaluate_strategy(strategy)
        strategy.metadata["self_evaluation"] = eval_result
        strategy.metadata["user_request"] = user_request
        summary = await self.agent3.generate_strategy_summary(strategy, user_request=user_request)
        await self.memory.working.set("active_strategy", strategy, category="strategy")
        self._pending[strategy.strategy_id] = {"strategy": strategy, "reward": None}
        self._schedule_evaluation(strategy)
        return {"strategy": strategy.model_dump(mode="json"), "status": "generated", "summary": summary}

    async def request_strategy_with_summary(self) -> dict:
        """Auto-generate strategy and include an LLM summary."""
        result = await self.request_strategy()
        try:
            strategy = Strategy.model_validate(result["strategy"])
            result["summary"] = await self.agent3.generate_strategy_summary(strategy)
        except Exception:
            result["summary"] = result["strategy"].get("rationale", "")
        return result

    async def run_analysis_baseline(self, query: str, tickers: Optional[list[str]] = None, user_id: str = "default") -> dict:
        """Flow 1: news + market → analysis → strategy → backtest."""
        await self._ensure_fresh_context()
        session = await self.analysis_flow.run_baseline(query, tickers=tickers, user_id=user_id)
        payload = self.analysis_flow.session_to_dict(session)
        if session.baseline_strategy:
            self._pending[session.baseline_strategy.strategy_id] = {
                "strategy": session.baseline_strategy,
                "reward": session.baseline_reward,
                "analysis_id": session.analysis_id,
            }
        return payload

    async def submit_analysis_feedback(self, analysis_id: str, feedback: AnalysisFeedback) -> dict:
        """Flow 2: user feedback → revised analysis → strategy → backtest."""
        session = await self.analysis_flow.submit_feedback(analysis_id, feedback)
        payload = self.analysis_flow.session_to_dict(session)
        if session.feedback_strategy:
            self._pending[session.feedback_strategy.strategy_id] = {
                "strategy": session.feedback_strategy,
                "reward": session.feedback_reward,
                "analysis_id": session.analysis_id,
            }
        return payload

    async def get_analysis_session(self, analysis_id: str) -> dict:
        return self.analysis_flow.get_session_payload(analysis_id)

    async def compare_analysis_session(self, analysis_id: str) -> dict:
        return await self.analysis_flow.compare_session(analysis_id)

    async def list_strategy_library(
        self,
        user_id: str = "default",
        limit: int = 50,
        tag: str | None = None,
        min_reward: float | None = None,
    ) -> list[dict]:
        entries = self.analysis_store.list_library(
            user_id=user_id, limit=limit, tag=tag, min_reward=min_reward, include_global=True
        )
        return [e.model_dump(mode="json") for e in entries]

    async def get_strategy_library_entry(self, entry_id: str) -> dict:
        entry = self.strategy_library.get(entry_id)
        if entry is None:
            return {"status": "not_found"}
        return entry.model_dump(mode="json")

    async def search_strategy_library(self, query: str, user_id: str = "default", limit: int = 20) -> list[dict]:
        return [e.model_dump(mode="json") for e in self.strategy_library.search(query, user_id=user_id, limit=limit)]

    async def promote_strategy_to_library(self, entry: dict, force: bool = False) -> dict:
        from schemas.strategy_library import StrategyLibraryEntry

        parsed = StrategyLibraryEntry.model_validate(entry)
        if not force:
            reward = parsed.backtest_reward
            trades = (parsed.backtest_metrics or {}).get("backtest_result", {}).get("total_trades")
            if trades is None:
                trades = (parsed.backtest_metrics or {}).get("total_trades", 0)
            partial = (parsed.backtest_metrics or {}).get("backtest_result", {}).get("partial_data")
            if partial is None:
                partial = (parsed.backtest_metrics or {}).get("partial_data", False)
            if partial or (reward is None) or reward < self.settings.library_min_reward or int(trades or 0) < self.settings.library_min_trades:
                return {"status": "rejected", "reason": "promote_gate_failed", "entry": parsed.model_dump(mode="json")}
        saved = await self.analysis_store.promote_library_entry(parsed)
        return {"status": "ok", "entry": saved.model_dump(mode="json")}

    async def get_preference_stats(self, user_id: str = "default") -> dict:
        base = await self.agent5.report()
        store_stats = self.analysis_store.preference_stats(user_id=user_id)
        return {**base, **store_stats, "seed_library_count": sum(1 for e in self.strategy_library.list_entries(limit=500) if e.user_id == "*")}
