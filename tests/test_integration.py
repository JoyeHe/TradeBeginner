from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest
import pytest_asyncio

from orchestrator.pipeline import TradingPipeline
from schemas.news import NewsItem, SentimentDigest
from schemas.strategy import MarketRegime


def _diag(function_tested: str, input_value, expected, actual, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


def _strategy_payload(size_pct: float = 10.0, strategy_id: str | None = None) -> str:
    sid = strategy_id or str(uuid4())
    return json.dumps(
        {
            "strategy_id": sid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_regime": "bull",
            "positions": [
                {
                    "asset": "AAPL",
                    "action": "long",
                    "size_pct": size_pct,
                    "entry_price_target": None,
                    "stop_loss_pct": 3.0,
                    "take_profit_pct": 8.0,
                    "time_horizon_days": 10,
                    "confidence": 0.7,
                }
            ],
            "rationale": "Integration test generated strategy.",
            "data_sources_used": ["market_snapshot", "news_digest"],
            "risk_metrics": {"total_exposure_pct": size_pct},
            "metadata": {"integration": True},
        }
    )


class _DummyAgent:
    def __init__(self, *args, **kwargs):
        self.prompts: list[str] = []

    async def arun(self, prompt: str):
        self.prompts.append(prompt)
        # If refinement prompt has explicit request, return slightly smaller size.
        if "feedback" in prompt.lower() or "refinement_target" in prompt.lower():
            return SimpleNamespace(content=_strategy_payload(size_pct=5.0))
        return SimpleNamespace(content=_strategy_payload(size_pct=10.0))


def _fake_market_bars(symbol: str, days: int = 260) -> list[dict]:
    idx = pd.date_range("2025-01-01", periods=days, freq="D")
    base = 100.0
    rows = []
    for i, t in enumerate(idx):
        close = base + i * 0.2 + 2.0 * math.sin(i / 4)
        if symbol == "^VIX":
            close = 16.0
        rows.append(
            {
                "timestamp": t.to_pydatetime(),
                "open": close - 0.2,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000_000,
                "asset": symbol,
            }
        )
    return rows


@pytest_asyncio.fixture
async def integrated_system(monkeypatch: pytest.MonkeyPatch, tmp_path):
    # Avoid real model provider initialization.
    monkeypatch.setattr("agents.agent1_news.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent2_market.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent3_strategy.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent4_executor.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent6_backtest.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent1_news.build_agno_model", lambda s: object())
    monkeypatch.setattr("agents.agent2_market.build_agno_model", lambda s: object())
    monkeypatch.setattr("agents.agent3_strategy.build_agno_model", lambda s: object())
    monkeypatch.setattr("agents.agent4_executor.build_agno_model", lambda s: object())
    monkeypatch.setattr("agents.agent6_backtest.build_agno_model", lambda s: object())

    # External news feed mocked.
    async def fake_fetch_news_feed(settings, tickers=None, lookback_hours=24, max_items=50, rss_urls=None):
        return [
            NewsItem(
                news_id="n-1",
                source="mock",
                title="AAPL surge after earnings beat",
                summary="Positive guidance and strong growth",
                published_at=datetime.now(timezone.utc).replace(tzinfo=None),
                tickers_mentioned=["AAPL"],
            )
        ]

    async def fake_generate_digest(items):
        return SentimentDigest(
            timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
            overall_market_sentiment=-0.8 if any("risk-off" in x.title.lower() for x in items) else 0.6,
            sector_sentiments={"Technology": 0.7},
            top_positive_events=items[:1],
            top_negative_events=[],
            trending_tickers=["AAPL"],
            macro_summary="Mocked digest",
        )

    monkeypatch.setattr("agents.agent1_news.fetch_news_feed", fake_fetch_news_feed)
    monkeypatch.setattr("agents.agent1_news.generate_sentiment_digest", fake_generate_digest)

    # External market data mocked.
    from schemas.market_data import OHLCVBar

    async def fake_fetch_price_data(tickers, start_date, end_date, interval="1d"):
        out = {}
        for t in tickers:
            out[t] = [OHLCVBar(**row) for row in _fake_market_bars(t, days=260)]
        return out

    monkeypatch.setattr("tools.market_tools.fetch_price_data", fake_fetch_price_data)
    monkeypatch.setattr("agents.agent4_executor.fetch_price_data", fake_fetch_price_data)

    from config.settings import Settings

    settings = Settings(
        _env_file=None,
        chroma_persist_dir=str(tmp_path / "chroma"),
        paper_trading=True,
        llm_api_key="test-key",
    )
    pipeline = TradingPipeline(settings)
    await pipeline.initialize()
    try:
        yield pipeline
    finally:
        await pipeline.shutdown()


@pytest.mark.asyncio
async def test_full_cycle_happy_path(integrated_system: TradingPipeline):
    """FUNCTION TESTED: orchestrator.pipeline.TradingPipeline full cycle happy path"""
    generated = await integrated_system.request_strategy()
    sid = generated["strategy"]["strategy_id"]
    await asyncio.sleep(0.05)
    result = await integrated_system.get_strategy_result(sid)
    assert result["status"] in {"ready", "evaluating"}
    approved = await integrated_system.user_approve(sid)
    assert approved["execution"] is not None
    episodes = await integrated_system.memory.episodic.retrieve_recent(5)
    assert len(episodes) >= 1
    stats = await integrated_system.agent5.report()
    assert stats["total_records"] >= 1


@pytest.mark.asyncio
async def test_full_cycle_with_modification(integrated_system: TradingPipeline):
    """FUNCTION TESTED: orchestrator.pipeline.TradingPipeline generate->modify->approve"""
    generated = await integrated_system.request_strategy()
    sid = generated["strategy"]["strategy_id"]
    strategy_data = generated["strategy"]
    strategy_data["positions"][0]["size_pct"] = 5.0
    approved = await integrated_system.user_approve(sid, modifications={"positions": strategy_data["positions"]})
    assert approved["execution"] is not None


@pytest.mark.asyncio
async def test_full_cycle_with_rejection(integrated_system: TradingPipeline):
    """FUNCTION TESTED: orchestrator.pipeline.TradingPipeline generate->reject"""
    generated = await integrated_system.request_strategy()
    sid = generated["strategy"]["strategy_id"]
    out = await integrated_system.user_reject(sid, reason="Too risky")
    assert out["status"] == "rejected"


@pytest.mark.asyncio
async def test_full_cycle_risk_rejection(integrated_system: TradingPipeline):
    """FUNCTION TESTED: orchestrator.pipeline.TradingPipeline risk block path"""
    integrated_system.settings.max_position_size_pct = 1.0
    generated = await integrated_system.request_strategy()
    sid = generated["strategy"]["strategy_id"]
    out = await integrated_system.user_approve(sid)
    assert out["execution"] is None, _diag(
        "orchestrator.pipeline.TradingPipeline.user_approve",
        "max_position_size_pct=1 with generated 10%",
        "execution blocked",
        out,
        "SAFETY_CRITICAL_DEFAULT",
    )


@pytest.mark.asyncio
async def test_full_cycle_with_refinement(integrated_system: TradingPipeline):
    """FUNCTION TESTED: orchestrator.pipeline.TradingPipeline refine flow"""
    generated = await integrated_system.request_strategy()
    sid = generated["strategy"]["strategy_id"]
    refined = await integrated_system.user_request_refinement(sid, feedback="Reduce exposure")
    new_id = refined["strategy"]["strategy_id"]
    assert new_id != sid
    approved = await integrated_system.user_approve(new_id)
    assert approved["execution"] is not None


@pytest.mark.asyncio
async def test_market_data_flows_to_strategy(integrated_system: TradingPipeline):
    """FUNCTION TESTED: Agent2->WorkingMemory->Agent3 data flow"""
    await integrated_system.request_strategy()
    trace = await integrated_system.memory.working.get("rl_trace_log_latest")
    assert trace is not None
    market_snapshot = trace["state_context"]["market_snapshot"]
    assert market_snapshot is not None
    # Ensure computed indicator payload from Agent2 reaches Agent3 context.
    assert "AAPL" in market_snapshot["indicators"]
    assert market_snapshot["indicators"]["AAPL"]["sma_20"] is not None


@pytest.mark.asyncio
async def test_news_sentiment_flows_to_strategy(integrated_system: TradingPipeline):
    """FUNCTION TESTED: Agent1->WorkingMemory->Agent3 data flow"""
    await integrated_system.request_strategy()
    trace = await integrated_system.memory.working.get("rl_trace_log_latest")
    sentiment = trace["state_context"]["news_digest"]["overall_market_sentiment"]
    assert sentiment == 0.6


@pytest.mark.asyncio
async def test_episodic_memory_accumulates(integrated_system: TradingPipeline):
    """FUNCTION TESTED: episodic accumulation across cycles"""
    for _ in range(3):
        generated = await integrated_system.request_strategy()
        await integrated_system.user_approve(generated["strategy"]["strategy_id"])
    episodes = await integrated_system.memory.episodic.retrieve_recent(10)
    assert len(episodes) >= 3, _diag(
        "memory.episodic.EpisodicMemory.retrieve_recent",
        "3 cycles",
        ">=3 episodes",
        len(episodes),
        "DATA_INTEGRITY",
    )


@pytest.mark.asyncio
async def test_portfolio_state_consistency(integrated_system: TradingPipeline):
    """FUNCTION TESTED: portfolio accounting consistency"""
    for _ in range(2):
        generated = await integrated_system.request_strategy()
        await integrated_system.user_approve(generated["strategy"]["strategy_id"])
    portfolio = await integrated_system.get_portfolio()
    pos_value = sum(v["market_value"] for v in portfolio.positions.values())
    assert abs((portfolio.cash + pos_value) - portfolio.total_value) < 1e-6, _diag(
        "tools.execution_tools.PaperTradingEngine.get_portfolio_state",
        "cash + positions",
        portfolio.total_value,
        portfolio.cash + pos_value,
        "WRONG_CALCULATION",
    )


@pytest.mark.asyncio
async def test_rl_trace_completeness(integrated_system: TradingPipeline):
    """FUNCTION TESTED: RL trace completeness in full pipeline"""
    generated = await integrated_system.request_strategy()
    sid = generated["strategy"]["strategy_id"]
    await asyncio.sleep(0.05)
    pending = await integrated_system.get_strategy_result(sid)
    reward = pending.get("reward")
    trace = await integrated_system.memory.working.get("rl_trace_log_latest")
    assert trace["state_context"]
    assert trace["prompt"]
    assert trace["raw_response"]
    assert trace["model_metadata"]
    assert reward is None or "terminal_reward" in reward
