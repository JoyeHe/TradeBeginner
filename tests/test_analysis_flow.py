"""Tests for analysis baseline + feedback flows (Flow 1 & Flow 2)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from memory.analysis_store import AnalysisStore
from orchestrator.analysis_flow import AnalysisFlow
from schemas.analysis import FlowType, MarketAnalysis, MarketOutlook, OutlookDirection, SentimentView
from schemas.analysis_feedback import AnalysisFeedback, DimensionFeedback, FeedbackDimension
from schemas.rewards import BacktestResult, RewardSignal
from schemas.strategy import MarketRegime, Position, PositionAction, RiskMetrics, Strategy
from tools.analysis_helpers import apply_feedback_heuristic, heuristic_market_analysis


def _sample_news_bundle(score: float = 0.4) -> dict:
    return {
        "item_count": 2,
        "overall_sentiment": score,
        "llm_summary": "Tech earnings beat expectations.",
        "headlines": [
            {"news_id": "n1", "title": "AAPL beats", "sentiment_score": score, "category": "earnings"},
        ],
        "key_themes": ["earnings"],
    }


def _sample_market_evidence() -> dict:
    return {"tickers": {"AAPL": {"close": 180.0, "rsi_14": 55.0}}, "regime": "bull"}


def _sample_strategy() -> Strategy:
    return Strategy(
        market_regime=MarketRegime.BULL,
        positions=[
            Position(
                asset="AAPL",
                action=PositionAction.LONG,
                size_pct=10.0,
                stop_loss_pct=3.0,
                take_profit_pct=8.0,
                time_horizon_days=10,
                confidence=0.7,
            )
        ],
        rationale="Test strategy",
        risk_metrics=RiskMetrics(total_exposure_pct=10.0),
    )


def _sample_reward() -> RewardSignal:
    bt = BacktestResult(
        strategy_id="s-test",
        backtest_start=datetime.now(timezone.utc),
        backtest_end=datetime.now(timezone.utc),
        total_return=0.08,
        sharpe_ratio=1.1,
        max_drawdown=0.05,
        win_rate=0.6,
        profit_factor=1.3,
        total_trades=5,
        avg_trade_return=0.01,
        volatility=0.12,
    )
    return RewardSignal(strategy_id="s-test", terminal_reward=0.42, backtest_result=bt)


@pytest.fixture
def analysis_flow(memory_manager):
    store = AnalysisStore()
    agent1 = AsyncMock()
    agent1.user_search = AsyncMock(
        return_value={
            "items": [],
            "item_count": 0,
            "digest": {"overall_market_sentiment": 0.35},
            "llm_summary": "Positive tech sentiment",
        }
    )
    agent2 = AsyncMock()
    agent2.user_lookup = AsyncMock(return_value={"data": {"AAPL": {"last_bar": {"close": 180}, "indicators": {"rsi_14": 55}}}})

    agent3_settings = type("S", (), {"llm_api_key": None})()
    agent3 = AsyncMock()
    agent3.settings = agent3_settings
    agent3._agent_run = AsyncMock(return_value=None)
    agent3.gather_context = AsyncMock(return_value={"semantic_knowledge": [], "news_digest": {}})
    agent3.generate_strategy = AsyncMock(side_effect=lambda ctx: _sample_strategy())
    agent3.explain_strategy_and_backtest = AsyncMock(return_value="NL explanation")
    agent3.generate_strategy_from_feedback = AsyncMock(side_effect=lambda **kw: _sample_strategy())
    agent3.compare_strategy_outcomes = AsyncMock(return_value="Compare NL")
    agent3._extract_json = lambda raw: raw

    agent5 = AsyncMock()
    agent5.on_analysis_feedback = AsyncMock(return_value="pref-1")

    agent6 = AsyncMock()
    agent6.evaluate_strategy = AsyncMock(return_value=_sample_reward())

    return AnalysisFlow(agent1, agent2, agent3, agent5, agent6, store, memory_manager)


def test_heuristic_market_analysis_bullish():
    analysis = heuristic_market_analysis(
        "AAPL outlook", _sample_news_bundle(0.4), _sample_market_evidence()
    )
    assert analysis.outlook.direction == OutlookDirection.BULLISH
    assert analysis.flow_type == FlowType.BASELINE
    assert "AAPL" in analysis.outlook.affected_assets


def test_apply_feedback_heuristic_bearish():
    base = heuristic_market_analysis("query", _sample_news_bundle(0.4), _sample_market_evidence())
    revised = apply_feedback_heuristic(base, "I think it will go down, bearish outlook")
    assert revised.flow_type == FlowType.REVISED
    assert revised.outlook.direction == OutlookDirection.BEARISH
    assert revised.analysis_id != base.analysis_id
    assert revised.parent_analysis_id == base.analysis_id


@pytest.mark.asyncio
async def test_run_baseline_flow(analysis_flow):
    session = await analysis_flow.run_baseline("AAPL earnings", tickers=["AAPL"])
    assert session.status == "ready"
    assert session.baseline_analysis is not None
    assert session.baseline_strategy is not None
    assert session.baseline_reward is not None
    assert len(analysis_flow.store.list_library()) == 1

    payload = analysis_flow.session_to_dict(session)
    assert payload["analysis_id"] == session.analysis_id
    assert payload["baseline_analysis"]["outlook"]["direction"] in ("bullish", "bearish", "neutral", "mixed")


@pytest.mark.asyncio
async def test_submit_feedback_flow(analysis_flow):
    session = await analysis_flow.run_baseline("tech sector", tickers=["AAPL"])
    feedback = AnalysisFeedback(
        analysis_id=session.analysis_id,
        overall_verdict="partial",
        dimension_feedbacks=[
            DimensionFeedback(
                dimension=FeedbackDimension.MARKET_OUTLOOK,
                verdict="disagree",
                correction="More bearish near term",
            )
        ],
        free_text="Expect pullback after rally",
    )
    updated = await analysis_flow.submit_feedback(session.analysis_id, feedback)
    assert updated.revised_analysis is not None
    assert updated.feedback_strategy is not None
    assert updated.revised_analysis.flow_type == FlowType.REVISED
    analysis_flow.agent5.on_analysis_feedback.assert_awaited_once()

    compare = await analysis_flow.compare_session(session.analysis_id)
    assert compare["analysis_id"] == session.analysis_id
    assert compare["baseline"]["terminal_reward"] is not None
    assert compare["revised"]["terminal_reward"] is not None
    assert len(analysis_flow.store.list_library()) == 2


@pytest.mark.asyncio
async def test_run_baseline_includes_explanation(analysis_flow):
    analysis_flow.agent3.explain_strategy_and_backtest = AsyncMock(return_value="NL baseline explanation")
    session = await analysis_flow.run_baseline("AAPL earnings", tickers=["AAPL"])
    payload = analysis_flow.session_to_dict(session)
    assert payload["baseline_explanation"] == "NL baseline explanation"


@pytest.mark.asyncio
async def test_submit_feedback_includes_explanations_and_compare(analysis_flow):
    analysis_flow.agent3.explain_strategy_and_backtest = AsyncMock(side_effect=["base NL", "revised NL"])
    analysis_flow.agent3.generate_strategy_from_feedback = AsyncMock(side_effect=lambda **kw: _sample_strategy())
    analysis_flow.agent3.compare_strategy_outcomes = AsyncMock(return_value="Compare NL")
    session = await analysis_flow.run_baseline("tech", tickers=["AAPL"])
    feedback = AnalysisFeedback(
        analysis_id=session.analysis_id,
        user_id="default",
        overall_verdict="partial",
        dimension_feedbacks=[
            DimensionFeedback(
                dimension=FeedbackDimension.MARKET_OUTLOOK,
                verdict="disagree",
                correction="more neutral",
            )
        ],
        free_text="more cautious",
    )
    updated = await analysis_flow.submit_feedback(session.analysis_id, feedback)
    payload = analysis_flow.session_to_dict(updated)
    assert payload["feedback_explanation"]
    assert payload["comparison_narrative"]
    compare = await analysis_flow.compare_session(session.analysis_id)
    assert compare["comparison_narrative"]


@pytest.mark.asyncio
async def test_pipeline_run_analysis_baseline(integrated_pipeline, monkeypatch):
    pipeline = integrated_pipeline

    async def fake_user_search(query):
        return {
            "items": [{"news_id": "x1", "title": "AAPL up", "sentiment_score": 0.3}],
            "item_count": 1,
            "digest": {"overall_market_sentiment": 0.3},
            "llm_summary": "Bullish tone",
        }

    monkeypatch.setattr(pipeline.agent1, "user_search", fake_user_search)

    payload = await pipeline.run_analysis_baseline("AAPL news", tickers=["AAPL"])
    assert payload["status"] == "ready"
    assert payload["baseline_analysis"] is not None
    assert payload["baseline_strategy"] is not None

    session = await pipeline.get_analysis_session(payload["analysis_id"])
    assert session["analysis_id"] == payload["analysis_id"]

    library = await pipeline.list_strategy_library()
    assert len(library) >= 1

    prefs = await pipeline.get_preference_stats()
    assert "analysis_preference_records" in prefs
