from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents.agent3_strategy import Agent3Strategy
from config.settings import Settings
from memory.working import WorkingMemory
from schemas.rewards import BacktestResult, RewardSignal
from schemas.strategy import MarketRegime, Position, PositionAction, RiskMetrics, Strategy


def _diag(function_tested: str, input_value, expected, actual, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


class _DummyLLMAgent:
    def __init__(self, *args, **kwargs):
        self.responses = []
        self.prompts = []

    async def arun(self, prompt: str):
        self.prompts.append(prompt)
        content = self.responses.pop(0) if self.responses else ""
        return SimpleNamespace(content=content)


class _MemoryStub:
    def __init__(self, context_payload: dict):
        self._context_payload = context_payload
        self.working = WorkingMemory(max_items=100)

    async def get_strategy_context(self):
        return self._context_payload


def _strategy_json(valid_strategy: Strategy, **updates) -> str:
    data = valid_strategy.model_dump(mode="json")
    data.update(updates)
    return json.dumps(data)


@pytest.fixture
def base_context(valid_market_snapshot, valid_portfolio_state):
    return {
        "working_memory": {
            "market_snapshot": valid_market_snapshot,
            "news_digest": {"overall_market_sentiment": 0.2, "timestamp": datetime.utcnow().isoformat()},
            "portfolio_state": valid_portfolio_state,
        },
        "recent_episodes": [{"trace_id": "e1", "context": {"market_regime": "bull"}}],
        "semantic_knowledge": [{"content": "In bull markets favor momentum with controlled stop losses."}],
    }


def _build_agent(monkeypatch: pytest.MonkeyPatch, memory: _MemoryStub) -> tuple[Agent3Strategy, _DummyLLMAgent]:
    monkeypatch.setattr("agents.agent3_strategy.Agent", _DummyLLMAgent)
    monkeypatch.setattr("agents.agent3_strategy.build_agno_model", lambda s: object())
    agent = Agent3Strategy(Settings(_env_file=None, llm_api_key="test-key"), memory)
    llm = agent.agent
    return agent, llm


@pytest.mark.asyncio
async def test_gather_context_all_sources(monkeypatch: pytest.MonkeyPatch, base_context):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy.gather_context"""
    memory = _MemoryStub(base_context)
    agent, _ = _build_agent(monkeypatch, memory)
    context = await agent.gather_context()
    assert set(context.keys()) == {
        "market_snapshot",
        "news_digest",
        "portfolio_state",
        "recent_episodes",
        "semantic_knowledge",
    }
    assert context["market_snapshot"] is not None
    assert context["recent_episodes"]
    assert context["semantic_knowledge"]


@pytest.mark.asyncio
async def test_gather_context_stale_data_documented(monkeypatch: pytest.MonkeyPatch, base_context):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy.gather_context"""
    base_context["working_memory"]["news_digest"]["timestamp"] = (datetime.utcnow() - timedelta(hours=1)).isoformat()
    memory = _MemoryStub(base_context)
    agent, _ = _build_agent(monkeypatch, memory)
    context = await agent.gather_context()
    assert context["news_digest"]["timestamp"] is not None


@pytest.mark.asyncio
async def test_gather_context_empty_memory(monkeypatch: pytest.MonkeyPatch):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy.gather_context"""
    memory = _MemoryStub({"working_memory": {}, "recent_episodes": [], "semantic_knowledge": []})
    agent, _ = _build_agent(monkeypatch, memory)
    context = await agent.gather_context()
    assert context["market_snapshot"] is None
    assert context["recent_episodes"] == []


@pytest.mark.asyncio
async def test_parse_perfect_json_response(monkeypatch: pytest.MonkeyPatch, base_context, valid_strategy: Strategy):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy.generate_strategy"""
    memory = _MemoryStub(base_context)
    agent, llm = _build_agent(monkeypatch, memory)
    llm.responses = [_strategy_json(valid_strategy)]
    out = await agent.generate_strategy(await agent.gather_context())
    assert isinstance(out, Strategy)
    assert out.strategy_id == valid_strategy.strategy_id


@pytest.mark.asyncio
async def test_parse_json_in_markdown_block(monkeypatch: pytest.MonkeyPatch, base_context, valid_strategy: Strategy):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy._extract_json"""
    memory = _MemoryStub(base_context)
    agent, llm = _build_agent(monkeypatch, memory)
    llm.responses = [f"```json\n{_strategy_json(valid_strategy)}\n```"]
    out = await agent.generate_strategy(await agent.gather_context())
    assert out.strategy_id == valid_strategy.strategy_id


@pytest.mark.asyncio
async def test_parse_partial_json_triggers_retry(monkeypatch: pytest.MonkeyPatch, base_context, valid_strategy: Strategy):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy.generate_strategy retry"""
    memory = _MemoryStub(base_context)
    agent, llm = _build_agent(monkeypatch, memory)
    llm.responses = ['{"strategy_id":"broken"', _strategy_json(valid_strategy)]
    out = await agent.generate_strategy(await agent.gather_context())
    assert isinstance(out, Strategy)
    assert len(llm.prompts) == 2, _diag(
        "agents.agent3_strategy.Agent3Strategy.generate_strategy",
        "partial json then valid json",
        "2 LLM calls",
        len(llm.prompts),
        "MISSING_ERROR_HANDLING",
    )


@pytest.mark.asyncio
async def test_parse_missing_fields_triggers_retry(monkeypatch: pytest.MonkeyPatch, base_context, valid_strategy: Strategy):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy.generate_strategy schema repair"""
    memory = _MemoryStub(base_context)
    agent, llm = _build_agent(monkeypatch, memory)
    bad = valid_strategy.model_dump(mode="json")
    bad.pop("risk_metrics", None)
    llm.responses = [json.dumps(bad), _strategy_json(valid_strategy)]
    out = await agent.generate_strategy(await agent.gather_context())
    assert isinstance(out, Strategy)
    assert len(llm.prompts) == 2


@pytest.mark.asyncio
async def test_parse_invalid_values_falls_back(monkeypatch: pytest.MonkeyPatch, base_context, valid_strategy: Strategy):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy.generate_strategy invalid values"""
    memory = _MemoryStub(base_context)
    agent, llm = _build_agent(monkeypatch, memory)
    bad = valid_strategy.model_dump(mode="json")
    bad["positions"][0]["size_pct"] = 150
    bad["positions"][0]["confidence"] = 2.0
    bad["positions"][0]["stop_loss_pct"] = -5
    llm.responses = [json.dumps(bad), json.dumps(bad), json.dumps(bad)]
    out = await agent.generate_strategy(await agent.gather_context())
    assert out.metadata.get("fallback") is True, _diag(
        "agents.agent3_strategy.Agent3Strategy.generate_strategy",
        "invalid constrained values",
        "fallback strategy after retries",
        out.metadata,
        "CONSTRAINT_NOT_ENFORCED",
    )


@pytest.mark.asyncio
async def test_parse_non_json_response_max_retries(monkeypatch: pytest.MonkeyPatch, base_context):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy.generate_strategy non-json path"""
    memory = _MemoryStub(base_context)
    agent, llm = _build_agent(monkeypatch, memory)
    llm.responses = ["I think you should buy AAPL"] * 3
    out = await agent.generate_strategy(await agent.gather_context())
    assert out.metadata.get("fallback") is True
    assert len(llm.prompts) == 3


@pytest.mark.asyncio
async def test_parse_empty_and_refusal_response(monkeypatch: pytest.MonkeyPatch, base_context):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy.generate_strategy empty/refusal handling"""
    memory = _MemoryStub(base_context)
    agent, llm = _build_agent(monkeypatch, memory)
    llm.responses = ["", "I cannot provide financial advice.", ""]
    out = await agent.generate_strategy(await agent.gather_context())
    assert out.metadata.get("fallback") is True


@pytest.mark.asyncio
async def test_rl_trace_logging(monkeypatch: pytest.MonkeyPatch, base_context, valid_strategy: Strategy):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy._log_rl_trace"""
    memory = _MemoryStub(base_context)
    agent, llm = _build_agent(monkeypatch, memory)
    llm.responses = [_strategy_json(valid_strategy)]
    strategy = await agent.generate()
    trace = await memory.working.get("rl_trace_log_latest")
    assert strategy.strategy_id == valid_strategy.strategy_id
    assert trace is not None
    assert trace["prompt"]
    assert trace["raw_response"]
    assert trace["state_context"]
    assert trace["model_metadata"]["model"] == agent.settings.llm_model


@pytest.mark.asyncio
async def test_prompt_deterministic_from_state(monkeypatch: pytest.MonkeyPatch, base_context):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy._make_prompt"""
    memory = _MemoryStub(base_context)
    agent, _ = _build_agent(monkeypatch, memory)
    context1 = await agent.gather_context()
    context2 = await agent.gather_context()
    p1 = agent._make_prompt(context1)
    p2 = agent._make_prompt(context2)
    assert p1 == p2, _diag(
        "agents.agent3_strategy.Agent3Strategy._make_prompt",
        "same state twice",
        "byte-identical prompts",
        "different prompts",
        "STATE_LEAKAGE",
    )


@pytest.mark.asyncio
async def test_strategy_written_to_working_memory(monkeypatch: pytest.MonkeyPatch, base_context, valid_strategy: Strategy):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy.generate"""
    memory = _MemoryStub(base_context)
    agent, llm = _build_agent(monkeypatch, memory)
    llm.responses = [_strategy_json(valid_strategy)]
    out = await agent.generate()
    active = await memory.working.get("active_strategy")
    assert active is not None
    assert active.strategy_id == out.strategy_id


@pytest.mark.asyncio
async def test_explain_strategy_and_backtest(monkeypatch: pytest.MonkeyPatch):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy.explain_strategy_and_backtest"""
    memory = _MemoryStub({"working_memory": {}, "recent_episodes": [], "semantic_knowledge": []})
    agent, _ = _build_agent(monkeypatch, memory)
    monkeypatch.setattr(
        "agents.common.llm_chat",
        AsyncMock(return_value="This long strategy uses AAPL with a controlled risk profile."),
    )
    strategy = Strategy(
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
        rationale="earnings momentum",
        risk_metrics=RiskMetrics(total_exposure_pct=10.0),
    )
    reward = RewardSignal(
        strategy_id=strategy.strategy_id,
        terminal_reward=0.42,
        backtest_result=BacktestResult(
            strategy_id=strategy.strategy_id,
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
        ),
    )
    text = await agent.explain_strategy_and_backtest(strategy, reward, query="AAPL outlook")
    assert "AAPL" in text or "risk" in text.lower() or len(text) > 20


@pytest.mark.asyncio
async def test_generate_strategy_from_feedback_includes_fusion(monkeypatch: pytest.MonkeyPatch):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy.generate_strategy_from_feedback"""
    memory = _MemoryStub({"working_memory": {}, "recent_episodes": [], "semantic_knowledge": []})
    agent, _ = _build_agent(monkeypatch, memory)
    captured: dict = {}

    async def fake_generate(context):
        captured["context"] = context
        return Strategy(
            market_regime=MarketRegime.BULL,
            positions=[
                Position(
                    asset="AAPL",
                    action=PositionAction.LONG,
                    size_pct=5.0,
                    stop_loss_pct=3.0,
                    take_profit_pct=6.0,
                    time_horizon_days=7,
                    confidence=0.6,
                )
            ],
            rationale="revised after feedback",
            risk_metrics=RiskMetrics(total_exposure_pct=5.0),
        )

    monkeypatch.setattr(agent, "generate_strategy", fake_generate)
    from schemas.analysis import FlowType, MarketAnalysis, MarketOutlook, OutlookDirection, SentimentView

    analysis = MarketAnalysis(
        user_id="u",
        query_context="AAPL outlook",
        flow_type=FlowType.BASELINE,
        sentiment=SentimentView(overall_score=0.2, key_themes=["earnings"]),
        outlook=MarketOutlook(
            direction=OutlookDirection.NEUTRAL,
            horizon_days=14,
            confidence=0.5,
            affected_assets=["AAPL"],
            narrative="Neutral near-term",
        ),
        news_bundle={},
        market_evidence={},
    )
    baseline = await fake_generate({})
    out = await agent.generate_strategy_from_feedback(
        baseline_analysis=analysis,
        baseline_strategy=baseline,
        baseline_reward=None,
        feedback_text="Make outlook more cautious; cut size",
        overall_verdict="partial",
        preference_hints=[{"prior_feedback": "prefer lower size"}],
    )
    assert out.positions
    ctx = captured["context"]
    assert "feedback" in ctx or "user_feedback" in ctx
    assert "baseline_strategy" in ctx or "prior_strategy" in ctx


@pytest.mark.asyncio
async def test_compare_strategy_outcomes(monkeypatch: pytest.MonkeyPatch):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy.compare_strategy_outcomes"""
    memory = _MemoryStub({"working_memory": {}, "recent_episodes": [], "semantic_knowledge": []})
    agent, _ = _build_agent(monkeypatch, memory)
    monkeypatch.setattr(
        "agents.common.llm_chat",
        AsyncMock(return_value="Revised cuts size and improves drawdown vs baseline."),
    )
    s1 = Strategy(
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
        rationale="baseline momentum",
        risk_metrics=RiskMetrics(total_exposure_pct=10.0),
    )
    r1 = RewardSignal(
        strategy_id=s1.strategy_id,
        terminal_reward=0.35,
        backtest_result=BacktestResult(
            strategy_id=s1.strategy_id,
            backtest_start=datetime.now(timezone.utc),
            backtest_end=datetime.now(timezone.utc),
            total_return=0.06,
            sharpe_ratio=0.9,
            max_drawdown=0.08,
            win_rate=0.55,
            profit_factor=1.1,
            total_trades=4,
            avg_trade_return=0.01,
            volatility=0.14,
        ),
    )
    s2 = Strategy(
        market_regime=MarketRegime.BULL,
        positions=[
            Position(
                asset="AAPL",
                action=PositionAction.LONG,
                size_pct=5.0,
                stop_loss_pct=2.5,
                take_profit_pct=6.0,
                time_horizon_days=10,
                confidence=0.6,
            )
        ],
        rationale="revised cautious sizing",
        risk_metrics=RiskMetrics(total_exposure_pct=5.0),
    )
    r2 = RewardSignal(
        strategy_id=s2.strategy_id,
        terminal_reward=0.48,
        backtest_result=BacktestResult(
            strategy_id=s2.strategy_id,
            backtest_start=datetime.now(timezone.utc),
            backtest_end=datetime.now(timezone.utc),
            total_return=0.05,
            sharpe_ratio=1.2,
            max_drawdown=0.04,
            win_rate=0.62,
            profit_factor=1.4,
            total_trades=4,
            avg_trade_return=0.012,
            volatility=0.10,
        ),
    )
    text = await agent.compare_strategy_outcomes(
        baseline_strategy=s1,
        baseline_reward=r1,
        revised_strategy=s2,
        revised_reward=r2,
        feedback_text="more cautious",
    )
    assert len(text) > 20


@pytest.mark.asyncio
async def test_refine_strategy_with_feedback(monkeypatch: pytest.MonkeyPatch, base_context, valid_strategy: Strategy):
    """FUNCTION TESTED: agents.agent3_strategy.Agent3Strategy.refine_strategy"""
    memory = _MemoryStub(base_context)
    agent, llm = _build_agent(monkeypatch, memory)
    original = valid_strategy
    refined_data = original.model_dump(mode="json")
    refined_data["strategy_id"] = "00000000-0000-4000-8000-000000000099"
    refined_data["positions"][0]["size_pct"] = 5.0
    llm.responses = [json.dumps(refined_data)]
    refined = await agent.refine_strategy(original, "Reduce AAPL position to 5%")
    assert refined.positions[0].size_pct == 5.0
    assert refined.strategy_id != original.strategy_id

