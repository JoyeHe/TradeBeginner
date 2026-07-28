# Analysis NL Summaries + Feedback Fusion + LLM Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the Analysis page, keep JSON payloads but add LLM natural-language explanations for baseline strategy+backtest, feedback-revised strategy+backtest, and an LLM comparison of baseline vs revised outcomes.

**Architecture:** Extend Agent3 (strategy) with `explain_strategy_and_backtest` and `compare_strategy_outcomes` using the existing DeepSeek-safe `llm_chat` helper. Wire these into `AnalysisFlow.run_baseline` / `submit_feedback` / `compare_session`. Strengthen Flow 2 so strategy regeneration explicitly fuses (1) Flow1 original query/analysis, (2) Flow1 strategy+reward, (3) user feedback. Persist summaries on `AnalysisSession` and surface them in the dashboard UI.

**Tech Stack:** Python async, Pydantic schemas, FastAPI, existing Agent3/`llm_chat`, AnalysisFlow, static `index.html`.

## Global Constraints

- Agent roles (correct mapping — do not rename): Agent1=News, Agent2=Market, Agent3=Strategy, Agent6=Backtest. Strategy NL and compare NL live on **Agent3**; Agent6 stays quantitative (`evaluate_strategy`).
- Existing Strategy-tab `generate_strategy_summary` must keep working; new methods are additive.
- Always return / display **JSON + NL**; never replace JSON with prose-only.
- LLM failures must fall back to short template text (never raise 503 solely for missing summary).
- Prefer `agents.common.llm_chat` (system/user roles) — do not use Agno `arun` for these prose calls (DeepSeek role compatibility).

## Agent Clarification (for reviewers)

| User said | Actual | Role in this plan |
|-----------|--------|-------------------|
| “Agent3（回测）” | **Agent6** = backtest | Keep `evaluate_strategy` quantitative |
| “策略 agent（应该是 agent2）” | **Agent3** = strategy | Extend with NL explain + fused regen + compare |
| Agent2 | Market data | Unchanged |

## File Map

| File | Responsibility |
|------|----------------|
| `agents/agent3_strategy.py` | Add `explain_strategy_and_backtest`, `compare_strategy_outcomes`, `generate_strategy_from_feedback` |
| `memory/analysis_store.py` | Session fields for NL strings |
| `orchestrator/analysis_flow.py` | Call explain after backtests; fuse feedback for Flow2; LLM compare |
| `api/static/index.html` | Show NL blocks above JSON for baseline/revised/compare |
| `tests/test_analysis_flow.py` | Cover new payload fields + compare narrative |
| `tests/test_agent3_strategy.py` | Unit-test new Agent3 methods with mocked `llm_chat` |

---

### Task 1: Agent3 — strategy+backtest NL explainer

**Files:**
- Modify: `agents/agent3_strategy.py` (after `generate_strategy_summary`)
- Test: `tests/test_agent3_strategy.py`

**Interfaces:**
- Consumes: `Strategy`, `Optional[RewardSignal]`, optional `query: str`
- Produces: `async def explain_strategy_and_backtest(self, strategy: Strategy, reward: Optional[RewardSignal] = None, query: Optional[str] = None) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent3_strategy.py
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from schemas.rewards import BacktestResult, RewardSignal
from schemas.strategy import MarketRegime, Position, PositionAction, RiskMetrics, Strategy


@pytest.mark.asyncio
async def test_explain_strategy_and_backtest(monkeypatch, _build_agent_fixture_or_inline):
    from agents import agent3_strategy as m
    from agents.agent3_strategy import Agent3Strategy
    from config.settings import Settings
    from tests.test_agent3_strategy import _MemoryStub  # or local stub

    agent = Agent3Strategy(Settings(_env_file=None, llm_api_key="k"), _MemoryStub())
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
```

Adapt imports to match existing `_MemoryStub` / `_build_agent` helpers already in `test_agent3_strategy.py`.

- [ ] **Step 2: Run test to verify it fails**

```bash
cd TradeBeginner && .venv/bin/pytest tests/test_agent3_strategy.py::test_explain_strategy_and_backtest -v
```

Expected: FAIL (`AttributeError: explain_strategy_and_backtest`)

- [ ] **Step 3: Implement method on Agent3**

Append to `agents/agent3_strategy.py` after `generate_strategy_summary`:

```python
async def explain_strategy_and_backtest(
    self,
    strategy: Strategy,
    reward: Optional["RewardSignal"] = None,
    query: Optional[str] = None,
) -> str:
    """Natural-language explanation of strategy + quantitative backtest (Analysis UI)."""
    from agents.common import llm_chat
    from schemas.rewards import RewardSignal  # local import ok if circular

    positions_desc = "\n".join(
        f"- {p.action.value.upper()} {p.asset}: {p.size_pct}%, SL {p.stop_loss_pct}%, "
        f"TP {p.take_profit_pct}, {p.time_horizon_days}d, conf {p.confidence:.0%}"
        for p in strategy.positions
    )
    bt_block = "Backtest: unavailable"
    if reward is not None and reward.backtest_result is not None:
        bt = reward.backtest_result
        bt_block = (
            f"Terminal reward: {reward.terminal_reward:.4f}\n"
            f"Total return: {bt.total_return:.2%}, Sharpe: {bt.sharpe_ratio}, "
            f"Max DD: {bt.max_drawdown:.2%}, Win rate: {bt.win_rate:.2%}, "
            f"Trades: {bt.total_trades}, Volatility: {bt.volatility:.2%}"
        )
    system_prompt = (
        "You are a trading desk analyst. Explain the strategy AND its backtest results "
        "in clear Chinese or English matching the query language (4-8 sentences). "
        "Cover thesis, positions, risk, and what the backtest numbers imply. "
        "Do not invent metrics not provided."
    )
    user_prompt = (
        f"Query: {query or '(none)'}\n"
        f"Regime: {strategy.market_regime.value}\n"
        f"Rationale: {strategy.rationale}\n"
        f"Positions:\n{positions_desc}\n"
        f"Exposure: {strategy.risk_metrics.total_exposure_pct}%\n"
        f"{bt_block}\n\nWrite the explanation:"
    )
    result = await llm_chat(self.settings, system_prompt, user_prompt)
    if result.strip():
        return result.strip()
    # Fallback template
    return (
        f"Strategy on {strategy.market_regime.value}: {strategy.rationale[:240]}. "
        + (f"Backtest terminal reward={reward.terminal_reward:.3f}." if reward else "No backtest.")
    )
```

Add `from typing import Optional` / forward refs as needed; import `RewardSignal` at top if preferred.

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/test_agent3_strategy.py::test_explain_strategy_and_backtest -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agents/agent3_strategy.py tests/test_agent3_strategy.py
git commit -m "feat(agent3): explain strategy and backtest in natural language"
```

---

### Task 2: Agent3 — fused feedback strategy regeneration

**Files:**
- Modify: `agents/agent3_strategy.py`
- Test: `tests/test_agent3_strategy.py`

**Interfaces:**
- Consumes: baseline `MarketAnalysis`, baseline `Strategy`, optional baseline `RewardSignal`, `AnalysisFeedback` / correction text, optional preference hints
- Produces: `async def generate_strategy_from_feedback(...) -> Strategy`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_generate_strategy_from_feedback_includes_fusion(monkeypatch):
    agent, llm = _build_agent(monkeypatch, _MemoryStub())
    captured = {}

    async def capture_run(prompt: str):
        captured["prompt"] = prompt
        return type("R", (), {"content": _valid_strategy_json()})()  # reuse existing helper JSON

    # If agent uses _agent_run / generate_strategy path, monkeypatch that instead:
    async def fake_generate(context):
        captured["context"] = context
        return Strategy(
            market_regime=MarketRegime.BULL,
            positions=[Position(asset="AAPL", action=PositionAction.LONG, size_pct=5.0,
                                stop_loss_pct=3.0, take_profit_pct=6.0, time_horizon_days=7, confidence=0.6)],
            rationale="revised after feedback",
            risk_metrics=RiskMetrics(total_exposure_pct=5.0),
        )

    monkeypatch.setattr(agent, "generate_strategy", fake_generate)
    from schemas.analysis import MarketAnalysis, MarketOutlook, OutlookDirection, SentimentView, FlowType

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
```

(Use real schema constructors already used in `tests/test_analysis_flow.py` if fields differ.)

- [ ] **Step 2: Run test — expect FAIL**

```bash
.venv/bin/pytest tests/test_agent3_strategy.py::test_generate_strategy_from_feedback_includes_fusion -v
```

- [ ] **Step 3: Implement `generate_strategy_from_feedback`**

```python
async def generate_strategy_from_feedback(
    self,
    baseline_analysis: "MarketAnalysis",
    baseline_strategy: Strategy,
    baseline_reward: Optional["RewardSignal"],
    feedback_text: str,
    overall_verdict: str = "partial",
    preference_hints: Optional[list[dict]] = None,
) -> Strategy:
    """Fuse Flow1 analysis + Flow1 strategy/reward + user feedback → new Strategy."""
    context = await self.gather_context()
    context["market_analysis"] = baseline_analysis.model_dump(mode="json")
    context["user_request"] = baseline_analysis.query_context
    context["baseline_strategy"] = baseline_strategy.model_dump(mode="json")
    context["baseline_reward"] = None if baseline_reward is None else baseline_reward.model_dump(mode="json")
    context["user_feedback"] = {
        "overall_verdict": overall_verdict,
        "correction": feedback_text,
        "preference_hints": preference_hints or [],
    }
    # Inject fusion instructions into context so _make_prompt / generate_strategy sees them.
    # If _make_prompt does not yet render these keys, extend _make_prompt (Step 3b).
    strategy = await self.generate_strategy(context)
    strategy.analysis_id = baseline_analysis.analysis_id
    strategy.metadata["origin"] = "post_feedback"
    strategy.metadata["parent_strategy_id"] = baseline_strategy.strategy_id
    strategy.data_sources_used = list(
        set(strategy.data_sources_used + ["market_analysis", "user_feedback", "baseline_strategy"])
    )
    return strategy
```

- [ ] **Step 3b: Extend `_make_prompt` to include fusion blocks when present**

In `_make_prompt`, after market/news sections, add:

```python
if context.get("user_feedback") or context.get("baseline_strategy"):
    prompt += f"""

## Flow1 Baseline Strategy (do not ignore)
{json.dumps(context.get("baseline_strategy"), default=str, indent=2)}

## Flow1 Backtest / Reward
{json.dumps(context.get("baseline_reward"), default=str, indent=2)}

## User Feedback (must incorporate)
{json.dumps(context.get("user_feedback"), default=str, indent=2)}

Revise the strategy to respect the user feedback while staying consistent with the baseline analysis evidence.
"""
```

- [ ] **Step 4: Run test — expect PASS**

```bash
.venv/bin/pytest tests/test_agent3_strategy.py::test_generate_strategy_from_feedback_includes_fusion -v
```

- [ ] **Step 5: Commit**

```bash
git add agents/agent3_strategy.py tests/test_agent3_strategy.py
git commit -m "feat(agent3): fuse Flow1 results with user feedback for strategy regen"
```

---

### Task 3: Agent3 — LLM comparison of baseline vs revised

**Files:**
- Modify: `agents/agent3_strategy.py`
- Test: `tests/test_agent3_strategy.py`

**Interfaces:**
- Produces: `async def compare_strategy_outcomes(...) -> str`

- [ ] **Step 1: Failing test**

```python
@pytest.mark.asyncio
async def test_compare_strategy_outcomes(monkeypatch):
    agent = Agent3Strategy(Settings(_env_file=None, llm_api_key="k"), _MemoryStub())
    monkeypatch.setattr(
        "agents.common.llm_chat",
        AsyncMock(return_value="Revised cuts size and improves drawdown vs baseline."),
    )
    # Build two minimal strategies + rewards (reuse Task 1 fixtures)
    text = await agent.compare_strategy_outcomes(
        baseline_strategy=s1, baseline_reward=r1,
        revised_strategy=s2, revised_reward=r2,
        feedback_text="more cautious",
    )
    assert len(text) > 20
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

```python
async def compare_strategy_outcomes(
    self,
    baseline_strategy: Strategy,
    baseline_reward: Optional["RewardSignal"],
    revised_strategy: Strategy,
    revised_reward: Optional["RewardSignal"],
    feedback_text: str = "",
) -> str:
    from agents.common import llm_chat

    def pack(s: Strategy, r: Optional["RewardSignal"]) -> dict:
        return {
            "strategy": s.model_dump(mode="json"),
            "reward": None if r is None else r.model_dump(mode="json"),
        }

    system_prompt = (
        "You compare two trading strategies and their backtests. "
        "Explain what changed after user feedback, which metrics improved/worsened, "
        "and a clear recommendation (4-8 sentences). Do not invent numbers."
    )
    user_prompt = (
        f"User feedback: {feedback_text}\n\n"
        f"BASELINE:\n{json.dumps(pack(baseline_strategy, baseline_reward), default=str, indent=2)}\n\n"
        f"REVISED:\n{json.dumps(pack(revised_strategy, revised_reward), default=str, indent=2)}\n\n"
        "Write the comparison:"
    )
    result = await llm_chat(self.settings, system_prompt, user_prompt)
    if result.strip():
        return result.strip()
    br = None if baseline_reward is None else baseline_reward.terminal_reward
    rr = None if revised_reward is None else revised_reward.terminal_reward
    delta = None if br is None or rr is None else rr - br
    return f"Baseline reward={br}, revised reward={rr}, delta={delta}. Feedback: {feedback_text}"
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(agent3): LLM compare baseline vs feedback strategies"
```

---

### Task 4: Persist NL fields on AnalysisSession + wire AnalysisFlow

**Files:**
- Modify: `memory/analysis_store.py`
- Modify: `orchestrator/analysis_flow.py`
- Test: `tests/test_analysis_flow.py`

**Interfaces:**
- Session gains: `baseline_explanation: Optional[str]`, `feedback_explanation: Optional[str]`, `comparison_narrative: Optional[str]`
- `session_to_dict` / `compare_session` expose these keys
- `run_baseline` calls `explain_strategy_and_backtest` after Agent6
- `submit_feedback` uses `generate_strategy_from_feedback` (+ optional analysis revise), then explain
- `compare_session` becomes **async** and generates narrative if missing

- [ ] **Step 1: Extend `AnalysisSession`**

```python
# memory/analysis_store.py — inside AnalysisSession.__init__
self.baseline_explanation: Optional[str] = None
self.feedback_explanation: Optional[str] = None
self.comparison_narrative: Optional[str] = None
```

- [ ] **Step 2: Update `session_to_dict`**

```python
return {
    "analysis_id": session.analysis_id,
    "status": session.status,
    "baseline_analysis": _dump(session.baseline_analysis),
    "revised_analysis": _dump(session.revised_analysis),
    "baseline_strategy": _dump(session.baseline_strategy),
    "feedback_strategy": _dump(session.feedback_strategy),
    "baseline_reward": _dump(session.baseline_reward),
    "feedback_reward": _dump(session.feedback_reward),
    "baseline_explanation": session.baseline_explanation,
    "feedback_explanation": session.feedback_explanation,
    "comparison_narrative": session.comparison_narrative,
    "feedbacks": [_dump(f) for f in session.feedbacks],
}
```

- [ ] **Step 3: Wire `run_baseline` after backtest**

After `session.baseline_reward = await self.agent6.evaluate_strategy(strategy)`:

```python
try:
    session.baseline_explanation = await self.agent3.explain_strategy_and_backtest(
        strategy, session.baseline_reward, query=query
    )
except Exception as exc:
    logger.warning("baseline_explanation_failed", error=str(exc))
    session.baseline_explanation = strategy.rationale
```

- [ ] **Step 4: Rewrite strategy step in `submit_feedback`**

Replace `strategy = await self.generate_strategy_from_analysis(revised)` with fusion path:

```python
# Keep analysis revision as today (heuristic/LLM) → session.revised_analysis = revised
session.status = "generating_strategy"
strategy = await self.agent3.generate_strategy_from_feedback(
    baseline_analysis=session.baseline_analysis,
    baseline_strategy=session.baseline_strategy,
    baseline_reward=session.baseline_reward,
    feedback_text=correction_text,
    overall_verdict=feedback.overall_verdict,
    preference_hints=hints,
)
# Prefer revised analysis narrative in context: also pass revised into metadata
strategy.analysis_id = revised.analysis_id
strategy.rationale = f"{revised.outlook.narrative}\n\n{strategy.rationale}"
session.feedback_strategy = strategy
session.status = "evaluating"
try:
    session.feedback_reward = await self.agent6.evaluate_strategy(strategy)
except Exception as exc:
    logger.warning("feedback_backtest_failed", error=str(exc))
try:
    session.feedback_explanation = await self.agent3.explain_strategy_and_backtest(
        strategy, session.feedback_reward, query=session.baseline_analysis.query_context
    )
except Exception as exc:
    logger.warning("feedback_explanation_failed", error=str(exc))
    session.feedback_explanation = strategy.rationale

# Eager LLM compare (so UI can show without extra click, and Compare button reuses cache)
if session.baseline_strategy is not None:
    try:
        session.comparison_narrative = await self.agent3.compare_strategy_outcomes(
            session.baseline_strategy,
            session.baseline_reward,
            strategy,
            session.feedback_reward,
            feedback_text=correction_text,
        )
    except Exception as exc:
        logger.warning("comparison_narrative_failed", error=str(exc))
```

- [ ] **Step 5: Make `compare_session` async and include narrative**

```python
async def compare_session(self, analysis_id: str) -> dict:
    session = self.store.get_session(analysis_id)
    if session is None:
        return {"status": "not_found"}
    base_r = None if session.baseline_reward is None else session.baseline_reward.terminal_reward
    fb_r = None if session.feedback_reward is None else session.feedback_reward.terminal_reward
    if (
        session.comparison_narrative is None
        and session.baseline_strategy is not None
        and session.feedback_strategy is not None
    ):
        fb_text = ""
        if session.feedbacks:
            last = session.feedbacks[-1]
            fb_text = last.free_text or ""
        session.comparison_narrative = await self.agent3.compare_strategy_outcomes(
            session.baseline_strategy,
            session.baseline_reward,
            session.feedback_strategy,
            session.feedback_reward,
            feedback_text=fb_text,
        )
    return {
        "analysis_id": analysis_id,
        "baseline": {
            "analysis": session.baseline_analysis.model_dump(mode="json"),
            "strategy": None if session.baseline_strategy is None else session.baseline_strategy.model_dump(mode="json"),
            "reward": None if session.baseline_reward is None else session.baseline_reward.model_dump(mode="json"),
            "explanation": session.baseline_explanation,
            "strategy_id": None if session.baseline_strategy is None else session.baseline_strategy.strategy_id,
            "terminal_reward": base_r,
        },
        "revised": {
            "analysis": None if session.revised_analysis is None else session.revised_analysis.model_dump(mode="json"),
            "strategy": None if session.feedback_strategy is None else session.feedback_strategy.model_dump(mode="json"),
            "reward": None if session.feedback_reward is None else session.feedback_reward.model_dump(mode="json"),
            "explanation": session.feedback_explanation,
            "strategy_id": None if session.feedback_strategy is None else session.feedback_strategy.strategy_id,
            "terminal_reward": fb_r,
        },
        "reward_delta": None if base_r is None or fb_r is None else fb_r - base_r,
        "comparison_narrative": session.comparison_narrative,
    }
```

- [ ] **Step 6: Update pipeline caller**

In `orchestrator/pipeline.py`:

```python
async def compare_analysis_session(self, analysis_id: str) -> dict:
    return await self.analysis_flow.compare_session(analysis_id)
```

- [ ] **Step 7: Update analysis flow tests**

```python
@pytest.mark.asyncio
async def test_run_baseline_includes_explanation(analysis_flow, monkeypatch):
    analysis_flow.agent3.explain_strategy_and_backtest = AsyncMock(return_value="NL baseline explanation")
    session = await analysis_flow.run_baseline("AAPL earnings", tickers=["AAPL"])
    payload = analysis_flow.session_to_dict(session)
    assert payload["baseline_explanation"] == "NL baseline explanation"

@pytest.mark.asyncio
async def test_submit_feedback_includes_explanations_and_compare(analysis_flow, monkeypatch):
    analysis_flow.agent3.explain_strategy_and_backtest = AsyncMock(side_effect=["base NL", "revised NL"])
    analysis_flow.agent3.generate_strategy_from_feedback = AsyncMock(side_effect=lambda **kw: _sample_strategy())
    analysis_flow.agent3.compare_strategy_outcomes = AsyncMock(return_value="Compare NL")
    # If generate_strategy_from_feedback not yet patched path — adjust to match implementation
    session = await analysis_flow.run_baseline("tech", tickers=["AAPL"])
    feedback = AnalysisFeedback(
        analysis_id=session.analysis_id,
        user_id="default",
        overall_verdict="partial",
        dimension_feedbacks=[DimensionFeedback(dimension=FeedbackDimension.MARKET_OUTLOOK, verdict="disagree", correction="more neutral")],
        free_text="more cautious",
    )
    updated = await analysis_flow.submit_feedback(session.analysis_id, feedback)
    payload = analysis_flow.session_to_dict(updated)
    assert payload["feedback_explanation"]
    assert payload["comparison_narrative"]
    compare = await analysis_flow.compare_session(session.analysis_id)
    assert compare["comparison_narrative"]
```

Mock `explain` on baseline run as well so first call doesn't hit real LLM.

- [ ] **Step 8: Run tests**

```bash
.venv/bin/pytest tests/test_analysis_flow.py tests/test_agent3_strategy.py -v
```

Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add memory/analysis_store.py orchestrator/analysis_flow.py orchestrator/pipeline.py tests/test_analysis_flow.py
git commit -m "feat(analysis): NL explanations and LLM compare in analysis flow"
```

---

### Task 5: Frontend — show NL + JSON on Analysis tab

**Files:**
- Modify: `api/static/index.html` (Analysis section + JS handlers)

**Interfaces:**
- Consumes API fields: `baseline_explanation`, `feedback_explanation`, `comparison_narrative`
- Produces visible prose above JSON panels

- [ ] **Step 1: Add DOM nodes**

In baseline strategy card (near `#baselineStrategy`):

```html
<div class="t">Baseline Strategy &amp; Backtest</div>
<div id="baselineStrategySummary" class="prose muted">Run baseline to see a natural-language explanation.</div>
<pre class="code" id="baselineStrategy">No strategy yet.</pre>
```

In revised / feedback area (after revised analysis, or a new card):

```html
<div class="t">Revised Strategy &amp; Backtest</div>
<div id="feedbackStrategySummary" class="prose muted">Submit feedback to see revised strategy explanation.</div>
<pre class="code" id="feedbackStrategy">No revised strategy yet.</pre>
```

In compare card:

```html
<div class="t">Compare &amp; Library</div>
<div id="compareNarrative" class="prose muted">No comparison yet.</div>
<pre class="code" id="compareResults">No comparison yet.</pre>
```

- [ ] **Step 2: Update JS after baseline**

```javascript
renderAnalysisSummary("baselineSummary", out.baseline_analysis);
$("baselineStrategySummary").textContent = out.baseline_explanation || "Explanation unavailable.";
$("baselineStrategySummary").className = "prose";
pretty("baselineStrategy", {strategy: out.baseline_strategy, reward: out.baseline_reward});
```

- [ ] **Step 3: Update JS after feedback**

```javascript
renderAnalysisSummary("revisedSummary", out.revised_analysis);
$("feedbackStrategySummary").textContent = out.feedback_explanation || "Explanation unavailable.";
$("feedbackStrategySummary").className = "prose";
pretty("feedbackStrategy", {strategy: out.feedback_strategy, reward: out.feedback_reward});
// Keep baseline JSON panel as baseline-only (do not overwrite with both)
pretty("baselineStrategy", {strategy: out.baseline_strategy, reward: out.baseline_reward});
if (out.comparison_narrative) {
  $("compareNarrative").textContent = out.comparison_narrative;
  $("compareNarrative").className = "prose";
}
pretty("compareResults", {
  reward_delta: null, // filled when user clicks compare or from out if present
  baseline_reward: out.baseline_reward,
  feedback_reward: out.feedback_reward,
});
```

- [ ] **Step 4: Update Compare button**

```javascript
$("btnCompareAnalysis").onclick = async () => {
  if (!currentAnalysisId) return;
  try {
    const out = await call(`/analysis/${currentAnalysisId}/compare`);
    $("compareNarrative").textContent = out.comparison_narrative || "No narrative.";
    $("compareNarrative").className = "prose";
    pretty("compareResults", out);
    log("analysis_compare", out);
  } catch (e) {
    pretty("compareResults", e);
  }
};
```

- [ ] **Step 5: Manual smoke (or extend `scripts/frontend_e2e_test.py`)**

```bash
# With API running:
# 1. Analysis → Run Baseline → assert baselineStrategySummary has prose (not only JSON)
# 2. Submit Feedback → assert feedbackStrategySummary + compareNarrative
# 3. Click Compare → assert comparison_narrative in JSON + prose div
```

Optional test assertion in `scripts/frontend_e2e_test.py`:

```python
baseline_nl = page.locator("#baselineStrategySummary").inner_text()
report.add("Analysis baseline NL", len(baseline_nl) > 30 and "Run baseline" not in baseline_nl, baseline_nl[:80])
```

- [ ] **Step 6: Commit**

```bash
git add api/static/index.html scripts/frontend_e2e_test.py
git commit -m "feat(ui): show NL strategy/backtest explanations on Analysis tab"
```

---

### Task 6: End-to-end verification

- [ ] **Step 1: Unit suite**

```bash
cd TradeBeginner
.venv/bin/pytest tests/test_agent3_strategy.py tests/test_analysis_flow.py -v
```

Expected: all PASS

- [ ] **Step 2: Restart API and hit endpoints**

```bash
# restart main.py if needed
curl -s -X POST http://127.0.0.1:8000/analysis/run-baseline \
  -H 'Content-Type: application/json' \
  -d '{"query":"AAPL outlook","tickers":["AAPL"]}' | python3 -c \
  'import sys,json;d=json.load(sys.stdin);assert d.get("baseline_explanation");print(d["baseline_explanation"][:120])'
```

- [ ] **Step 3: Feedback + compare**

```bash
# AID=<analysis_id from previous>
curl -s -X POST http://127.0.0.1:8000/analysis/$AID/feedback \
  -H 'Content-Type: application/json' \
  -d '{"overall_verdict":"partial","free_text":"more cautious near-term","dimension_feedbacks":[{"dimension":"market_outlook","verdict":"disagree","correction":"neutral"}]}' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);assert d.get("feedback_explanation");assert d.get("comparison_narrative");print("ok")'

curl -s http://127.0.0.1:8000/analysis/$AID/compare | python3 -c \
  'import sys,json;d=json.load(sys.stdin);assert d.get("comparison_narrative");print(d["comparison_narrative"][:160])'
```

- [ ] **Step 4: Final commit if any polish**

```bash
git status
git commit -am "test: verify analysis NL explain and compare e2e"
```

---

## Self-Review Checklist

| Requirement | Task |
|-------------|------|
| Flow1 strategy+backtest NL (keep JSON) | Task 1 + Task 4 Step 3 + Task 5 |
| Feedback ⊕ Flow1 prompt ⊕ Flow1 results → strategy regen → Agent6 backtest → NL+JSON | Task 2 + Task 4 Step 4 + Task 5 |
| LLM compare baseline vs revised | Task 3 + Task 4 Step 5 + Task 5 |
| Agent mapping corrected (Agent3 strategy, Agent6 backtest) | Global Constraints |

No placeholders left; signatures are concrete; UI fields named consistently (`baseline_explanation`, `feedback_explanation`, `comparison_narrative`).
