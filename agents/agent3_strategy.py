"""Agent 3: Strategy generation with RL-ready trace logging."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Optional
import uuid

import structlog
from agno.agent import Agent
from pydantic import ValidationError

from agents.common import agent_arun, build_agno_model
from config.settings import Settings
from memory import MemoryManager
from schemas.strategy import MarketRegime, Position, PositionAction, RiskMetrics, Strategy
from tools.agno_tools import StrategyAgentTools

if TYPE_CHECKING:
    from schemas.analysis import MarketAnalysis
    from schemas.rewards import RewardSignal

logger = structlog.get_logger(__name__)

STRATEGY_SYSTEM_PROMPT = """
You are Agent 3, the strategy generation brain of a multi-agent trading system.
Generate structured, risk-aware strategies from market data, sentiment, and memory context.
Always return valid JSON that matches the requested schema.
"""


class StrategyGenerationError(RuntimeError):
    """Raised when strategy JSON cannot be parsed/validated."""


class Agent3Strategy:
    """Generate and refine structured strategy actions."""

    def __init__(self, settings: Settings, memory: MemoryManager, agent1=None, agent2=None):
        self.settings = settings
        self.memory = memory
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent = Agent(
            name="agent3_strategy",
            model=build_agno_model(settings),
            tools=[StrategyAgentTools()],
            instructions=[STRATEGY_SYSTEM_PROMPT],
            markdown=False,
        )

    async def _agent_run(self, prompt: str) -> str:
        """Invoke Agno agent.arun (OpenAI-compatible / DeepSeek-safe via Agent instructions)."""
        if not self.settings.llm_api_key:
            return ""
        return await agent_arun(self.agent, prompt)

    async def gather_context(self) -> dict:
        """Assemble deterministic state for strategy generation."""
        base = await self.memory.get_strategy_context()
        working = base.get("working_memory", {})
        return {
            "market_snapshot": self._dump(working.get("market_snapshot")),
            "news_digest": self._dump(working.get("news_digest")),
            "portfolio_state": self._dump(working.get("portfolio_state")),
            "recent_episodes": base.get("recent_episodes", []),
            "semantic_knowledge": base.get("semantic_knowledge", []),
        }

    async def request_additional_research(self, research_type: str, parameters: dict) -> dict:
        """Request extra research from Agent 1 or Agent 2."""
        if research_type == "ticker_news" and self.agent1 is not None:
            return {"ticker_news": await self.agent1.research_tickers(parameters.get("tickers", []))}
        if research_type == "ticker_technical" and self.agent2 is not None:
            ticker = parameters.get("ticker", "")
            return {"ticker_technical": await self.agent2.analyze_ticker(ticker)}
        if research_type == "screen_stocks" and self.agent2 is not None:
            from tools.market_tools import screen_stocks

            return {"screen_result": await screen_stocks(parameters)}
        return {"note": "no_additional_research_executed"}

    def _make_prompt(self, context: dict) -> str:
        schema_json = json.dumps(Strategy.model_json_schema(), indent=2)
        prompt = f"""
{STRATEGY_SYSTEM_PROMPT}

## Current Market Conditions
{json.dumps(context.get("market_snapshot"), default=str, indent=2)}

## News & Sentiment Digest
{json.dumps(context.get("news_digest"), default=str, indent=2)}

## Current Portfolio
{json.dumps(context.get("portfolio_state"), default=str, indent=2)}

## Historical Context
{json.dumps(context.get("recent_episodes"), default=str, indent=2)}

## Relevant Market Knowledge
{json.dumps(context.get("semantic_knowledge"), default=str, indent=2)}

## Market Analysis (if available)
{json.dumps(context.get("market_analysis"), default=str, indent=2)}
"""
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
        prompt += f"""
## Task
Generate one strategy as strict JSON matching this schema:
{schema_json}
"""
        return prompt

    def _extract_json(self, text: str) -> str:
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        code = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.MULTILINE)
        if code:
            return code.group(1)
        raw = re.search(r"(\{[\s\S]*\})", text, flags=re.MULTILINE)
        if raw:
            return raw.group(1)
        raise StrategyGenerationError("No JSON object found in response.")

    async def _repair_json(self, bad_text: str) -> str:
        prompt = (
            "Your previous response was not valid JSON.\n"
            "Return ONLY valid JSON matching this schema:\n"
            f"{json.dumps(Strategy.model_json_schema(), indent=2)}\n"
            f"Previous response:\n{bad_text}"
        )
        return await self._agent_run(prompt)

    def _default_strategy(self, context: dict) -> Strategy:
        regime = MarketRegime.UNCERTAIN
        market = context.get("market_snapshot") or {}
        breadth = market.get("market_breadth") or {}
        if isinstance(breadth, dict) and breadth.get("regime") in {x.value for x in MarketRegime}:
            regime = MarketRegime(breadth["regime"])
        return Strategy(
            market_regime=regime,
            positions=[
                Position(
                    asset="SPY",
                    action=PositionAction.LONG,
                    size_pct=10.0,
                    entry_price_target=None,
                    stop_loss_pct=3.0,
                    take_profit_pct=6.0,
                    time_horizon_days=10,
                    confidence=0.45,
                )
            ],
            rationale="Fallback strategy generated due to unavailable LLM response.",
            data_sources_used=["market_snapshot", "news_digest", "episodic_memory", "semantic_memory"],
            risk_metrics=RiskMetrics(total_exposure_pct=10.0),
            metadata={"fallback": True},
        )

    async def generate_strategy(self, context: dict) -> Strategy:
        """Generate validated strategy JSON with retries and repair."""
        prompt = self._make_prompt(context)
        raw_text = await self._agent_run(prompt)
        for attempt in range(3):
            try:
                payload = self._extract_json(raw_text)
                parsed = Strategy.model_validate(json.loads(payload))
                await self._log_rl_trace(context, prompt, raw_text, parsed)
                return parsed
            except (json.JSONDecodeError, ValidationError, StrategyGenerationError) as exc:
                if attempt == 2:
                    logger.warning("strategy_generation_fallback", error=str(exc))
                    parsed = self._default_strategy(context)
                    await self._log_rl_trace(context, prompt, raw_text, parsed, error=str(exc))
                    return parsed
                raw_text = await self._repair_json(raw_text)
        raise StrategyGenerationError("Unexpected strategy generation flow.")

    async def self_evaluate_strategy(self, strategy: Strategy) -> dict:
        """Attach quick self-evaluation metadata."""
        concentration = sum(p.size_pct for p in strategy.positions[:3])
        concerns = []
        if concentration > 50:
            concerns.append("Top positions exceed 50% exposure.")
        return {"concerns": concerns, "confidence_alignment": "moderate"}

    async def generate(self) -> Strategy:
        """Generate trading strategy from current system context."""
        context = await self.gather_context()
        strategy = await self.generate_strategy(context)
        eval_result = await self.self_evaluate_strategy(strategy)
        strategy.metadata["self_evaluation"] = eval_result
        await self.memory.working.set("active_strategy", strategy, category="strategy")
        await self.memory.working.set("rl_trace_last_context", context, category="strategy")
        return strategy

    async def refine_strategy(self, strategy: Strategy, feedback: str) -> Strategy:
        """Refine strategy via feedback-aware regeneration."""
        context = await self.gather_context()
        context["refinement_target"] = strategy.model_dump(mode="json")
        context["feedback"] = feedback
        refined = await self.generate_strategy(context)
        await self.memory.working.set("active_strategy", refined, category="strategy")
        return refined

    async def _extract_tickers_from_request(self, user_request: str) -> list[str]:
        """Use LLM to extract asset tickers from a user request."""
        raw = await self._agent_run(
            "Extract ticker symbols from the user request. "
            "Return a JSON array of strings, e.g. [\"SOL-USD\",\"BTC-USD\"]. "
            "Map common names: SOL→SOL-USD, BTC/Bitcoin→BTC-USD, ETH→ETH-USD, "
            "AAPL/Apple→AAPL, MSFT/Microsoft→MSFT, NVDA/Nvidia→NVDA etc. "
            "Return [] if no specific assets are mentioned.\n\n"
            f"User request: {user_request}"
        )
        try:
            import re as _re
            m = _re.search(r"\[[\s\S]*\]", raw)
            if m:
                return json.loads(m.group())
        except Exception:
            pass
        return []

    async def _fetch_user_specific_data(self, user_request: str, tickers: list[str]) -> dict:
        """Ask Agent1 for topic-specific news and Agent2 for ticker-specific data."""
        user_data: dict = {}
        if self.agent1 is not None:
            try:
                news = await self.agent1.user_search(user_request)
                user_data["user_news"] = {
                    "item_count": news.get("item_count", 0),
                    "source_breakdown": news.get("source_breakdown", {}),
                    "digest": news.get("digest", {}),
                    "top_items": [
                        {"title": i.get("title", ""), "sentiment": i.get("sentiment_score"), "category": i.get("category")}
                        for i in news.get("items", [])[:10]
                    ],
                }
                logger.info("agent3_fetched_news_from_agent1", tickers=tickers, items=news.get("item_count", 0))
            except Exception as exc:
                logger.warning("agent3_agent1_fetch_failed", error=str(exc))

        if self.agent2 is not None and tickers:
            try:
                market = await self.agent2.user_lookup(" ".join(tickers))
                user_data["user_market"] = market.get("data", {})
                logger.info("agent3_fetched_market_from_agent2", tickers=list(market.get("data", {}).keys()))
            except Exception as exc:
                logger.warning("agent3_agent2_fetch_failed", error=str(exc))

        return user_data

    async def generate_from_user_request(self, user_request: str) -> Strategy:
        """Generate a strategy driven by an explicit user instruction.

        Full pipeline:
        1. Extract tickers from user request via LLM
        2. Request Agent1 for topic-specific news + sentiment
        3. Request Agent2 for ticker-specific price data + indicators
        4. Combine with standard context (background market/news/memory)
        5. Generate strategy via Agno agent.arun() with all data
        """
        tickers = await self._extract_tickers_from_request(user_request)
        logger.info("agent3_extracted_tickers", tickers=tickers, request=user_request)

        user_data = await self._fetch_user_specific_data(user_request, tickers)

        context = await self.gather_context()
        context["user_request"] = user_request
        context["user_specific_news"] = user_data.get("user_news")
        context["user_specific_market"] = user_data.get("user_market")
        context["requested_tickers"] = tickers

        prompt = self._make_prompt(context)
        if user_data.get("user_news"):
            prompt += (
                "\n\n## USER-SPECIFIC NEWS (from Agent 1)\n"
                f"Agent 1 found {user_data['user_news']['item_count']} articles "
                f"from sources: {user_data['user_news']['source_breakdown']}\n"
                f"Digest sentiment: {user_data['user_news']['digest'].get('overall_market_sentiment', 'N/A')}\n"
                f"Top headlines:\n"
                + "\n".join(
                    f"- [{h.get('category','?')}] {h.get('title','')} (sentiment: {h.get('sentiment')})"
                    for h in user_data["user_news"].get("top_items", [])[:8]
                )
                + "\n"
            )
        if user_data.get("user_market"):
            prompt += "\n## USER-SPECIFIC MARKET DATA (from Agent 2)\n"
            for tk, data in user_data["user_market"].items():
                bar = data.get("last_bar", {})
                ind = data.get("indicators", {})
                prompt += (
                    f"**{tk}**: close={bar.get('close')}, "
                    f"RSI={ind.get('rsi_14')}, SMA20={ind.get('sma_20')}, "
                    f"MACD={ind.get('macd')}, ADX={ind.get('adx')}\n"
                )
        prompt += (
            "\n\n## USER REQUEST\n"
            f"The user has specifically asked: \"{user_request}\"\n"
            "Generate a strategy that directly addresses this request using "
            "the user-specific news and market data provided above. "
            "If the request mentions specific assets, time horizons, or "
            "market regions, incorporate them into the positions.\n"
        )
        raw_text = await self._agent_run(prompt)
        for attempt in range(3):
            try:
                payload = self._extract_json(raw_text)
                parsed = Strategy.model_validate(json.loads(payload))
                await self._log_rl_trace(context, prompt, raw_text, parsed)
                return parsed
            except (json.JSONDecodeError, ValidationError, StrategyGenerationError) as exc:
                if attempt == 2:
                    logger.warning("user_request_strategy_fallback", error=str(exc))
                    parsed = self._default_strategy(context)
                    parsed.metadata["user_request"] = user_request
                    parsed.metadata["fallback"] = True
                    await self._log_rl_trace(context, prompt, raw_text, parsed, error=str(exc))
                    return parsed
                raw_text = await self._repair_json(raw_text)
        raise StrategyGenerationError("Unexpected strategy generation flow.")

    async def generate_strategy_summary(self, strategy: Strategy, user_request: Optional[str] = None) -> str:
        """Use the LLM to produce a readable natural-language explanation of a strategy."""
        from agents.common import llm_chat

        positions_desc = "\n".join(
            f"- {p.action.value.upper()} {p.asset}: {p.size_pct}% allocation, "
            f"stop-loss {p.stop_loss_pct}%, "
            f"{'take-profit ' + str(p.take_profit_pct) + '%' if p.take_profit_pct else 'no take-profit'}, "
            f"{p.time_horizon_days}-day horizon, confidence {p.confidence:.0%}"
            for p in strategy.positions
        )
        user_prompt = (
            f"Market regime: {strategy.market_regime.value}\n"
            f"Rationale: {strategy.rationale}\n"
            f"Positions:\n{positions_desc}\n"
            f"Total exposure: {strategy.risk_metrics.total_exposure_pct}%\n"
        )
        if user_request:
            user_prompt += f"\nThe user originally asked: \"{user_request}\"\n"
        user_prompt += "\nProvide your summary:"
        system_prompt = (
            "You are a financial advisor assistant. Explain the trading strategy "
            "in clear, concise natural language (4-6 sentences). Include the overall thesis, "
            "key positions, risk profile, and time horizon."
        )
        result = await llm_chat(self.settings, system_prompt, user_prompt)
        return result if result.strip() else strategy.rationale

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
        strategy = await self.generate_strategy(context)
        strategy.analysis_id = baseline_analysis.analysis_id
        strategy.metadata["origin"] = "post_feedback"
        strategy.metadata["parent_strategy_id"] = baseline_strategy.strategy_id
        strategy.data_sources_used = list(
            set(strategy.data_sources_used + ["market_analysis", "user_feedback", "baseline_strategy"])
        )
        return strategy

    async def explain_strategy_and_backtest(
        self,
        strategy: Strategy,
        reward: Optional[RewardSignal] = None,
        query: Optional[str] = None,
    ) -> str:
        """Natural-language explanation of strategy + quantitative backtest (Analysis UI)."""
        from agents.common import llm_chat

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
        return (
            f"Strategy on {strategy.market_regime.value}: {strategy.rationale[:240]}. "
            + (f"Backtest terminal reward={reward.terminal_reward:.3f}." if reward else "No backtest.")
        )

    async def _log_rl_trace(self, context: dict, prompt: str, raw_response: str, strategy: Strategy, error: Optional[str] = None) -> None:
        trace_payload = {
            "trace_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "state_context": context,
            "prompt": prompt,
            "raw_response": raw_response,
            "action_strategy": strategy.model_dump(mode="json"),
            "model_metadata": {
                "provider": self.settings.llm_provider,
                "model": self.settings.llm_model,
                "temperature": self.settings.llm_temperature,
                "base_url": self.settings.llm_base_url,
            },
            "error": error,
        }
        await self.memory.working.set("rl_trace_log_latest", trace_payload, category="rl")

    @staticmethod
    def _dump(value):
        return None if value is None else (value.model_dump(mode="json") if hasattr(value, "model_dump") else value)

