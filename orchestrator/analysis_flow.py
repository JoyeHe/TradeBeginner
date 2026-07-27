"""Analysis baseline + intuition feedback flows (Flow 1 & Flow 2)."""

from __future__ import annotations

import json
from typing import Optional

import structlog
from pydantic import ValidationError

from agents.agent1_news import Agent1News
from agents.agent2_market import Agent2Market
from agents.agent3_strategy import Agent3Strategy, StrategyGenerationError
from agents.agent5_behavior import Agent5BehaviorStub
from agents.agent6_backtest import Agent6Backtest
from memory.analysis_store import AnalysisSession, AnalysisStore
from schemas.analysis import FlowType, MarketAnalysis
from schemas.analysis_feedback import AnalysisFeedback
from schemas.strategy import Strategy
from schemas.strategy_library import LibraryOrigin, StrategyLibraryEntry
from tools.analysis_helpers import (
    apply_feedback_heuristic,
    build_market_evidence_from_lookup,
    build_market_evidence_from_snapshot,
    build_news_bundle_from_search,
    heuristic_market_analysis,
)

logger = structlog.get_logger(__name__)

ANALYSIS_SYSTEM = """
You are a financial market analyst. Given news sentiment and market evidence,
produce a structured JSON MarketAnalysis: sentiment (overall_score -1..1, key_themes,
supporting_headline_ids), outlook (direction bullish/bearish/neutral/mixed, horizon_days,
confidence, affected_assets, narrative), and risks list.
Return ONLY valid JSON matching the provided schema.
"""


class AnalysisFlow:
    """Orchestrates two-flow analysis → strategy → backtest pipeline."""

    def __init__(
        self,
        agent1: Agent1News,
        agent2: Agent2Market,
        agent3: Agent3Strategy,
        agent5: Agent5BehaviorStub,
        agent6: Agent6Backtest,
        store: AnalysisStore,
        memory,
    ):
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent3 = agent3
        self.agent5 = agent5
        self.agent6 = agent6
        self.store = store
        self.memory = memory

    async def _fetch_inputs(self, query: str, tickers: Optional[list[str]] = None) -> tuple[dict, dict]:
        news = await self.agent1.user_search(query)
        news_bundle = build_news_bundle_from_search(news)
        if tickers:
            market_raw = await self.agent2.user_lookup(" ".join(tickers))
            market_evidence = build_market_evidence_from_lookup(market_raw)
        else:
            snapshot = await self.memory.working.get("market_snapshot")
            market_evidence = build_market_evidence_from_snapshot(snapshot)
        return news_bundle, market_evidence

    async def generate_market_analysis(
        self,
        query: str,
        news_bundle: dict,
        market_evidence: dict,
        user_id: str = "default",
        flow_type: FlowType = FlowType.BASELINE,
        parent_analysis_id: Optional[str] = None,
        preference_hints: Optional[list[dict]] = None,
    ) -> MarketAnalysis:
        schema_json = json.dumps(MarketAnalysis.model_json_schema(), indent=2)
        hints = ""
        if preference_hints:
            hints = "\n## User historical corrections\n" + json.dumps(preference_hints, default=str, indent=2)
        prompt = (
            f"{ANALYSIS_SYSTEM}\n\n"
            f"Query: {query}\n\n"
            f"News bundle:\n{json.dumps(news_bundle, default=str, indent=2)}\n\n"
            f"Market evidence:\n{json.dumps(market_evidence, default=str, indent=2)}\n"
            f"{hints}\n\n"
            f"Schema:\n{schema_json}\n"
            "Set flow_type to \"baseline\" or \"revised\" as appropriate."
        )
        raw = await self.agent3._agent_run(prompt)
        if raw:
            try:
                payload = self.agent3._extract_json(raw)
                data = json.loads(payload)
                data.setdefault("user_id", user_id)
                data.setdefault("query_context", query)
                data.setdefault("news_bundle", news_bundle)
                data.setdefault("market_evidence", market_evidence)
                data["flow_type"] = flow_type.value
                if parent_analysis_id:
                    data["parent_analysis_id"] = parent_analysis_id
                return MarketAnalysis.model_validate(data)
            except (json.JSONDecodeError, ValidationError, StrategyGenerationError) as exc:
                logger.warning("analysis_llm_parse_failed", error=str(exc))
        return heuristic_market_analysis(
            query, news_bundle, market_evidence, user_id=user_id, flow_type=flow_type, parent_analysis_id=parent_analysis_id
        )

    async def generate_strategy_from_analysis(self, analysis: MarketAnalysis) -> Strategy:
        context = await self.agent3.gather_context()
        context["market_analysis"] = analysis.model_dump(mode="json")
        context["user_request"] = analysis.query_context
        strategy = await self.agent3.generate_strategy(context)
        strategy.analysis_id = analysis.analysis_id
        strategy.rationale = f"{analysis.outlook.narrative}\n\n{strategy.rationale}"
        strategy.data_sources_used = list(set(strategy.data_sources_used + analysis.data_sources + ["market_analysis"]))
        return strategy

    async def run_baseline(self, query: str, tickers: Optional[list[str]] = None, user_id: str = "default") -> AnalysisSession:
        news_bundle, market_evidence = await self._fetch_inputs(query, tickers)
        analysis = await self.generate_market_analysis(
            query, news_bundle, market_evidence, user_id=user_id, flow_type=FlowType.BASELINE
        )
        session = self.store.create_session(analysis)
        session.status = "generating_strategy"
        strategy = await self.generate_strategy_from_analysis(analysis)
        session.baseline_strategy = strategy
        session.status = "evaluating"
        try:
            session.baseline_reward = await self.agent6.evaluate_strategy(strategy)
        except Exception as exc:
            logger.warning("baseline_backtest_failed", error=str(exc))
        try:
            session.baseline_explanation = await self.agent3.explain_strategy_and_backtest(
                strategy, session.baseline_reward, query=query
            )
        except Exception as exc:
            logger.warning("baseline_explanation_failed", error=str(exc))
            session.baseline_explanation = strategy.rationale
        session.status = "ready"
        entry = StrategyLibraryEntry(
            user_id=user_id,
            source_analysis_id=analysis.analysis_id,
            strategy=strategy,
            backtest_reward=None if session.baseline_reward is None else session.baseline_reward.terminal_reward,
            backtest_metrics={} if session.baseline_reward is None else session.baseline_reward.model_dump(mode="json"),
            tags=[analysis.outlook.direction.value, analysis.flow_type.value],
            origin=LibraryOrigin.BASELINE,
        )
        self.store.add_library_entry(entry)
        strategy.library_entry_id = entry.entry_id
        await self.memory.semantic.store_knowledge(
            content=f"{analysis.outlook.narrative} | sentiment={analysis.sentiment.overall_score}",
            metadata={"analysis_id": analysis.analysis_id, "user_id": user_id, "origin": "baseline"},
            category="analysis",
        )
        await self.memory.working.set(f"analysis_session:{analysis.analysis_id}", self.session_to_dict(session), category="analysis")
        return session

    async def submit_feedback(self, analysis_id: str, feedback: AnalysisFeedback) -> AnalysisSession:
        session = self.store.get_session(analysis_id)
        if session is None:
            raise KeyError(f"analysis session not found: {analysis_id}")
        self.store.add_feedback(feedback)
        session.status = "revising_analysis"

        hints = self.store.retrieve_similar_corrections(session.baseline_analysis.query_context, user_id=feedback.user_id)
        correction_text = feedback.free_text or ""
        for dim in feedback.dimension_feedbacks:
            if dim.correction:
                correction_text += f" [{dim.dimension.value}: {dim.correction}]"

        if feedback.overall_verdict == "agree" and not correction_text.strip():
            revised = session.baseline_analysis.model_copy(deep=True)
            revised.flow_type = FlowType.REVISED
            revised.parent_analysis_id = analysis_id
            revised.metadata["user_agreed"] = True
        elif self.agent3.settings.llm_api_key and correction_text.strip():
            news_bundle = session.baseline_analysis.news_bundle
            market_evidence = session.baseline_analysis.market_evidence
            revised = await self.generate_market_analysis(
                session.baseline_analysis.query_context,
                news_bundle,
                market_evidence,
                user_id=feedback.user_id,
                flow_type=FlowType.REVISED,
                parent_analysis_id=analysis_id,
                preference_hints=hints
                + [{"prior_feedback": correction_text, "overall_verdict": feedback.overall_verdict}],
            )
        else:
            revised = apply_feedback_heuristic(session.baseline_analysis, correction_text or feedback.overall_verdict)

        session.revised_analysis = revised
        session.status = "generating_strategy"
        strategy = await self.agent3.generate_strategy_from_feedback(
            baseline_analysis=session.baseline_analysis,
            baseline_strategy=session.baseline_strategy,
            baseline_reward=session.baseline_reward,
            feedback_text=correction_text,
            overall_verdict=feedback.overall_verdict,
            preference_hints=hints,
        )
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
        session.status = "ready"

        await self.agent5.on_analysis_feedback(session.baseline_analysis, revised, feedback)
        self.store.record_preference(
            {
                "user_id": feedback.user_id,
                "analysis_id": analysis_id,
                "query_context": session.baseline_analysis.query_context,
                "free_text": feedback.free_text,
                "correction_summary": correction_text,
                "overall_verdict": feedback.overall_verdict,
            }
        )
        entry = StrategyLibraryEntry(
            user_id=feedback.user_id,
            source_analysis_id=revised.analysis_id,
            strategy=strategy,
            backtest_reward=None if session.feedback_reward is None else session.feedback_reward.terminal_reward,
            backtest_metrics={} if session.feedback_reward is None else session.feedback_reward.model_dump(mode="json"),
            tags=[revised.outlook.direction.value, "post_feedback"],
            origin=LibraryOrigin.POST_FEEDBACK,
        )
        self.store.add_library_entry(entry)
        strategy.library_entry_id = entry.entry_id
        await self.memory.semantic.store_knowledge(
            content=f"User correction: {correction_text} | Revised: {revised.outlook.narrative}",
            metadata={"analysis_id": revised.analysis_id, "user_id": feedback.user_id, "origin": "post_feedback"},
            category="analysis_correction",
        )
        await self.memory.working.set(f"analysis_session:{analysis_id}", self.session_to_dict(session), category="analysis")
        return session

    def session_to_dict(self, session: AnalysisSession) -> dict:
        def _dump(obj):
            return None if obj is None else (obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj)

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

    def get_session_payload(self, analysis_id: str) -> dict:
        session = self.store.get_session(analysis_id)
        if session is None:
            return {"status": "not_found"}
        return self.session_to_dict(session)

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
            try:
                session.comparison_narrative = await self.agent3.compare_strategy_outcomes(
                    session.baseline_strategy,
                    session.baseline_reward,
                    session.feedback_strategy,
                    session.feedback_reward,
                    feedback_text=fb_text,
                )
            except Exception as exc:
                logger.warning("comparison_narrative_failed", error=str(exc))
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
