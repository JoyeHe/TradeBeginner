"""In-process store for analysis sessions, feedback, and strategy library."""

from __future__ import annotations

from typing import Optional

import structlog

from schemas.analysis import MarketAnalysis
from schemas.analysis_feedback import AnalysisFeedback
from schemas.rewards import RewardSignal
from schemas.strategy import Strategy
from schemas.strategy_library import StrategyLibraryEntry

logger = structlog.get_logger(__name__)


class AnalysisSession:
    """One baseline + optional revised analysis workflow."""

    def __init__(self, analysis: MarketAnalysis):
        self.baseline_analysis: MarketAnalysis = analysis
        self.revised_analysis: Optional[MarketAnalysis] = None
        self.baseline_strategy: Optional[Strategy] = None
        self.feedback_strategy: Optional[Strategy] = None
        self.baseline_reward: Optional[RewardSignal] = None
        self.feedback_reward: Optional[RewardSignal] = None
        self.baseline_explanation: Optional[str] = None
        self.feedback_explanation: Optional[str] = None
        self.comparison_narrative: Optional[str] = None
        self.feedbacks: list[AnalysisFeedback] = []
        self.status: str = "analysis_ready"

    @property
    def analysis_id(self) -> str:
        return self.baseline_analysis.analysis_id


class AnalysisStore:
    """Memory-backed analysis session and strategy library store."""

    def __init__(self) -> None:
        self._sessions: dict[str, AnalysisSession] = {}
        self._library: list[StrategyLibraryEntry] = []
        self._preference_records: list[dict] = []

    def create_session(self, analysis: MarketAnalysis) -> AnalysisSession:
        session = AnalysisSession(analysis)
        self._sessions[analysis.analysis_id] = session
        return session

    def get_session(self, analysis_id: str) -> Optional[AnalysisSession]:
        return self._sessions.get(analysis_id)

    def add_feedback(self, feedback: AnalysisFeedback) -> None:
        session = self._sessions.get(feedback.analysis_id)
        if session is not None:
            session.feedbacks.append(feedback)

    def add_library_entry(self, entry: StrategyLibraryEntry) -> None:
        self._library.append(entry)

    def list_library(self, user_id: str = "default", limit: int = 50) -> list[StrategyLibraryEntry]:
        items = [e for e in self._library if e.user_id == user_id]
        return sorted(items, key=lambda x: x.created_at, reverse=True)[:limit]

    def record_preference(self, record: dict) -> None:
        self._preference_records.append(record)

    def retrieve_similar_corrections(self, query: str, user_id: str = "default", n: int = 5) -> list[dict]:
        """Simple keyword overlap retrieval for personalization."""
        tokens = set(query.lower().split())
        scored: list[tuple[float, dict]] = []
        for rec in self._preference_records:
            if rec.get("user_id", "default") != user_id:
                continue
            text = (rec.get("query_context", "") + " " + rec.get("free_text", "") + " " + rec.get("correction_summary", "")).lower()
            overlap = sum(1 for t in tokens if t in text)
            if overlap > 0:
                scored.append((float(overlap), rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:n]]

    def preference_stats(self, user_id: str = "default") -> dict:
        user_recs = [r for r in self._preference_records if r.get("user_id", "default") == user_id]
        return {
            "total_preference_records": len(user_recs),
            "library_entries": len([e for e in self._library if e.user_id == user_id]),
            "active_sessions": len(self._sessions),
        }
