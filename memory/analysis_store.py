"""In-process store for analysis sessions, feedback, and strategy library."""

from __future__ import annotations

from typing import Optional

import structlog

from memory.strategy_library_store import StrategyLibraryStore
from schemas.analysis import MarketAnalysis
from schemas.analysis_feedback import AnalysisFeedback
from schemas.strategy_library import StrategyLibraryEntry

logger = structlog.get_logger(__name__)


class AnalysisSession:
    """One baseline + optional revised analysis workflow."""

    def __init__(self, analysis: MarketAnalysis):
        self.baseline_analysis: MarketAnalysis = analysis
        self.revised_analysis: Optional[MarketAnalysis] = None
        self.baseline_strategy = None
        self.feedback_strategy = None
        self.baseline_reward = None
        self.feedback_reward = None
        self.baseline_explanation: Optional[str] = None
        self.feedback_explanation: Optional[str] = None
        self.comparison_narrative: Optional[str] = None
        self.material_change: Optional[bool] = None
        self.feedbacks: list[AnalysisFeedback] = []
        self.status: str = "analysis_ready"

    @property
    def analysis_id(self) -> str:
        return self.baseline_analysis.analysis_id


class AnalysisStore:
    """Memory-backed analysis session store with delegated strategy library."""

    def __init__(self, library: Optional[StrategyLibraryStore] = None) -> None:
        self._sessions: dict[str, AnalysisSession] = {}
        self._library_fallback: list[StrategyLibraryEntry] = []
        self.library = library
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

    def add_library_entry(self, entry: StrategyLibraryEntry) -> StrategyLibraryEntry:
        if self.library is not None:
            return self.library.add_entry(entry)
        self._library_fallback.append(entry)
        return entry

    async def upsert_library_entry(self, entry: StrategyLibraryEntry) -> StrategyLibraryEntry:
        if self.library is not None:
            return await self.library.upsert(entry)
        return self.add_library_entry(entry)

    async def promote_library_entry(self, entry: StrategyLibraryEntry) -> StrategyLibraryEntry:
        if self.library is not None:
            return await self.library.promote_or_merge(entry)
        return self.add_library_entry(entry)

    def list_library(
        self,
        user_id: str = "default",
        limit: int = 50,
        tag: Optional[str] = None,
        min_reward: Optional[float] = None,
        include_global: bool = True,
    ) -> list[StrategyLibraryEntry]:
        if self.library is not None:
            return self.library.list_entries(
                user_id=user_id,
                limit=limit,
                tag=tag,
                min_reward=min_reward,
                include_global=include_global,
            )
        items = [e for e in self._library_fallback if e.user_id == user_id or (include_global and e.user_id == "*")]
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
        prefs = [r for r in self._preference_records if r.get("user_id", "default") == user_id]
        return {
            "total_preference_records": len(prefs),
            "library_entries": len(self.list_library(user_id=user_id, limit=1000)),
        }
