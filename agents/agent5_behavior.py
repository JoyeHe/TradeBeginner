"""Agent 5 behavior cloning collector (stub)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
import uuid

from memory import MemoryManager
from schemas.analysis import MarketAnalysis
from schemas.analysis_feedback import AnalysisFeedback
from schemas.strategy import Strategy


class BehaviorDataCollector:
    """Collect preference data from user strategy interactions."""

    def __init__(self, memory: MemoryManager):
        self.memory = memory
        self._records: list[dict] = []

    async def record_modification(self, original: Strategy, modified: Strategy, user_notes: Optional[str] = None) -> str:
        record_id = str(uuid.uuid4())
        rec = {
            "record_id": record_id,
            "type": "behavior_cloning",
            "modification_type": "edited",
            "timestamp": datetime.utcnow().isoformat(),
            "original": original.model_dump(mode="json"),
            "modified": modified.model_dump(mode="json"),
            "user_notes": user_notes,
        }
        self._records.append(rec)
        await self.memory.working.set(f"behavior:{record_id}", rec, category="behavior")
        return record_id

    async def record_approval_without_modification(self, strategy: Strategy) -> str:
        record_id = str(uuid.uuid4())
        rec = {
            "record_id": record_id,
            "type": "behavior_cloning",
            "modification_type": "none",
            "timestamp": datetime.utcnow().isoformat(),
            "original": strategy.model_dump(mode="json"),
            "modified": strategy.model_dump(mode="json"),
        }
        self._records.append(rec)
        return record_id

    async def record_rejection(self, strategy: Strategy, reason: Optional[str] = None) -> str:
        record_id = str(uuid.uuid4())
        rec = {
            "record_id": record_id,
            "type": "behavior_cloning",
            "modification_type": "rejected",
            "timestamp": datetime.utcnow().isoformat(),
            "original": strategy.model_dump(mode="json"),
            "modified": None,
            "reason": reason,
        }
        self._records.append(rec)
        return record_id

    async def get_training_data(self, min_records: int = 100) -> list[dict]:
        if len(self._records) < min_records:
            return []
        return [
            {
                "original_prompt": r["original"],
                "original_output": r["original"],
                "modified_output": r["modified"],
                "label": r["modification_type"],
            }
            for r in self._records
        ]

    async def get_statistics(self) -> dict:
        behavior_records = [r for r in self._records if r.get("type") == "behavior_cloning"]
        total = len(behavior_records)
        modified = sum(1 for r in behavior_records if r.get("modification_type") == "edited")
        rejected = sum(1 for r in behavior_records if r.get("modification_type") == "rejected")
        return {
            "total_records": total,
            "modification_rate": (modified / total) if total else 0.0,
            "rejection_rate": (rejected / total) if total else 0.0,
        }


class Agent5BehaviorStub:
    """Placeholder behavior cloning agent for future training pipeline."""

    def __init__(self, memory: MemoryManager):
        self.collector = BehaviorDataCollector(memory)

    async def on_strategy_approved(self, original: Strategy, final: Strategy, user_notes: Optional[str] = None):
        if original.strategy_id == final.strategy_id and original == final:
            await self.collector.record_approval_without_modification(original)
        else:
            await self.collector.record_modification(original, final, user_notes)

    async def on_strategy_rejected(self, strategy: Strategy, reason: Optional[str] = None):
        await self.collector.record_rejection(strategy, reason)

    async def on_analysis_feedback(
        self,
        original: MarketAnalysis,
        revised: MarketAnalysis,
        feedback: AnalysisFeedback,
    ) -> str:
        record_id = str(uuid.uuid4())
        rec = {
            "record_id": record_id,
            "type": "analysis_preference",
            "timestamp": datetime.utcnow().isoformat(),
            "original_analysis_id": original.analysis_id,
            "revised_analysis_id": revised.analysis_id,
            "feedback": feedback.model_dump(mode="json"),
            "original_outlook": original.outlook.model_dump(mode="json"),
            "revised_outlook": revised.outlook.model_dump(mode="json"),
        }
        self.collector._records.append(rec)
        await self.collector.memory.working.set(f"analysis_pref:{record_id}", rec, category="behavior")
        return record_id

    async def report(self) -> dict:
        stats = await self.collector.get_statistics()
        prefs = sum(1 for r in self.collector._records if r.get("type") == "analysis_preference")
        stats["analysis_preference_records"] = prefs
        return stats

