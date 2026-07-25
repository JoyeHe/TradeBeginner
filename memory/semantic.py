"""Semantic memory backed by ChromaDB and structured fact store."""

from __future__ import annotations

from typing import Any, Optional
import uuid

from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection

from memory.base import BaseMemory


class SemanticMemory(BaseMemory):
    """Semantic vector memory plus structured key/value facts."""

    def __init__(self, persist_dir: str) -> None:
        self._persist_dir = persist_dir
        self._collection: Optional[Collection] = None
        self._facts: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        """Initialize Chroma collection."""
        try:
            client = PersistentClient(path=self._persist_dir)
            self._collection = client.get_or_create_collection(name="tradebeginner_semantic")
        except Exception:
            self._collection = None

    async def store_knowledge(self, content: str, metadata: dict | None = None, category: str = "general") -> str:
        """Embed and store semantic knowledge."""
        metadata = dict(metadata or {})
        metadata["category"] = category
        knowledge_id = str(uuid.uuid4())
        if self._collection is not None:
            self._collection.add(ids=[knowledge_id], documents=[content], metadatas=[metadata])
        else:
            self._facts[knowledge_id] = {"content": content, "metadata": metadata}
        return knowledge_id

    async def query(self, query: str, n: int = 5, category: Optional[str] = None) -> list[dict]:
        """Semantic search by text query."""
        if self._collection is not None:
            where = {"category": category} if category else None
            result = self._collection.query(query_texts=[query], n_results=n, where=where)
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            dists = result.get("distances", [[]])[0]
            return [
                {"content": doc, "metadata": meta or {}, "similarity_score": 1.0 - float(dist or 0.0)}
                for doc, meta, dist in zip(docs, metas, dists)
            ]

        out: list[dict] = []
        for value in self._facts.values():
            if category and value["metadata"].get("category") != category:
                continue
            if query.lower() in str(value["content"]).lower():
                out.append({"content": value["content"], "metadata": value["metadata"], "similarity_score": 0.5})
        return out[:n]

    async def store_structured_fact(self, key: str, value: Any, category: str = "fact") -> None:
        """Store deterministic structured fact."""
        self._facts[key] = {"content": value, "metadata": {"category": category, "structured": True}}

    async def get_structured_fact(self, key: str) -> Any:
        """Get one structured fact by key."""
        entry = self._facts.get(key)
        return None if entry is None else entry["content"]

    async def get_facts_by_category(self, category: str) -> list[dict]:
        """Get structured facts by category."""
        return [{"key": k, **v} for k, v in self._facts.items() if v["metadata"].get("category") == category]

    async def update_knowledge(self, knowledge_id: str, content: str) -> None:
        """Update an existing knowledge document."""
        if self._collection is not None:
            self._collection.update(ids=[knowledge_id], documents=[content])
        elif knowledge_id in self._facts:
            self._facts[knowledge_id]["content"] = content

    async def delete_knowledge(self, knowledge_id: str) -> None:
        """Delete knowledge by ID."""
        if self._collection is not None:
            self._collection.delete(ids=[knowledge_id])
        self._facts.pop(knowledge_id, None)

    async def health(self) -> dict[str, Any]:
        """Return semantic memory health."""
        return {"backend": "chromadb" if self._collection is not None else "in_memory", "facts": len(self._facts)}

