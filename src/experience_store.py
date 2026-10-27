from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReasoningChain:

    query: str
    passages_used: List[str]
    answer: str
    raw_text: str
    parse_success: bool = True

    @property
    def is_reliable(self) -> bool:
        return self.parse_success


def parse_reasoning_chain(
    content: str,
    reasoning_text: str,
    question: str,
    passage_hash_ids: List[str],
) -> ReasoningChain:
    answer = content.strip()
    parse_success = bool(answer)

    return ReasoningChain(
        query=question,
        passages_used=passage_hash_ids,
        answer=answer,
        raw_text=reasoning_text,
        parse_success=parse_success,
    )


@dataclass
class ExperienceNode:

    node_id: str
    query_embedding: np.ndarray
    activation_map: Dict[str, float]
    reasoning_embedding: np.ndarray
    chain_summary: str
    answer: str
    timestamp: int
    confidence: float
    question_id: str = ""
    decomposition_ids: List[str] = field(default_factory=list)


    _match_score: float = field(default=0.0, compare=False, repr=False)


    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "query_embedding": self.query_embedding.tolist(),
            "activation_map": self.activation_map,
            "reasoning_embedding": self.reasoning_embedding.tolist(),
            "chain_summary": self.chain_summary,
            "answer": self.answer,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "question_id": self.question_id,
            "decomposition_ids": self.decomposition_ids,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperienceNode":
        return cls(
            node_id=d["node_id"],
            query_embedding=np.array(d["query_embedding"], dtype=np.float32),
            activation_map=d["activation_map"],
            reasoning_embedding=np.array(d["reasoning_embedding"], dtype=np.float32),
            chain_summary=d["chain_summary"],
            answer=d["answer"],
            timestamp=d["timestamp"],
            confidence=float(d["confidence"]),
            question_id=d.get("question_id", ""),
            decomposition_ids=list(d.get("decomposition_ids", [])),
        )


class ExperienceStore:

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._nodes: List[ExperienceNode] = []


        self._lock = threading.Lock()
        self._load()


    def add_experience(self, node: ExperienceNode) -> None:
        with self._lock:
            self._nodes.append(node)

    def get_all(self) -> List[ExperienceNode]:
        with self._lock:
            return list(self._nodes)

    def __len__(self) -> int:
        with self._lock:
            return len(self._nodes)

    def __repr__(self) -> str:
        return f"ExperienceStore(path={self.db_path!r}, n={len(self._nodes)})"


    def _save(self, entity_id_to_text: dict | None = None) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump([n.to_dict() for n in self._nodes], f, ensure_ascii=False)
        self._save_readable(entity_id_to_text=entity_id_to_text)

    _READABLE_SKIP = {"query_embedding", "reasoning_embedding"}

    def _save_readable(self, entity_id_to_text: dict | None = None) -> None:
        readable_path = self.db_path.replace(".json", "_readable.json")
        readable_nodes = []
        for n in self._nodes:
            d = {k: v for k, v in n.to_dict().items() if k not in self._READABLE_SKIP}
            if entity_id_to_text:
                d["activation_map"] = {
                    entity_id_to_text.get(eid, eid): score
                    for eid, score in d["activation_map"].items()
                }
            readable_nodes.append(d)
        with open(readable_path, "w", encoding="utf-8") as f:
            json.dump(readable_nodes, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        if not os.path.exists(self.db_path):
            logger.info(f"[ExperienceStore] No existing memory at {self.db_path}. Starting fresh.")
            return
        with open(self.db_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._nodes = [ExperienceNode.from_dict(d) for d in raw]
        logger.info(
            f"[ExperienceStore] Loaded {len(self._nodes)} experience nodes from {self.db_path}"
        )

    def consolidate(self, alpha: float, tau_merge: float) -> int:
        from src.experience_retrieval import consolidate_memory

        with self._lock:
            nodes_snapshot = list(self._nodes)

        pruned, n_merged = consolidate_memory(nodes_snapshot, alpha, tau_merge)

        if n_merged:
            with self._lock:
                self._nodes = pruned

        return n_merged

    def save(self, entity_id_to_text: dict | None = None) -> None:
        self._save(entity_id_to_text=entity_id_to_text)

    def load(self) -> None:
        self._load()


    @staticmethod
    def make_node_id() -> str:
        return uuid.uuid4().hex
