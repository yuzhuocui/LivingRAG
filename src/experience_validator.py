from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.experience_store import ExperienceNode, ReasoningChain

logger = logging.getLogger(__name__)


@dataclass
class EQSResult:
    gs: Optional[float]
    ns: float
    phi: float
    passes: bool


def _sparse_cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    norm1 = math.sqrt(sum(w * w for w in v1.values()))
    norm2 = math.sqrt(sum(w * w for w in v2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def _split_claims(text: str) -> List[str]:
    import re

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def compute_grounding_score(
    chain: ReasoningChain,
    passage_texts: Dict[str, str],
    nli_model,
    tau_nli: float,
    embedding_model=None,
    top_k_evidence: int = 5,
) -> tuple[float, List[str]]:
    if nli_model is None:
        return 1.0, []


    gs_text = chain.raw_text
    if chain.answer:
        gs_text = gs_text + " " + chain.answer
    claims = _split_claims(gs_text)
    if not claims:
        return 1.0, []

    passage_sentences: List[str] = []
    for passage in passage_texts.values():
        passage_sentences.extend(_split_claims(passage))
    if not passage_sentences:
        return 1.0, []


    if embedding_model is not None and len(passage_sentences) > top_k_evidence:
        claim_embs = embedding_model.encode(
            claims, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        )
        sent_embs = embedding_model.encode(
            passage_sentences, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        )
        sim_matrix = claim_embs @ sent_embs.T


        k = min(top_k_evidence, len(passage_sentences))
        all_pairs: List[tuple] = []
        pair_claim_idx: List[int] = []
        for i in range(len(claims)):
            top_k_idxs = np.argsort(sim_matrix[i])[-k:]
            for j in top_k_idxs:
                all_pairs.append((passage_sentences[j], claims[i]))
                pair_claim_idx.append(i)
    else:
        all_pairs = [(sent, claim) for claim in claims for sent in passage_sentences]
        pair_claim_idx = [i for i in range(len(claims)) for _ in passage_sentences]


    ENTAIL_IDX = 2

    all_scores = nli_model.predict(all_pairs, apply_softmax=True, show_progress_bar=False)

    claim_best_entail = [0.0] * len(claims)
    for pair_i, scores in zip(pair_claim_idx, all_scores):
        ent = float(scores[ENTAIL_IDX])
        if ent > claim_best_entail[pair_i]:
            claim_best_entail[pair_i] = ent

    grounded_count = 0
    ungrounded: List[str] = []
    for i, claim in enumerate(claims):
        if claim_best_entail[i] >= tau_nli:
            grounded_count += 1
        else:
            ungrounded.append(claim)

    gs = grounded_count / len(claims)
    return gs, ungrounded


def compute_novelty_score(
    query_embedding: np.ndarray,
    base_activation: Dict[str, float],
    experience_nodes: List[ExperienceNode],
    alpha: float,
) -> float:
    if not experience_nodes:
        return 1.0

    max_sim = 0.0
    for node in experience_nodes:

        s_sem = float(np.dot(query_embedding, node.query_embedding))

        s_str = _sparse_cosine(base_activation, node.activation_map)
        sim = alpha * s_sem + (1.0 - alpha) * s_str
        if sim > max_sim:
            max_sim = sim

    return 1.0 - max_sim


def run_eqs(
    chain: ReasoningChain,
    experience_nodes: List[ExperienceNode],
    query_embedding: np.ndarray,
    base_activation: Dict[str, float],
    alpha: float,
    tau_nli: float,
    tau_gs: float,
    tau_novel: float,
    passage_texts: Optional[Dict[str, str]] = None,
    nli_model=None,
    embedding_model=None,
    top_k_evidence: int = 5,
) -> EQSResult:

    ns = compute_novelty_score(query_embedding, base_activation, experience_nodes, alpha)
    if ns < tau_novel:

        return EQSResult(gs=None, ns=ns, phi=ns, passes=False)


    gs, _ = compute_grounding_score(
        chain, passage_texts or {}, nli_model, tau_nli,
        embedding_model=embedding_model,
        top_k_evidence=top_k_evidence,
    )

    phi = min(gs, ns)
    passes = gs >= tau_gs

    return EQSResult(gs=gs, ns=ns, phi=phi, passes=passes)
