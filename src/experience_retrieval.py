from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

import numpy as np

from src.experience_store import ExperienceNode

logger = logging.getLogger(__name__)


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


def dual_channel_score(
    query_embedding: np.ndarray,
    base_activation: Dict[str, float],
    node: ExperienceNode,
    alpha: float,
) -> tuple[float, float, float]:
    s_sem = float(np.dot(query_embedding, node.query_embedding))
    s_str = _sparse_cosine(base_activation, node.activation_map)
    return alpha * s_sem + (1.0 - alpha) * s_str, s_sem, s_str


def dual_channel_match(
    query_embedding: np.ndarray,
    base_activation: Dict[str, float],
    experience_nodes: List[ExperienceNode],
    alpha: float,
    top_k: int,
) -> List[Tuple[ExperienceNode, float]]:
    if not experience_nodes:
        return []

    scored: List[Tuple[ExperienceNode, float]] = []
    for node in experience_nodes:
        s_k, _, _ = dual_channel_score(
            query_embedding=query_embedding,
            base_activation=base_activation,
            node=node,
            alpha=alpha,
        )
        scored.append((node, s_k))

    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


def adaptive_fusion(
    seed_entity_hash_ids: List[str],
    seed_entity_scores: List[float],
    matched: List[Tuple[ExperienceNode, float]],
    session_counter: int,
    beta: float,
    tau_s: float,
    mu: float,
    fusion_epsilon: float,
    use_temporal_decay: bool = False,
    max_exp_entities: int = 0,
) -> Dict[str, float]:
    a_base = dict(zip(seed_entity_hash_ids, seed_entity_scores))

    if not matched:
        return a_base


    scores_arr = np.array([s for _, s in matched], dtype=np.float64)
    scores_scaled = scores_arr / tau_s
    scores_scaled -= scores_scaled.max()
    exp_scores = np.exp(scores_scaled)
    sm_weights = exp_scores / exp_scores.sum()


    final_weights: List[float] = []
    for (node, _), sm_w in zip(matched, sm_weights):
        if use_temporal_decay:
            delta_t = max(session_counter - node.timestamp, 0)
            decay = math.exp(-mu * delta_t)
        else:
            decay = 1.0
        final_weights.append(float(sm_w) * decay * node.confidence)


    experience_contributions: Dict[str, float] = {}
    for (node, _), w_k in zip(matched, final_weights):
        act_map = node.activation_map
        if max_exp_entities > 0 and len(act_map) > max_exp_entities:


            act_map = dict(
                sorted(act_map.items(), key=lambda kv: kv[1], reverse=True)[
                    :max_exp_entities
                ]
            )
        for ehid, act_score in act_map.items():
            contrib = beta * w_k * act_score
            experience_contributions[ehid] = (
                experience_contributions.get(ehid, 0.0) + contrib
            )

    a_fused: Dict[str, float] = dict(a_base)
    for ehid, contrib in experience_contributions.items():
        a_fused[ehid] = a_fused.get(ehid, 0.0) + contrib


    max_val = max(a_fused.values())
    if max_val > 0:
        a_fused = {k: v / max_val for k, v in a_fused.items()}


    a_fused = {k: v for k, v in a_fused.items() if v >= fusion_epsilon}

    return a_fused


def consolidate_memory(
    nodes: List[ExperienceNode],
    alpha: float,
    tau_merge: float,
) -> tuple[List[ExperienceNode], int]:
    if len(nodes) <= 1:
        return list(nodes), 0

    merged_ids: set = set()
    result: List[ExperienceNode] = []

    for i, node_i in enumerate(nodes):
        if node_i.node_id in merged_ids:
            continue

        current = node_i
        for node_j in nodes[i + 1:]:
            if node_j.node_id in merged_ids:
                continue


            s_sem = float(np.dot(current.query_embedding, node_j.query_embedding))
            s_str = _sparse_cosine(current.activation_map, node_j.activation_map)
            sim = alpha * s_sem + (1.0 - alpha) * s_str

            if sim < tau_merge:
                continue


            winner = current if current.confidence >= node_j.confidence else node_j
            current = ExperienceNode(
                node_id=winner.node_id,
                query_embedding=winner.query_embedding,
                activation_map=winner.activation_map,
                reasoning_embedding=winner.reasoning_embedding,
                chain_summary=winner.chain_summary,
                answer=winner.answer,
                timestamp=max(current.timestamp, node_j.timestamp),
                confidence=max(current.confidence, node_j.confidence),
                question_id=getattr(winner, "question_id", ""),
                decomposition_ids=list(getattr(winner, "decomposition_ids", []) or []),
            )
            merged_ids.add(node_j.node_id)

        result.append(current)

    n_merged = len(merged_ids)
    if n_merged:
        logger.info(
            "[consolidate] %d node(s) merged → memory size %d → %d",
            n_merged, len(nodes), len(result),
        )
    return result, n_merged
