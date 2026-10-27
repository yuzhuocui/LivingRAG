from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from tqdm import tqdm

from src.LinearRAG import LinearRAG
from src.config import LivingRAGConfig
from src.experience_store import ExperienceNode, ExperienceStore, ReasoningChain, parse_reasoning_chain
from src.experience_validator import EQSResult, run_eqs
from src.experience_retrieval import dual_channel_match, adaptive_fusion, dual_channel_score
from src.question_template import mask_question_template

logger = logging.getLogger(__name__)


def _extract_key_sentences(raw_text: str, entity_texts: list, max_sentences: int = 10) -> str:
    import re


    parts = re.split(r"(?<=[.!?])\s+|\n+", raw_text.strip())
    sentences = [s.strip() for s in parts if len(s.strip()) > 20]

    if not sentences:
        return raw_text[:600]

    entity_texts_lower = [e.lower() for e in entity_texts if e]


    scored = [
        (sum(1 for e in entity_texts_lower if e in s.lower()), i, s)
        for i, s in enumerate(sentences)
    ]


    top = sorted(scored, key=lambda x: (-x[0], x[1]))[:max_sentences]

    top_ordered = sorted(top, key=lambda x: x[1])

    key_sentences = [s for _, _, s in top_ordered]
    return " ".join(f"({i + 1}) {s}" for i, s in enumerate(key_sentences))


_LIVING_RAG_SYSTEM_PROMPT = (
    "As an advanced reading comprehension assistant, your task is to analyze "
    "text passages and corresponding questions meticulously. "
    "Provide a concise, definitive answer only."
)


class LivingRAG(LinearRAG):

    def __init__(self, global_config: LivingRAGConfig) -> None:
        super().__init__(global_config)


        if global_config.exp_db_path:
            exp_db_path = global_config.exp_db_path
        else:
            exp_db_path = os.path.join(
                self.config.working_dir,
                self.config.dataset_name,
                "experience_memory.json",
            )
        self.experience_store = ExperienceStore(db_path=exp_db_path)
        logger.info(
            f"[LivingRAG] Experience memory initialised: {self.experience_store}"
        )
        self._scaffold_template_cache: dict[str, tuple[str, np.ndarray | None]] = {}
        self._scaffold_template_cache_lock = threading.Lock()


        self.session_counter: int = len(self.experience_store)


        self._writes_since_consolidate: int = 0


        from sentence_transformers.cross_encoder import CrossEncoder
        logger.info(f"[LivingRAG] Loading NLI model: {global_config.nli_model_path}")
        self.nli_model = CrossEncoder(
            global_config.nli_model_path,
            num_labels=3,
            tokenizer_args={"use_fast": False},
        )
        logger.info("[LivingRAG] NLI model loaded — EQS Stage 1 (GS) active.")

    @staticmethod
    def _decomposition_ids_from_question_id(question_id: str) -> list[str]:
        if not question_id:
            return []
        core = question_id.removeprefix("musique_")
        if "__" not in core:
            return []
        return [part for part in core.split("__", 1)[1].split("_") if part]

    @classmethod
    def _question_family_keys(cls, questions: list[dict]) -> list[str]:
        parent = list(range(len(questions)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        first_seen: dict[str, int] = {}
        for idx, question_info in enumerate(questions):
            qid = question_info.get("id", "")
            ids = cls._decomposition_ids_from_question_id(qid)
            for did in ids:
                if did in first_seen:
                    union(first_seen[did], idx)
                else:
                    first_seen[did] = idx

        return [f"family:{find(i)}" for i in range(len(questions))]

    @staticmethod
    def _source_question_from_summary(summary: str) -> str:
        match = re.search(r"^Q:\s*(.*?)\n", summary or "", re.S)
        return match.group(1).strip() if match else ""

    def _ensure_scaffold_template_cache(
        self,
        experience_nodes: list[ExperienceNode],
    ) -> dict[str, tuple[str, np.ndarray | None]]:
        missing: list[tuple[str, str]] = []
        with self._scaffold_template_cache_lock:
            for node in experience_nodes:
                if node.node_id in self._scaffold_template_cache:
                    continue
                source_q = self._source_question_from_summary(node.chain_summary)
                template = mask_question_template(source_q, None) if source_q else ""
                if template:
                    missing.append((node.node_id, template))
                else:
                    self._scaffold_template_cache[node.node_id] = ("", None)

        if missing:
            embeddings = self.config.embedding_model.encode(
                [template for _, template in missing],
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.config.batch_size,
            )
            embeddings = np.asarray(embeddings)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            with self._scaffold_template_cache_lock:
                for (node_id, template), emb in zip(missing, embeddings):
                    self._scaffold_template_cache.setdefault(
                        node_id,
                        (template, np.asarray(emb, dtype=np.float32)),
                    )

        with self._scaffold_template_cache_lock:
            return {
                node.node_id: self._scaffold_template_cache.get(node.node_id, ("", None))
                for node in experience_nodes
            }

    @staticmethod
    def _rank_in_hook_a(node_id: str, hook_a_experiences: list[dict]) -> int | None:
        short_id = node_id[:8]
        for exp in hook_a_experiences:
            if exp.get("node_id") == short_id:
                return exp.get("rank")
        return None

    def _scaffold_log_entry(
        self,
        node: ExperienceNode,
        rank: int,
        retrieval_result: dict,
        scaffold_score: float,
        legacy_score: float,
        semantic_score: float,
        activation_score: float,
        template_score: float,
        source_template: str = "",
    ) -> dict:
        query_metadata_set = set(retrieval_result.get("decomposition_ids", []) or [])
        hook_a_experiences = retrieval_result.get("hook_a_experiences", []) or []
        return {
            "rank": rank,
            "is_scaffold": True,
            "score": round(scaffold_score, 4),
            "legacy_score": round(legacy_score, 4),
            "semantic_score": round(semantic_score, 4),
            "activation_score": round(activation_score, 4),
            "template_score": round(template_score, 4),
            "conf": round(node.confidence, 4),
            "node_id": node.node_id[:8],
            "question_id": getattr(node, "question_id", ""),
            "decomposition_ids": list(getattr(node, "decomposition_ids", []) or []),
            "decomposition_overlap": sorted(
                query_metadata_set & set(getattr(node, "decomposition_ids", []) or [])
            ),
            "timestamp": node.timestamp,
            "activation_size": len(node.activation_map),
            "rank_in_hook_a": self._rank_in_hook_a(node.node_id, hook_a_experiences),
            "source_template": source_template,
            "answer": node.answer,
            "summary": node.chain_summary,
        }

    def _select_template_scaffold(
        self,
        retrieval_result: dict,
        experience_nodes: list[ExperienceNode],
    ) -> tuple[ExperienceNode | None, dict | None, str]:
        if not experience_nodes:
            return None, None, ""

        weight = max(0.0, min(1.0, float(self.config.scaffold_template_weight)))
        current_template = mask_question_template(
            retrieval_result.get("question", ""),
            retrieval_result.get("seed_entities", []),
        )
        if current_template:
            current_template_embedding = self.config.embedding_model.encode(
                current_template,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.config.batch_size,
            )
        else:
            current_template_embedding = None

        template_cache = self._ensure_scaffold_template_cache(experience_nodes)
        query_embedding = retrieval_result.get("query_embedding")
        base_activation = retrieval_result.get("base_activation", {}) or {}

        best: tuple[float, ExperienceNode, float, float, float, float, str] | None = None
        for node in experience_nodes:
            legacy_score, semantic_score, activation_score = dual_channel_score(
                query_embedding=query_embedding,
                base_activation=base_activation,
                node=node,
                alpha=self.config.exp_alpha,
            )
            source_template, source_template_embedding = template_cache.get(
                node.node_id, ("", None)
            )
            if current_template_embedding is not None and source_template_embedding is not None:
                template_score = float(np.dot(current_template_embedding, source_template_embedding))
            else:
                template_score = 0.0

            scaffold_score = (1.0 - weight) * legacy_score + weight * template_score
            item = (
                scaffold_score,
                node,
                legacy_score,
                semantic_score,
                activation_score,
                template_score,
                source_template,
            )
            if best is None or item[0] > best[0]:
                best = item

        if best is None:
            return None, None, current_template

        scaffold_score, node, legacy_score, semantic_score, activation_score, template_score, source_template = best
        return (
            node,
            self._scaffold_log_entry(
                node=node,
                rank=1,
                retrieval_result=retrieval_result,
                scaffold_score=scaffold_score,
                legacy_score=legacy_score,
                semantic_score=semantic_score,
                activation_score=activation_score,
                template_score=template_score,
                source_template=source_template,
            ),
            current_template,
        )

    def _build_hook_b_scaffold(
        self,
        retrieval_result: dict,
        experience_nodes: list[ExperienceNode],
    ) -> tuple[str, dict]:
        best_exp = retrieval_result.get("best_experience")
        weight = max(0.0, min(1.0, float(self.config.scaffold_template_weight)))
        info = {
            "hook_b_scaffold_used": False,
            "hook_b_scaffold_reason": "no_experience",
            "hook_b_scaffold_chars": 0,
            "hook_b_scaffold_source_question_id": "",
            "hook_b_scaffold_node_id": "",
            "hook_b_scaffold_score": 0.0,
            "hook_b_scaffold_legacy_score": 0.0,
            "hook_b_scaffold_semantic_score": 0.0,
            "hook_b_scaffold_activation_score": 0.0,
            "hook_b_scaffold_template_score": 0.0,
            "hook_b_scaffold_rank_in_hook_a": None,
            "hook_b_scaffold_current_template": "",
            "hook_b_scaffold_template_weight": weight,
            "hook_b_scaffold_experience": None,
        }

        if not experience_nodes:
            return "", info

        selected_exp: ExperienceNode | None = None
        selected_info: dict | None = None
        reason = "legacy"

        if weight <= 0.0:


            if best_exp is None:
                info["hook_b_scaffold_reason"] = "no_hook_a_experience"
                return "", info
            experiences = retrieval_result.get("hook_a_experiences", []) or []
            legacy_score = float(retrieval_result.get("hook_a_best_score", 0.0))
            selected_exp = best_exp
            selected_info = self._scaffold_log_entry(
                node=best_exp,
                rank=1,
                retrieval_result=retrieval_result,
                scaffold_score=legacy_score,
                legacy_score=legacy_score,
                semantic_score=0.0,
                activation_score=0.0,
                template_score=0.0,
                source_template="",
            )
            if experiences:
                selected_info.update({
                    "answer": experiences[0].get("answer", best_exp.answer),
                    "summary": experiences[0].get("summary", best_exp.chain_summary),
                })
        else:
            selected_exp, selected_info, current_template = self._select_template_scaffold(
                retrieval_result,
                experience_nodes,
            )
            info["hook_b_scaffold_current_template"] = current_template
            reason = "template_scaffold"

        if selected_exp is None or selected_info is None:
            info["hook_b_scaffold_reason"] = "no_scaffold_match"
            return "", info

        scaffold = selected_exp.chain_summary
        info.update({
            "hook_b_scaffold_used": True,
            "hook_b_scaffold_reason": reason,
            "hook_b_scaffold_chars": len(scaffold),
            "hook_b_scaffold_source_question_id": selected_info.get("question_id", ""),
            "hook_b_scaffold_node_id": selected_info.get("node_id", ""),
            "hook_b_scaffold_score": selected_info.get("score", 0.0),
            "hook_b_scaffold_legacy_score": selected_info.get("legacy_score", 0.0),
            "hook_b_scaffold_semantic_score": selected_info.get("semantic_score", 0.0),
            "hook_b_scaffold_activation_score": selected_info.get("activation_score", 0.0),
            "hook_b_scaffold_template_score": selected_info.get("template_score", 0.0),
            "hook_b_scaffold_rank_in_hook_a": selected_info.get("rank_in_hook_a"),
            "hook_b_scaffold_experience": selected_info,
        })
        return (
            "[Verified Reference - reuse directly if relevant, skip re-analysis]:\n"
            f"{scaffold}\n\n"
        ), info


    def _retrieve_single(
        self,
        question_info: dict,
        _q_label: int | None = None,
        experience_snapshot: list | None = None,
    ) -> dict:
        _q = _q_label if _q_label is not None else self.session_counter + 1
        question_id = question_info.get("id", "")
        decomposition_ids = self._decomposition_ids_from_question_id(question_id)
        experience_nodes = (
            experience_snapshot
            if experience_snapshot is not None
            else self.experience_store.get_all()
        )


        if not experience_nodes:
            logger.info("[Hook A] Q%d: M_t=0 — no experience yet, pure LinearRAG", _q)
            result = super()._retrieve_single(question_info)
            result["best_experience"] = None
            result["hook_a_active"] = False
            result["hook_a_best_score"] = 0.0
            result["hook_a_new_seeds"] = 0
            result["hook_a_boosted_seeds"] = 0
            result["hook_a_fused_seed_count"] = 0
            result["hook_a_above_threshold"] = 0
            result["hook_a_new_above_threshold"] = 0
            result["base_activation"] = {}
            result["hook_a_matched_nodes"] = []
            result["hook_a_matched_sources"] = []
            result["hook_a_experiences"] = []
            result["exp_seed_entities"] = []
            result["question_id"] = question_id
            result["decomposition_ids"] = decomposition_ids
            return result


        question = question_info["question"]
        question_embedding = self.config.embedding_model.encode(
            question,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.config.batch_size,
        )


        seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores = (
            self.get_seed_entities(question)
        )

        if not seed_entity_indices:

            result = super()._retrieve_single(question_info)
            result["best_experience"] = None
            result["hook_a_active"] = False
            result["hook_a_best_score"] = 0.0
            result["hook_a_new_seeds"] = 0
            result["hook_a_boosted_seeds"] = 0
            result["hook_a_fused_seed_count"] = 0
            result["hook_a_above_threshold"] = 0
            result["hook_a_new_above_threshold"] = 0
            result["base_activation"] = {}
            result["hook_a_matched_nodes"] = []
            result["hook_a_matched_sources"] = []
            result["hook_a_experiences"] = []
            result["exp_seed_entities"] = []
            result["question_id"] = question_id
            result["decomposition_ids"] = decomposition_ids
            logger.info(
                "[Hook A] Q%d: no NER seed entities — skipping fusion (pure LinearRAG)",
                _q,
            )
            return result


        base_activation: dict = dict(zip(seed_entity_hash_ids, seed_entity_scores))

        matched = dual_channel_match(
            query_embedding=question_embedding,
            base_activation=base_activation,
            experience_nodes=experience_nodes,
            alpha=self.config.exp_alpha,
            top_k=self.config.exp_top_k,
        )

        best_experience: ExperienceNode | None = matched[0][0] if matched else None
        best_match_score: float = matched[0][1] if matched else 0.0


        query_metadata_set = set(decomposition_ids)
        if matched:
            _scores_arr = np.array([s for _, s in matched], dtype=np.float64)
            _sc = _scores_arr / self.config.exp_tau_s
            _sc = _sc - _sc.max()
            _sm = np.exp(_sc) / np.exp(_sc).sum()
            _sum_w = float(sum(float(sw) * node.confidence for (node, _), sw in zip(matched, _sm)))
            _topk_info = [
                (round(s, 3), round(n.confidence, 2), n.node_id[:8])
                for n, s in matched
            ]
            _topk_source_info = [
                {
                    "rank": i + 1,
                    "node_id": n.node_id[:8],
                    "question_id": getattr(n, "question_id", ""),
                    "decomposition_overlap": sorted(
                        query_metadata_set
                        & set(getattr(n, "decomposition_ids", []) or [])
                    ),
                    "score": round(s, 4),
                    "conf": round(n.confidence, 4),
                    "activation_size": len(n.activation_map),
                }
                for i, (n, s) in enumerate(matched)
            ]
            _experiences_info = [
                {
                    "rank": i + 1,
                    "is_scaffold": i == 0,
                    "score": round(s, 4),
                    "conf": round(n.confidence, 4),
                    "node_id": n.node_id[:8],
                    "question_id": getattr(n, "question_id", ""),
                    "decomposition_ids": list(getattr(n, "decomposition_ids", []) or []),
                    "decomposition_overlap": sorted(
                        query_metadata_set
                        & set(getattr(n, "decomposition_ids", []) or [])
                    ),
                    "timestamp": n.timestamp,
                    "activation_size": len(n.activation_map),
                    "answer": n.answer,
                    "summary": n.chain_summary,
                }
                for i, (n, s) in enumerate(matched)
            ]
        else:
            _sum_w = 0.0
            _topk_info = []
            _topk_source_info = []
            _experiences_info = []


        a_fused = adaptive_fusion(
            seed_entity_hash_ids=seed_entity_hash_ids,
            seed_entity_scores=seed_entity_scores,
            matched=matched,
            session_counter=self.session_counter,
            beta=self.config.exp_beta,
            tau_s=self.config.exp_tau_s,
            mu=self.config.exp_mu,
            fusion_epsilon=self.config.fusion_epsilon,
            use_temporal_decay=self.config.use_temporal_decay,
            max_exp_entities=self.config.exp_activation_top_n,
        )


        fused_seed_hash_ids: list = []
        fused_seed_indices: list = []
        fused_seed_texts: list = []
        fused_seed_scores: list = []
        n_new = 0
        n_boosted = 0

        for ehid, score in a_fused.items():
            if ehid not in self.entity_embedding_store.hash_id_to_idx:
                continue
            idx = self.entity_embedding_store.hash_id_to_idx[ehid]
            text = self.entity_embedding_store.hash_id_to_text[ehid]
            fused_seed_hash_ids.append(ehid)
            fused_seed_indices.append(idx)
            fused_seed_texts.append(text)
            fused_seed_scores.append(score)
            if ehid not in base_activation:
                n_new += 1
            elif score > base_activation[ehid] + 1e-6:
                n_boosted += 1


        exp_seed_entities: list = [
            self.entity_embedding_store.hash_id_to_text[ehid]
            for ehid in a_fused
            if ehid not in base_activation
            and ehid in self.entity_embedding_store.hash_id_to_idx
        ]

        if not fused_seed_indices:

            fused_seed_hash_ids = seed_entity_hash_ids
            fused_seed_indices = seed_entity_indices
            fused_seed_texts = seed_entities
            fused_seed_scores = seed_entity_scores
            exp_seed_entities = []

        _above_thresh = sum(1 for s in fused_seed_scores if s >= self.config.iteration_threshold)
        _n_above_new = 0
        if n_new > 0:
            _new_scores = [
                score for ehid, score in a_fused.items()
                if ehid not in base_activation
                and ehid in self.entity_embedding_store.hash_id_to_idx
            ]
            _n_above_new = sum(1 for s in _new_scores if s >= self.config.iteration_threshold)
            _score_str = (
                f" | new∈[{min(_new_scores):.3f},{max(_new_scores):.3f}]"
                f"(ℓ∞-norm) {_n_above_new}/{n_new}↑thresh"
            )
        else:
            _score_str = ""
        logger.info(
            "[Hook A] Q%d: M_t=%d matched=%d Σw=%.3f (α=%.2f,β=%.2f) top-K=%s"
            " | sources=%s | seeds orig=%d new=+%d boost=%d total=%d "
            "above_thresh=%d(≥%.2f)%s",
            _q,
            len(experience_nodes),
            len(matched),
            _sum_w,
            self.config.exp_alpha,
            self.config.exp_beta,
            _topk_info,
            [
                (
                    item["node_id"],
                    item["question_id"],
                    item["decomposition_overlap"],
                    item["activation_size"],
                )
                for item in _topk_source_info
            ],
            len(seed_entity_hash_ids),
            n_new,
            n_boosted,
            len(fused_seed_hash_ids),
            _above_thresh,
            self.config.iteration_threshold,
            _score_str,
        )


        sorted_passage_hash_ids, sorted_passage_scores, raw_actived_entities = (
            self.graph_search_with_seed_entities(
                question,
                question_embedding,
                fused_seed_indices,
                fused_seed_texts,
                fused_seed_hash_ids,
                fused_seed_scores,
            )
        )

        final_passage_hash_ids = sorted_passage_hash_ids[: self.config.retrieval_top_k]
        final_passage_scores = sorted_passage_scores[: self.config.retrieval_top_k]
        final_passages = [
            self.passage_embedding_store.hash_id_to_text[pid]
            for pid in final_passage_hash_ids
        ]
        actived_entities_flat = {
            ehid: float(score)
            for ehid, (_, score, _) in raw_actived_entities.items()
        }

        return {
            "question": question,
            "sorted_passage": final_passages,
            "sorted_passage_scores": final_passage_scores,
            "sorted_passage_hash_ids": final_passage_hash_ids,
            "gold_answer": question_info["answer"],
            "question_id": question_id,
            "decomposition_ids": decomposition_ids,

            "seed_entities": seed_entities,
            "exp_seed_entities": exp_seed_entities,


            "query_embedding": question_embedding,
            "actived_entities": actived_entities_flat,


            "base_activation": base_activation,
            "best_experience": best_experience,

            "hook_a_active": True,
            "hook_a_best_score": round(best_match_score, 4),
            "hook_a_new_seeds": n_new,
            "hook_a_boosted_seeds": n_boosted,
            "hook_a_fused_seed_count": len(fused_seed_hash_ids),
            "hook_a_above_threshold": _above_thresh,
            "hook_a_new_above_threshold": _n_above_new,
            "hook_a_matched_nodes": _topk_info,
            "hook_a_matched_sources": _topk_source_info,
            "hook_a_experiences": _experiences_info,
        }


    def _encode_experience(
        self,
        chain: ReasoningChain,
        query_embedding: np.ndarray,
        final_activation: dict,
        confidence: float,
        question_id: str = "",
        decomposition_ids: list[str] | None = None,
    ) -> ExperienceNode:
        epsilon = self.config.exp_epsilon
        sparse_activation = {
            ehid: score
            for ehid, score in final_activation.items()
            if score >= epsilon
        }

        reasoning_text = f"{chain.query} {chain.answer}"
        reasoning_embedding = self.config.embedding_model.encode(
            reasoning_text,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.config.batch_size,
        )


        if chain.raw_text:

            _TOP_ENTITIES_FOR_SCAFFOLD = 15
            top_entity_hash_ids = sorted(
                final_activation.items(), key=lambda x: x[1], reverse=True
            )[:_TOP_ENTITIES_FOR_SCAFFOLD]
            entity_texts = [
                self.entity_embedding_store.hash_id_to_text.get(ehid, "")
                for ehid, _ in top_entity_hash_ids
            ]
            key_reasoning = _extract_key_sentences(chain.raw_text, entity_texts, self.config.max_sentences)
            chain_summary = f"Q: {chain.query}\nReasoning: {key_reasoning}\nA: {chain.answer}"
        else:
            chain_summary = f"Q: {chain.query}\nA: {chain.answer}"

        return ExperienceNode(
            node_id=ExperienceStore.make_node_id(),
            query_embedding=query_embedding,
            activation_map=sparse_activation,
            reasoning_embedding=reasoning_embedding,
            chain_summary=chain_summary,
            answer=chain.answer,
            timestamp=self.session_counter,
            confidence=confidence,
            question_id=question_id,
            decomposition_ids=decomposition_ids or [],
        )


    def qa(self, questions):
        self._prepare_retrieval()
        results: list = [None] * len(questions)

        eqs_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eqs")

        def process_one(args: tuple) -> tuple:
            q_idx, question_info = args


            m_t_snapshot = self.experience_store.get_all()


            retrieval_result = self._retrieve_single(
                question_info, _q_label=q_idx + 1, experience_snapshot=m_t_snapshot
            )


            best_exp = retrieval_result.get("best_experience")
            scaffold_text, hook_b_info = self._build_hook_b_scaffold(
                retrieval_result,
                m_t_snapshot,
            )
            retrieval_result.update(hook_b_info)
            scaffold_node_id = hook_b_info.get("hook_b_scaffold_node_id", "")
            for exp in retrieval_result.get("hook_a_experiences", []) or []:
                exp["is_scaffold"] = (
                    hook_b_info["hook_b_scaffold_used"]
                    and exp.get("node_id") == scaffold_node_id
                )

            prompt_user = scaffold_text
            for passage in retrieval_result["sorted_passage"]:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {retrieval_result['question']}"
            messages = [
                {"role": "system", "content": _LIVING_RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_user},
            ]


            _n_pass = len(retrieval_result["sorted_passage"])
            _pass_chars = sum(len(p) for p in retrieval_result["sorted_passage"])
            _matched_nodes = retrieval_result.get("hook_a_matched_nodes", [])
            if best_exp is not None or hook_b_info["hook_b_scaffold_used"]:
                _all_ids = [e["node_id"] for e in retrieval_result.get("hook_a_experiences", [])]
                _all_sources = [
                    e.get("question_id") or e["node_id"]
                    for e in retrieval_result.get("hook_a_experiences", [])
                ]
                _exp_info = (
                    f"hook_a={len(_matched_nodes)}exp ids={_all_ids} | "
                    f"sources={_all_sources} | "
                    f"hook_b={hook_b_info['hook_b_scaffold_reason']}:"
                    f"{'inj' if hook_b_info['hook_b_scaffold_used'] else 'skip'} "
                    f"scaffold={hook_b_info.get('hook_b_scaffold_node_id', '')} "
                    f"score={hook_b_info.get('hook_b_scaffold_score', 0.0)} "
                    f"exp_chars={hook_b_info['hook_b_scaffold_chars']}"
                    f"(~{hook_b_info['hook_b_scaffold_chars'] // 4}tok)"
                )
            else:
                _exp_info = "no_exp"
            logger.info(
                "[Prompt] Q%d: n_pass=%d pass_chars=%d(~%dtok) | %s",
                q_idx + 1, _n_pass, _pass_chars, _pass_chars // 4, _exp_info,
            )


            content, reasoning, prompt_tokens, completion_tokens, reasoning_tokens = self.llm_model.infer(messages)
            reasoning_text = reasoning or ""


            chain: ReasoningChain = parse_reasoning_chain(
                content=content,
                reasoning_text=reasoning_text,
                question=retrieval_result["question"],
                passage_hash_ids=retrieval_result["sorted_passage_hash_ids"],
            )
            retrieval_result["pred_answer"] = chain.answer
            retrieval_result["pred_reasoning"] = reasoning_text
            retrieval_result["answer_parsed"] = chain.parse_success
            retrieval_result["reasoning_chain"] = {
                "answer": chain.answer,
            }
            retrieval_result["prompt_tokens"] = prompt_tokens
            retrieval_result["completion_tokens"] = completion_tokens
            retrieval_result["reasoning_tokens"] = reasoning_tokens

            if not chain.parse_success:
                logger.warning(
                    f"[LivingRAG] idx={q_idx}: empty content — "
                    f"response may be truncated."
                )


            if chain.is_reliable:
                eqs_future = eqs_executor.submit(
                    self._eqs_worker, q_idx, chain, retrieval_result, m_t_snapshot
                )
                eqs_info = eqs_future.result()
            else:
                eqs_info = {
                    "eqs_gs": None, "eqs_ns": None,
                    "eqs_phi": None,
                    "eqs_written": False,
                }

            retrieval_result.update(eqs_info)


            retrieval_result.pop("query_embedding", None)
            retrieval_result.pop("actived_entities", None)
            retrieval_result.pop("base_activation", None)
            retrieval_result.pop("best_experience", None)

            return q_idx, retrieval_result

        max_workers = self.config.max_workers
        logger.info(
            f"[LivingRAG] Streaming mode: max_workers={max_workers}. "
            f"Retrieval+LLM concurrent; EQS/NLI serialised in single worker. "
            f"Each question reads M_t snapshot at retrieval time."
        )

        if getattr(self.config, "living_family_serial", False):
            family_keys = self._question_family_keys(questions)
            family_queues: dict[str, deque] = defaultdict(deque)
            family_order: list[str] = []
            for idx, question_info in enumerate(questions):
                family = family_keys[idx]
                if family not in family_queues:
                    family_order.append(family)
                family_queues[family].append((idx, question_info))

            logger.info(
                "[LivingRAG] Family-serial scheduler enabled: %d question families, "
                "max_workers=%d. Questions sharing decomposition ids run serially.",
                len(family_order),
                max_workers,
            )

            with ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="qa"
            ) as qa_executor:
                active = {}
                next_family_idx = 0

                def submit_family(family: str) -> None:
                    if not family_queues[family]:
                        return
                    future = qa_executor.submit(process_one, family_queues[family].popleft())
                    active[future] = family

                while len(active) < max_workers and next_family_idx < len(family_order):
                    family = family_order[next_family_idx]
                    next_family_idx += 1
                    submit_family(family)

                with tqdm(total=len(questions), desc="Questions", ncols=80) as pbar:
                    while active:
                        for future in as_completed(list(active.keys())):
                            family = active.pop(future)
                            idx, result = future.result()
                            results[idx] = result
                            pbar.update(1)

                            if family_queues[family]:
                                submit_family(family)
                            else:
                                while (
                                    len(active) < max_workers
                                    and next_family_idx < len(family_order)
                                ):
                                    next_family = family_order[next_family_idx]
                                    next_family_idx += 1
                                    submit_family(next_family)
                            break
        else:
            with ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="qa"
            ) as qa_executor:
                futures = [
                    qa_executor.submit(process_one, (idx, q))
                    for idx, q in enumerate(questions)
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(questions),
                    desc="Questions",
                    ncols=80,
                ):
                    idx, result = future.result()
                    results[idx] = result

        eqs_executor.shutdown(wait=True)


        if self.config.consolidate_interval > 0 and self._writes_since_consolidate > 0:
            n_final = self.experience_store.consolidate(
                alpha=self.config.exp_alpha,
                tau_merge=self.config.tau_merge,
            )
            if n_final:
                logger.info(
                    "[LivingRAG] End-of-run consolidation: merged %d node(s), "
                    "|M_t|=%d",
                    n_final,
                    len(self.experience_store),
                )
        self.experience_store.save(
            entity_id_to_text=self.entity_embedding_store.hash_id_to_text
        )
        logger.info(
            "[LivingRAG] Experience memory saved: %d node(s) → %s",
            len(self.experience_store),
            self.experience_store.db_path,
        )

        return results


    def _eqs_worker(
        self,
        q_idx: int,
        chain: ReasoningChain,
        retrieval_result: dict,
        experience_snapshot: list,
    ) -> dict:


        self.session_counter += 1
        current_q = self.session_counter

        query_embedding: np.ndarray = retrieval_result["query_embedding"]
        final_activation: dict = retrieval_result["actived_entities"]


        base_activation: dict = retrieval_result.get("base_activation", {})
        passage_texts = dict(zip(
            retrieval_result["sorted_passage_hash_ids"],
            retrieval_result["sorted_passage"],
        ))


        experience_nodes = experience_snapshot

        _t0 = time.perf_counter()
        eqs: EQSResult = run_eqs(
            chain=chain,
            experience_nodes=experience_nodes,
            query_embedding=query_embedding,
            base_activation=base_activation,
            alpha=self.config.exp_alpha,
            tau_nli=self.config.tau_nli,
            tau_gs=self.config.tau_gs,
            tau_novel=self.config.tau_novel,
            passage_texts=passage_texts,
            nli_model=self.nli_model,
            embedding_model=self.config.embedding_model,
            top_k_evidence=self.config.gs_top_k_evidence,
        )

        _gs_str = f"{eqs.gs:.3f}" if eqs.gs is not None else "skip"
        _elapsed = f"{time.perf_counter() - _t0:.2f}s"
        _qid = retrieval_result.get("question_id", "")
        _decomp = retrieval_result.get("decomposition_ids", [])

        if eqs.passes:
            confidence = eqs.gs
            _enc_t0 = time.perf_counter()
            node = self._encode_experience(
                chain,
                query_embedding,
                final_activation,
                confidence,
                question_id=retrieval_result.get("question_id", ""),
                decomposition_ids=retrieval_result.get("decomposition_ids", []),
            )
            self.experience_store.add_experience(node)
            self._writes_since_consolidate += 1


            interval = self.config.consolidate_interval
            n_merged = 0
            if interval > 0 and self._writes_since_consolidate >= interval:
                _before = len(self.experience_store)
                logger.info(
                    "[LivingRAG] eqs#%d: consolidation triggered "
                    "(every %d writes, |M_t|=%d, τ_merge=%.2f)",
                    current_q, interval, _before, self.config.tau_merge,
                )
                n_merged = self.experience_store.consolidate(
                    alpha=self.config.exp_alpha,
                    tau_merge=self.config.tau_merge,
                )
                self._writes_since_consolidate = 0
                logger.info(
                    "[LivingRAG] eqs#%d: consolidation done — merged=%d, |M_t|=%d→%d",
                    current_q, n_merged, _before, len(self.experience_store),
                )

            _size_str = f"|M_t|={len(self.experience_store)}"
            if n_merged:
                _size_str += f"(merged {n_merged})"
            logger.info(
                f"[LivingRAG] Q{q_idx + 1} [eqs#{current_q}]: "
                f"qid={_qid} decomp={_decomp} "
                f"GS={_gs_str} NS={eqs.ns:.3f} Φ={eqs.phi:.3f} "
                f"elapsed={_elapsed} → Written conf={confidence:.3f} "
                f"encode={time.perf_counter() - _enc_t0:.2f}s {_size_str}"
            )
        else:
            logger.info(
                f"[LivingRAG] Q{q_idx + 1} [eqs#{current_q}]: "
                f"qid={_qid} decomp={_decomp} "
                f"GS={_gs_str} NS={eqs.ns:.3f} Φ={eqs.phi:.3f} "
                f"elapsed={_elapsed} → Skipped"
            )


        return {
            "eqs_gs": round(eqs.gs, 4) if eqs.gs is not None else None,
            "eqs_ns": round(eqs.ns, 4),
            "eqs_phi": round(eqs.phi, 4),
            "eqs_written": eqs.passes,
        }
