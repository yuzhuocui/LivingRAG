from src.embedding_store import EmbeddingStore
from src.utils import min_max_normalize
import os
import json
import threading
from collections import defaultdict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.ner import SpacyNER
import igraph as ig
import re
import logging
import torch
logger = logging.getLogger(__name__)


class LinearRAG:
    def __init__(self, global_config):
        self.config = global_config
        retrieval_method = "Vectorized" if self.config.use_vectorized_retrieval else "BFS"
        logger.info("[LinearRAG] Init: dataset=%s retrieval=%s", self.config.dataset_name, retrieval_method)


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.config.use_vectorized_retrieval:
            logger.info(f"Using device: {self.device} for vectorized retrieval")


        self._ppr_lock = threading.Lock()

        self.dataset_name = global_config.dataset_name
        ner_type = getattr(self.config, "ner_type", "spacy")
        if ner_type not in {"spacy", "llm"}:
            raise ValueError(f"Unsupported ner_type: {ner_type!r}. Expected 'spacy' or 'llm'.")
        self.load_embedding_store()
        self.llm_model = self.config.llm_model

        if ner_type == "llm":
            from src.llm_ner import LlmNER
            llm_ner_cache = os.path.join(
                self.config.working_dir, self.dataset_name, "llm_ner_cache.json"
            )
            self.spacy_ner = LlmNER(
                model=getattr(self.config, "llm_ner_model", "qwen/qwen3.5-9b"),
                cache_path=llm_ner_cache,
            )
            logger.info("[LinearRAG] Using LlmNER (model=%s, cache=%s)",
                        self.config.llm_ner_model, llm_ner_cache)
        else:
            self.spacy_ner = SpacyNER(self.config.spacy_model)
            logger.info("[LinearRAG] Using SpacyNER (model=%s)", self.config.spacy_model)

        self.graph = ig.Graph(directed=False)

    def load_embedding_store(self):

        ner_type = getattr(self.config, "ner_type", "spacy")
        base = os.path.join(self.config.working_dir, self.dataset_name)
        self.passage_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(base, "passage_embedding.parquet"), batch_size=self.config.batch_size, namespace="passage")
        self.entity_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(base, f"entity_embedding_{ner_type}.parquet"), batch_size=self.config.batch_size, namespace="entity")
        self.sentence_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(base, f"sentence_embedding_{ner_type}.parquet"), batch_size=self.config.batch_size, namespace="sentence")

    def load_existing_data(self,passage_hash_ids):
        ner_type = getattr(self.config, "ner_type", "spacy")
        self.ner_results_path = os.path.join(
            self.config.working_dir, self.dataset_name, f"ner_results_{ner_type}.json"
        )
        if os.path.exists(self.ner_results_path):
            existing_ner_reuslts = json.load(open(self.ner_results_path))
            existing_passage_hash_id_to_entities = existing_ner_reuslts["passage_hash_id_to_entities"]
            existing_sentence_to_entities = existing_ner_reuslts["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            return existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids
        else:
            return {}, {}, passage_hash_ids

    def qa(self, questions):


        self._prepare_retrieval()

        system_prompt = (
            "As an advanced reading comprehension assistant, your task is to analyze "
            "text passages and corresponding questions meticulously. "
            "Provide a concise, definitive answer only."
        )

        results = [None] * len(questions)
        max_workers = self.config.max_workers

        def process_one(args):
            q_idx, question_info = args


            retrieval_result = self._retrieve_single(question_info)


            prompt_user = ""
            for passage in retrieval_result["sorted_passage"]:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {retrieval_result['question']}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_user},
            ]


            content, reasoning_details, prompt_tokens, completion_tokens, reasoning_tokens = (
                self.llm_model.infer(messages)
            )


            retrieval_result["pred_answer"] = content.strip()
            retrieval_result["pred_reasoning"] = reasoning_details or ""
            retrieval_result["answer_parsed"] = bool(content.strip())
            retrieval_result["prompt_tokens"] = prompt_tokens
            retrieval_result["completion_tokens"] = completion_tokens
            retrieval_result["reasoning_tokens"] = reasoning_tokens


            retrieval_result.pop("query_embedding", None)
            retrieval_result.pop("actived_entities", None)

            return q_idx, retrieval_result

        logger.info(
            "[LinearRAG] Streaming QA mode: max_workers=%d. Retrieval+LLM concurrent.",
            max_workers,
        )

        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="linear_qa",
        ) as executor:
            futures = [
                executor.submit(process_one, (idx, question_info))
                for idx, question_info in enumerate(questions)
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Questions",
                ncols=80,
            ):
                q_idx, retrieval_result = future.result()
                results[q_idx] = retrieval_result

        return results

    def retrieve(self, questions):
        self._prepare_retrieval()
        retrieval_results = []
        for question_info in tqdm(questions, desc="Retrieving", ncols=80):
            retrieval_results.append(self._retrieve_single(question_info))
        return retrieval_results

    def _prepare_retrieval(self):
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        self.passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self.passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        self.sentence_hash_ids = list(self.sentence_embedding_store.hash_id_to_text.keys())
        self.sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
        self.node_name_to_vertex_idx = {
            v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()
        }
        self.vertex_idx_to_node_name = {
            v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()
        }


        if self.config.passage_length_norm_b > 0:
            lengths = [
                len(self.passage_embedding_store.hash_id_to_text[pid])
                for pid in self.passage_hash_ids
            ]
            self._avg_passage_len = float(sum(lengths) / len(lengths)) if lengths else 1.0
            logger.info(
                "[LinearRAG] passage_length_norm_b=%.2f, avg_passage_len=%.0f chars",
                self.config.passage_length_norm_b, self._avg_passage_len,
            )
        else:
            self._avg_passage_len = 1.0

        if self.config.use_vectorized_retrieval:
            logger.info("Precomputing sparse adjacency matrices for vectorized retrieval...")
            self._precompute_sparse_matrices()
            e2s_shape = self.entity_to_sentence_sparse.shape
            s2e_shape = self.sentence_to_entity_sparse.shape
            e2s_nnz = self.entity_to_sentence_sparse._nnz()
            s2e_nnz = self.sentence_to_entity_sparse._nnz()
            logger.info(f"Matrices built: Entity-Sentence {e2s_shape}, Sentence-Entity {s2e_shape}")
            logger.info(
                f"E2S Sparsity: {(1 - e2s_nnz / (e2s_shape[0] * e2s_shape[1])) * 100:.2f}%"
                f" (nnz={e2s_nnz})"
            )
            logger.info(
                f"S2E Sparsity: {(1 - s2e_nnz / (s2e_shape[0] * s2e_shape[1])) * 100:.2f}%"
                f" (nnz={s2e_nnz})"
            )
            logger.info(f"Device: {self.device}")

    def _retrieve_single(self, question_info):
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

        if len(seed_entities) != 0:
            sorted_passage_hash_ids, sorted_passage_scores, raw_actived_entities = (
                self.graph_search_with_seed_entities(
                    question,
                    question_embedding,
                    seed_entity_indices,
                    seed_entities,
                    seed_entity_hash_ids,
                    seed_entity_scores,
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
        else:
            sorted_passage_indices, sorted_passage_scores = self.dense_passage_retrieval(
                question_embedding
            )
            final_passage_indices = sorted_passage_indices[: self.config.retrieval_top_k]
            final_passage_scores = sorted_passage_scores[: self.config.retrieval_top_k]
            final_passages = [
                self.passage_embedding_store.texts[idx] for idx in final_passage_indices
            ]
            final_passage_hash_ids = [
                self.passage_embedding_store.hash_ids[idx] for idx in final_passage_indices
            ]
            actived_entities_flat = {}

        return {
            "question": question,
            "sorted_passage": final_passages,
            "sorted_passage_scores": final_passage_scores,
            "sorted_passage_hash_ids": final_passage_hash_ids,
            "gold_answer": question_info["answer"],

            "seed_entities": seed_entities,

            "query_embedding": question_embedding,
            "actived_entities": actived_entities_flat,
        }

    def _precompute_sparse_matrices(self):
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)


        entity_to_sentence_indices = []
        entity_to_sentence_values = []

        for entity_hash_id, sentence_hash_ids in self.entity_hash_id_to_sentence_hash_ids.items():
            entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
            for sentence_hash_id in sentence_hash_ids:
                sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
                entity_to_sentence_indices.append([entity_idx, sentence_idx])
                entity_to_sentence_values.append(1.0)


        sentence_to_entity_indices = []
        sentence_to_entity_values = []

        for sentence_hash_id, entity_hash_ids in self.sentence_hash_id_to_entity_hash_ids.items():
            sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
            for entity_hash_id in entity_hash_ids:
                entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
                sentence_to_entity_indices.append([sentence_idx, entity_idx])
                sentence_to_entity_values.append(1.0)


        if len(entity_to_sentence_indices) > 0:
            e2s_indices = torch.tensor(entity_to_sentence_indices, dtype=torch.long).t()
            e2s_values = torch.tensor(entity_to_sentence_values, dtype=torch.float32)
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                e2s_indices, e2s_values, (num_entities, num_sentences), device=self.device
            ).coalesce()
        else:
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_entities, num_sentences), device=self.device
            )

        if len(sentence_to_entity_indices) > 0:
            s2e_indices = torch.tensor(sentence_to_entity_indices, dtype=torch.long).t()
            s2e_values = torch.tensor(sentence_to_entity_values, dtype=torch.float32)
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                s2e_indices, s2e_values, (num_sentences, num_entities), device=self.device
            ).coalesce()
        else:
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_sentences, num_entities), device=self.device
            )

    def graph_search_with_seed_entities(self, question, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        if self.config.use_vectorized_retrieval:
            entity_weights, actived_entities = self.calculate_entity_scores_vectorized(question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
        else:
            entity_weights, actived_entities = self.calculate_entity_scores(question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
        passage_weights = self.calculate_passage_scores(question, question_embedding, actived_entities)
        node_weights = entity_weights + passage_weights
        ppr_sorted_passage_indices,ppr_sorted_passage_scores = self.run_ppr(node_weights)


        return ppr_sorted_passage_indices, ppr_sorted_passage_scores, actived_entities

    def run_ppr(self, node_weights):
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        with self._ppr_lock:
            pagerank_scores = self.graph.personalized_pagerank(
                vertices=range(len(self.node_name_to_vertex_idx)),
                damping=self.config.damping,
                directed=False,
                weights='weight',
                reset=reset_prob,
                implementation='prpack'
            )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_indices])
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]

        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]]
            for i in sorted_indices_in_doc_scores
        ]

        return sorted_passage_hash_ids, sorted_passage_scores.tolist()

    def calculate_entity_scores(self,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
        actived_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        for seed_entity_idx,seed_entity,seed_entity_hash_id,seed_entity_score in zip(seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 1)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score
        used_sentence_hash_ids = set()
        current_entities = actived_entities.copy()
        iteration = 1
        while len(current_entities) > 0 and iteration < self.config.max_iterations:
            new_entities = {}
            for entity_hash_id, (entity_id, entity_score, tier) in current_entities.items():
                if entity_score < self.config.iteration_threshold:
                    continue
                if entity_hash_id not in self.entity_hash_id_to_sentence_hash_ids:


                    continue
                sentence_hash_ids = [sid for sid in list(self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]) if sid not in used_sentence_hash_ids]
                if not sentence_hash_ids:
                    continue
                sentence_indices = [self.sentence_embedding_store.hash_id_to_idx[sid] for sid in sentence_hash_ids]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
                sentence_similarities = np.dot(sentence_embeddings, question_emb).flatten()
                top_sentence_indices = np.argsort(sentence_similarities)[::-1][:self.config.top_k_sentence]
                for top_sentence_index in top_sentence_indices:
                    top_sentence_hash_id = sentence_hash_ids[top_sentence_index]
                    top_sentence_score = sentence_similarities[top_sentence_index]
                    used_sentence_hash_ids.add(top_sentence_hash_id)
                    entity_hash_ids_in_sentence = self.sentence_hash_id_to_entity_hash_ids[top_sentence_hash_id]
                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        next_entity_score = entity_score * top_sentence_score
                        if next_entity_score < self.config.iteration_threshold:
                            continue
                        next_enitity_node_idx = self.node_name_to_vertex_idx[next_entity_hash_id]
                        entity_weights[next_enitity_node_idx] += next_entity_score
                        new_entities[next_entity_hash_id] = (next_enitity_node_idx, next_entity_score, iteration+1)
            actived_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
        return entity_weights, actived_entities

    def calculate_entity_scores_vectorized(self,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):

        entity_weights = np.zeros(len(self.graph.vs["name"]))
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)


        question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
        sentence_similarities_np = np.dot(self.sentence_embeddings, question_emb).flatten()


        sentence_similarities = torch.from_numpy(sentence_similarities_np).float().to(self.device)


        used_sentence_mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)


        seed_indices = torch.tensor([[idx] for idx in seed_entity_indices], dtype=torch.long).t()
        seed_values = torch.tensor(seed_entity_scores, dtype=torch.float32)
        entity_scores_sparse = torch.sparse_coo_tensor(
            seed_indices, seed_values, (num_entities,), device=self.device
        ).coalesce()


        entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)
        entity_scores_dense.scatter_(0, torch.tensor(seed_entity_indices, device=self.device),
                                     torch.tensor(seed_entity_scores, dtype=torch.float32, device=self.device))


        actived_entities = {}
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
        ):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 0)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score

        current_entity_scores_sparse = entity_scores_sparse


        for iteration in range(1, self.config.max_iterations):

            current_entity_scores_dense = current_entity_scores_sparse.to_dense()


            current_entity_scores_dense = torch.where(
                current_entity_scores_dense >= self.config.iteration_threshold,
                current_entity_scores_dense,
                torch.zeros_like(current_entity_scores_dense)
            )


            nonzero_mask = current_entity_scores_dense > 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)

            if len(nonzero_indices) == 0:
                break


            nonzero_values = current_entity_scores_dense[nonzero_indices]
            current_entity_scores_sparse = torch.sparse_coo_tensor(
                nonzero_indices.unsqueeze(0), nonzero_values, (num_entities,), device=self.device
            ).coalesce()


            current_scores_2d = torch.sparse_coo_tensor(
                torch.stack([nonzero_indices, torch.zeros_like(nonzero_indices)]),
                nonzero_values,
                (num_entities, 1),
                device=self.device
            ).coalesce()


            sentence_activation = torch.sparse.mm(
                self.entity_to_sentence_sparse.t(),
                current_scores_2d
            )

            if sentence_activation.is_sparse:
                sentence_activation = sentence_activation.to_dense()
            sentence_activation = sentence_activation.squeeze()


            sentence_activation = torch.where(
                used_sentence_mask,
                torch.zeros_like(sentence_activation),
                sentence_activation
            )


            selected_sentence_indices_list = []

            if len(nonzero_indices) > 0 and self.config.top_k_sentence > 0:

                for i, entity_idx in enumerate(nonzero_indices):
                    entity_score = nonzero_values[i]


                    entity_row = self.entity_to_sentence_sparse[entity_idx].coalesce()
                    entity_sentence_indices = entity_row.indices()[0]

                    if len(entity_sentence_indices) == 0:
                        continue


                    sentence_mask = ~used_sentence_mask[entity_sentence_indices]
                    available_sentence_indices = entity_sentence_indices[sentence_mask]

                    if len(available_sentence_indices) == 0:
                        continue


                    sentence_sims = sentence_similarities[available_sentence_indices]


                    k = min(self.config.top_k_sentence, len(sentence_sims))
                    if k > 0:
                        top_k_values, top_k_local_indices = torch.topk(sentence_sims, k)
                        top_k_sentence_indices = available_sentence_indices[top_k_local_indices]
                        selected_sentence_indices_list.append(top_k_sentence_indices)


                if len(selected_sentence_indices_list) > 0:
                    all_selected_sentences = torch.cat(selected_sentence_indices_list)
                    unique_selected_sentences = torch.unique(all_selected_sentences)


                    used_sentence_mask[unique_selected_sentences] = True


                    weighted_sentence_scores = sentence_activation * sentence_similarities


                    mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)
                    mask[unique_selected_sentences] = True
                    weighted_sentence_scores = torch.where(
                        mask,
                        weighted_sentence_scores,
                        torch.zeros_like(weighted_sentence_scores)
                    )
                else:

                    weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
            else:

                weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)


            weighted_nonzero_mask = weighted_sentence_scores > 0
            weighted_nonzero_indices = torch.nonzero(weighted_nonzero_mask, as_tuple=False).squeeze(-1)

            if len(weighted_nonzero_indices) > 0:
                weighted_nonzero_values = weighted_sentence_scores[weighted_nonzero_indices]
                weighted_scores_2d = torch.sparse_coo_tensor(
                    torch.stack([weighted_nonzero_indices, torch.zeros_like(weighted_nonzero_indices)]),
                    weighted_nonzero_values,
                    (num_sentences, 1),
                    device=self.device
                ).coalesce()

                next_entity_scores_result = torch.sparse.mm(
                    self.sentence_to_entity_sparse.t(),
                    weighted_scores_2d
                )

                if next_entity_scores_result.is_sparse:
                    next_entity_scores_result = next_entity_scores_result.to_dense()
                next_entity_scores_dense = next_entity_scores_result.squeeze()
            else:
                next_entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)


            entity_scores_dense += next_entity_scores_dense


            next_entity_scores_np = next_entity_scores_dense.cpu().numpy()
            active_indices = np.where(next_entity_scores_np >= self.config.iteration_threshold)[0]
            for entity_idx in active_indices:
                score = next_entity_scores_np[entity_idx]
                entity_hash_id = self.entity_hash_ids[entity_idx]

                actived_entities[entity_hash_id] = (entity_idx, float(score), iteration)


            next_nonzero_mask = next_entity_scores_dense > 0
            next_nonzero_indices = torch.nonzero(next_nonzero_mask, as_tuple=False).squeeze(-1)
            if len(next_nonzero_indices) > 0:
                next_nonzero_values = next_entity_scores_dense[next_nonzero_indices]
                current_entity_scores_sparse = torch.sparse_coo_tensor(
                    next_nonzero_indices.unsqueeze(0), next_nonzero_values,
                    (num_entities,), device=self.device
                ).coalesce()
            else:
                break


        entity_scores_final = entity_scores_dense.cpu().numpy()


        nonzero_indices = np.where(entity_scores_final > 0)[0]
        for entity_idx in nonzero_indices:
            score = entity_scores_final[entity_idx]
            entity_hash_id = self.entity_hash_ids[entity_idx]
            entity_node_idx = self.node_name_to_vertex_idx[entity_hash_id]
            entity_weights[entity_node_idx] = float(score)

        return entity_weights, actived_entities

    def calculate_passage_scores(self, question, question_embedding, actived_entities):
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(question_embedding)
        dpr_passage_scores = min_max_normalize(dpr_passage_scores)
        apply_attribute_boost = (
            self.config.enable_hybrid_attribute_fallback
            and self._is_attribute_query(question)
        )
        question_lower = question.lower()

        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            total_entity_bonus = 0
            passage_hash_id = self.passage_embedding_store.hash_ids[dpr_passage_index]
            dpr_passage_score = dpr_passage_scores[i]
            passage_text_lower = self.passage_embedding_store.hash_id_to_text[passage_hash_id].lower()
            for entity_hash_id, (entity_id, entity_score, tier) in actived_entities.items():
                entity_lower = self.entity_embedding_store.hash_id_to_text[entity_hash_id].lower()
                entity_occurrences = passage_text_lower.count(entity_lower)
                if entity_occurrences > 0:
                    denom = tier if tier >= 1 else 1
                    entity_bonus = entity_score * math.log(1 + entity_occurrences) / denom
                    total_entity_bonus += entity_bonus

            passage_score = self.config.passage_ratio * dpr_passage_score + math.log(1 + total_entity_bonus)

            if self.config.passage_length_norm_b > 0:
                b = self.config.passage_length_norm_b
                passage_len = len(passage_text_lower)
                length_norm = (1.0 - b) + b * passage_len / self._avg_passage_len
                passage_score /= length_norm

            if apply_attribute_boost:
                overlap = self._attribute_keyword_overlap(question_lower, passage_text_lower)
                if overlap > 0:
                    passage_score += self.config.attribute_keyword_boost * math.log(1 + overlap)

            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = passage_score * self.config.passage_node_weight
        return passage_weights

    def dense_passage_retrieval(self, question_embedding):
        question_emb = question_embedding.reshape(1, -1)
        question_passage_similarities = np.dot(self.passage_embeddings, question_emb.T).flatten()
        sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]
        sorted_passage_scores = question_passage_similarities[sorted_passage_indices].tolist()
        return sorted_passage_indices, sorted_passage_scores

    def _is_attribute_query(self, question):
        tokens = set(re.findall(r"\w+", question.lower()))
        return any(keyword in tokens for keyword in self.config.attribute_query_keywords)

    def _attribute_keyword_overlap(self, question_lower, passage_text_lower):
        overlap = 0
        for keyword in self.config.attribute_query_keywords:
            if keyword in question_lower and keyword in passage_text_lower:
                overlap += 1
        return overlap

    def get_seed_entities(self, question):
        question_entities = list(self.spacy_ner.question_ner(question))
        if len(question_entities) == 0:
            return [],[],[],[]
        question_entity_embeddings = self.config.embedding_model.encode(question_entities,normalize_embeddings=True,show_progress_bar=False,batch_size=self.config.batch_size)
        similarities = np.dot(self.entity_embeddings, question_entity_embeddings.T)

        ner_type = getattr(self.config, "ner_type", "spacy")
        use_llm_filter = ner_type == "llm"
        if use_llm_filter:
            top_k = getattr(self.config, "llm_ner_seed_top_k", 3)
            max_degree = getattr(self.config, "llm_ner_seed_max_degree", 300)

        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []
        seen_hash_ids: set = set()

        for query_entity_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_entity_idx]

            if use_llm_filter:


                top_k_indices = np.argpartition(entity_scores, -top_k)[-top_k:]
                top_k_indices = top_k_indices[np.argsort(entity_scores[top_k_indices])[::-1]]

                chosen_idx = None
                chosen_score = -1.0
                for candidate_idx in top_k_indices:
                    candidate_hash_id = self.entity_hash_ids[candidate_idx]
                    degree = len(self.entity_hash_id_to_sentence_hash_ids.get(candidate_hash_id, []))
                    if degree <= max_degree:
                        chosen_idx = candidate_idx
                        chosen_score = float(entity_scores[candidate_idx])
                        break

                if chosen_idx is None:

                    chosen_idx = int(top_k_indices[0])
                    chosen_score = float(entity_scores[chosen_idx])
                    logger.debug(
                        "[get_seed_entities] All top-%d candidates for %r exceed max_degree=%d; "
                        "falling back to top-1: %r (degree=%d)",
                        top_k, question_entities[query_entity_idx], max_degree,
                        self.entity_embedding_store.hash_id_to_text[self.entity_hash_ids[chosen_idx]],
                        len(self.entity_hash_id_to_sentence_hash_ids.get(self.entity_hash_ids[chosen_idx], [])),
                    )

                best_entity_idx = chosen_idx
                best_entity_score = chosen_score
            else:
                best_entity_idx = int(np.argmax(entity_scores))
                best_entity_score = float(entity_scores[best_entity_idx])

            best_entity_hash_id = self.entity_hash_ids[best_entity_idx]
            if best_entity_hash_id in seen_hash_ids:
                continue
            seen_hash_ids.add(best_entity_hash_id)

            best_entity_text = self.entity_embedding_store.hash_id_to_text[best_entity_hash_id]
            seed_entity_indices.append(best_entity_idx)
            seed_entity_texts.append(best_entity_text)
            seed_entity_hash_ids.append(best_entity_hash_id)
            seed_entity_scores.append(best_entity_score)

        return seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores

    def index(self, passages):
        self.node_to_node_stats = defaultdict(dict)
        self.entity_to_sentence_stats = defaultdict(dict)
        self.passage_embedding_store.insert_text(passages)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
        existing_passage_hash_id_to_entities,existing_sentence_to_entities, new_passage_hash_ids = self.load_existing_data(hash_id_to_passage.keys())
        if len(new_passage_hash_ids) > 0:
            new_hash_id_to_passage = {k : hash_id_to_passage[k] for k in new_passage_hash_ids}
            new_passage_hash_id_to_entities,new_sentence_to_entities = self.spacy_ner.batch_ner(new_hash_id_to_passage, self.config.max_workers)
            self.merge_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities)
        self.save_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        entity_nodes, sentence_nodes,passage_hash_id_to_entities,self.entity_to_sentence,self.sentence_to_entity = self.extract_nodes_and_edges(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_embedding_store.insert_text(list(entity_nodes))
        self.entity_hash_id_to_sentence_hash_ids = {}
        for entity, sentence in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
            self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = [self.sentence_embedding_store.text_to_hash_id[s] for s in sentence]
        self.sentence_hash_id_to_entity_hash_ids = {}
        for sentence, entities in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id[sentence]
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = [self.entity_embedding_store.text_to_hash_id[e] for e in entities]
        self.add_entity_to_passage_edges(passage_hash_id_to_entities)
        self.add_adjacent_passage_edges()
        self.augment_graph()
        output_graphml_path = os.path.join(self.config.working_dir,self.dataset_name, "LinearRAG.graphml")
        os.makedirs(os.path.dirname(output_graphml_path), exist_ok=True)
        self.graph.write_graphml(output_graphml_path)

    def add_adjacent_passage_edges(self):
        passage_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        index_pattern = re.compile(r'^(\d+):')
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip()))
        ]
        indexed_items.sort(key=lambda x: x[0])
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.node_to_node_stats[current_node][next_node] = 1.0

    def augment_graph(self):
        self.add_nodes()
        self.add_edges()

    def add_nodes(self):
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()} 
        entity_hash_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}

        passage_hash_ids = set(passage_hash_id_to_text.keys())

        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                self.graph.add_vertex(name=hash_id, content=text)

        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}   
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id]
            for passage_id in passage_hash_ids
            if passage_id in self.node_name_to_vertex_idx
        ]

    def add_edges(self):
        edges = []
        weights = []

        for node_hash_id, node_to_node_stats in self.node_to_node_stats.items():
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
        passage_to_entity_count = {}
        passage_to_all_score = defaultdict(int)
        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            passage_lower = passage.lower()
            for entity in entities:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]

                count = passage_lower.count(entity.lower())
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_score[passage_hash_id] += count
        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            total = passage_to_all_score[passage_hash_id]
            score = count / total if total > 0 else 0.0
            self.node_to_node_stats[passage_hash_id][entity_hash_id] = score

    def extract_nodes_and_edges(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        entity_nodes = set()
        sentence_nodes = set()
        passage_hash_id_to_entities = defaultdict(set)
        entity_to_sentence= defaultdict(set)
        sentence_to_entity = defaultdict(set)
        for passage_hash_id, entities in existing_passage_hash_id_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)
        for sentence,entities in existing_sentence_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)
        return entity_nodes, sentence_nodes, passage_hash_id_to_entities, entity_to_sentence, sentence_to_entity

    def merge_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities):
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def save_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        with open(self.ner_results_path, "w") as f:
            json.dump({"passage_hash_id_to_entities": existing_passage_hash_id_to_entities, "sentence_to_entities": existing_sentence_to_entities}, f)
