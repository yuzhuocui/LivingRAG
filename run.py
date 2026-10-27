import argparse
import csv
import dataclasses
import json
import logging
from sentence_transformers import SentenceTransformer
import os
import warnings
from src.evaluate import Evaluator
from src.utils import LLM_Model
from src.utils import setup_logging
from datetime import datetime

from src.config import LivingRAGConfig
from src.LivingRAG import LivingRAG


warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="dataset")

    parser.add_argument("--working_dir", default="import")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--batch_size", type=int, default=128)


    parser.add_argument("--model", type=str, default="living_rag", choices=["living_rag"])


    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--dataset_name", type=str, default="2wikimultihop")
    parser.add_argument("--llm_model", type=str, default="qwen/qwen3.6-plus")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--iteration_threshold", type=float, default=0.4)
    parser.add_argument("--passage_ratio", type=float, default=2)
    parser.add_argument("--top_k_sentence", type=int, default=3)
    parser.add_argument("--retrieval_top_k", type=int, default=5)
    parser.add_argument("--passage_node_weight", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--use_vectorized_retrieval", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=5000)
    parser.add_argument("--passage_length_norm_b", type=float, default=0.0)


    parser.add_argument("--ner_type", type=str, default="spacy", choices=["spacy", "llm"])

    parser.add_argument("--llm_ner_model", type=str, default="qwen/qwen3.5-9b")
    parser.add_argument("--llm_ner_seed_top_k", type=int, default=3)
    parser.add_argument("--llm_ner_seed_max_degree", type=int, default=300)
    parser.add_argument("--max_questions", type=int, default=0)


    parser.add_argument("--exp_alpha", type=float, default=0.6)
    parser.add_argument("--exp_top_k", type=int, default=5)
    parser.add_argument("--exp_beta", type=float, default=1.0)
    parser.add_argument("--exp_tau_s", type=float, default=0.3)
    parser.add_argument("--exp_mu", type=float, default=0.01)
    parser.add_argument("--exp_epsilon", type=float, default=0.05)
    parser.add_argument("--fusion_epsilon", type=float, default=0.05)
    parser.add_argument("--exp_activation_top_n", type=int, default=20)
    parser.add_argument("--living_family_serial", action="store_true", default=False)
    parser.add_argument("--use_temporal_decay", action="store_true", default=False)
    parser.add_argument("--tau_nli", type=float, default=0.7)
    parser.add_argument("--tau_gs", type=float, default=0.6)
    parser.add_argument("--tau_novel", type=float, default=0.7)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--tau_merge", type=float, default=0.9)
    parser.add_argument("--consolidate_interval", type=int, default=0)
    parser.add_argument("--nli_model_path", type=str, default="cross-encoder/nli-deberta-v3-large")
    parser.add_argument("--max_sentences", type=int, default=10)

    parser.add_argument("--scaffold_template_weight", type=float, default=0.0)

    return parser.parse_args()


def load_dataset(dataset_name, data_dir="dataset"):
    questions_path = f"{data_dir}/{dataset_name}/questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    chunks_path = f"{data_dir}/{dataset_name}/chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    passages = [f"{idx}:{chunk}" for idx, chunk in enumerate(chunks)]
    return questions, passages


def load_embedding_model(embedding_model_name):
    return SentenceTransformer(embedding_model_name, device="cuda")


def build_config(args, embedding_model, llm_model):
    base_kwargs = dict(
        dataset_name=args.dataset_name,
        working_dir=args.working_dir,
        batch_size=args.batch_size,
        embedding_model=embedding_model,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        llm_model=llm_model,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        passage_ratio=args.passage_ratio,
        top_k_sentence=args.top_k_sentence,
        retrieval_top_k=args.retrieval_top_k,
        passage_node_weight=args.passage_node_weight,
        damping=args.damping,
        use_vectorized_retrieval=args.use_vectorized_retrieval,
        ner_type=args.ner_type,
        llm_ner_model=args.llm_ner_model,
        llm_ner_seed_top_k=args.llm_ner_seed_top_k,
        llm_ner_seed_max_degree=args.llm_ner_seed_max_degree,
        passage_length_norm_b=args.passage_length_norm_b,
    )

    living_rag_kwargs = dict(
        exp_alpha=args.exp_alpha,
        exp_top_k=args.exp_top_k,
        exp_beta=args.exp_beta,
        exp_tau_s=args.exp_tau_s,
        exp_mu=args.exp_mu,
        exp_epsilon=args.exp_epsilon,
        fusion_epsilon=args.fusion_epsilon,
        use_temporal_decay=args.use_temporal_decay,
        exp_activation_top_n=args.exp_activation_top_n,
        living_family_serial=args.living_family_serial,
        tau_nli=args.tau_nli,
        tau_gs=args.tau_gs,
        tau_novel=args.tau_novel,
        max_retries=args.max_retries,
        tau_merge=args.tau_merge,
        consolidate_interval=args.consolidate_interval,
        nli_model_path=args.nli_model_path,
        max_sentences=args.max_sentences,
        scaffold_template_weight=args.scaffold_template_weight,
    )
    return LivingRAGConfig(**base_kwargs, **living_rag_kwargs)


def _log_config(args, config):
    logger = logging.getLogger(__name__)
    skip = {"embedding_model", "llm_model"}
    items = [f"model={args.model!r}"]
    llm = getattr(config, "llm_model", None)
    if llm is not None and hasattr(llm, "llm_config"):
        items.append(f"llm_model={llm.llm_config.get('model')!r}")
        items.append(f"max_tokens={llm.llm_config.get('max_tokens')!r}")
    items += [
        f"{f.name}={getattr(config, f.name)!r}"
        for f in dataclasses.fields(config)
        if f.name not in skip
    ]
    logger.info("Config:\n  %s", "\n  ".join(items))


def main():
    run_time = datetime.now()
    time_str = run_time.strftime("%Y-%m-%d_%H-%M-%S-%f")

    args = parse_arguments()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running LivingRAG.")
    if not api_key.isascii() or not api_key.isprintable() or any(c.isspace() for c in api_key) or api_key == "your-own-key":
        raise RuntimeError(
            "OPENAI_API_KEY is not a valid key value. Replace the example placeholder "
            "with your actual API key; do not include Chinese text, spaces, or newlines."
        )
    questions, passages = load_dataset(args.dataset_name, args.data_dir)
    embedding_model = load_embedding_model(args.embedding_model)

    if args.max_questions > 0:
        questions = questions[: args.max_questions]

    output_dir = os.path.join(args.results_dir, args.dataset_name, time_str)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(os.path.join(output_dir, "log.txt"))

    llm_model = LLM_Model(args.llm_model, max_tokens=args.max_tokens, reasoning_enabled=True, reasoning_effort="xhigh")
    config = build_config(args, embedding_model, llm_model)


    if args.model == "living_rag":
        config.exp_db_path = os.path.join(output_dir, "experience_memory.json")

    _log_config(args, config)

    rag_model = LivingRAG(global_config=config)


    rag_model.index(passages)


    questions = rag_model.qa(questions)

    predictions_path = os.path.join(output_dir, "predictions.json")
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)


    eval_llm_model = LLM_Model(args.llm_model, max_tokens=args.max_tokens, reasoning_enabled=False, reasoning_effort=None)
    evaluator = Evaluator(llm_model=eval_llm_model, predictions_path=predictions_path, dataset_name=args.dataset_name)
    evaluator.evaluate(max_workers=args.max_workers)


    token_csv_path = os.path.join(output_dir, "token_usage.csv")
    with open(token_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_idx",
            "prompt_tokens", "completion_tokens", "reasoning_tokens",
            "llm_accuracy", "contain_accuracy", "answer_parsed",
        ])
        for idx, r in enumerate(evaluator.prediction_results):
            writer.writerow([
                idx + 1,
                r.get("prompt_tokens", 0),
                r.get("completion_tokens", 0),
                r.get("reasoning_tokens", 0),
                r.get("llm_accuracy", ""),
                r.get("contain_accuracy", ""),
                int(bool(r.get("answer_parsed", True))),
            ])


if __name__ == "__main__":
    main()
