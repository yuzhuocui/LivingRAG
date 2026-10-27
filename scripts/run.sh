#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/run.sh DATASET [run.py options]" >&2
  exit 2
fi
dataset="$1"
shift
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY first.}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
case "$dataset" in
  2wikimultihop) options=(--top_k_sentence 1 --passage_ratio 0.05 --exp_beta 0.1 --max_sentences 5 --scaffold_template_weight 0.5) ;;
  hotpotqa) options=(--top_k_sentence 1 --passage_ratio 0.05 --exp_top_k 3 --exp_beta 0.25 --fusion_epsilon 0.08 --exp_activation_top_n 10 --max_sentences 5 --scaffold_template_weight 0.05) ;;
  musique) options=(--max_iterations 5 --top_k_sentence 4 --iteration_threshold 0.1 --exp_top_k 3 --fusion_epsilon 0.08 --exp_activation_top_n 10 --max_sentences 5 --scaffold_template_weight 0.05) ;;
  musique_full) options=(--retrieval_top_k 8 --top_k_sentence 2 --passage_ratio 5.0 --iteration_threshold 0.15 --exp_top_k 3 --fusion_epsilon 0.08 --exp_activation_top_n 10 --living_family_serial --max_sentences 5 --scaffold_template_weight 0.05) ;;
  wixqa) options=(--llm_ner_model qwen/qwen3.5-9b --retrieval_top_k 15 --top_k_sentence 2 --passage_node_weight 0.4 --iteration_threshold 0.1 --ner_type llm --passage_length_norm_b 0.1 --exp_epsilon 0.1 --scaffold_template_weight 0.2) ;;
  *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac
model_dir="${MODEL_DIR:-model}"
data_dir="${DATA_DIR:-dataset}"
options+=(--data_dir "$data_dir")

exec python run.py --dataset_name "$dataset" --llm_model qwen/qwen3.6-plus \
  --embedding_model "$model_dir/all-mpnet-base-v2" \
  --nli_model_path "$model_dir/nli-deberta-v3-large" "${options[@]}" "$@"
