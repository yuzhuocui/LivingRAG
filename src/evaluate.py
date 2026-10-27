import json
import os
from src.utils import normalize_answer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

_WIXQA_SYSTEM_PROMPT = (
    "You are an expert evaluator for Wix customer support answers. "
    "Your task is to judge whether a generated answer correctly resolves the customer's issue, "
    "compared to an official gold answer extracted from a Wix help article."
)

_WIXQA_USER_PROMPT = """\
Evaluate whether the generated answer is correct for this Wix customer support question.

Generated answer:
{pred}

Gold answer (from official Wix help article):
{gold}

Mark as CORRECT only if ALL of the following hold:
1. The same Wix product / feature is recommended (e.g. "Wix Bookings" vs "Wix Stores" is a meaningful difference).
2. The core action described would guide the customer to actually solve their problem.
3. It does not claim something is impossible when the gold shows it is possible (or vice versa).
4. It does not contain factually incorrect steps that would mislead the customer.

Mark as INCORRECT if ANY of the following hold:
- It recommends the wrong Wix product or feature.
- It describes the wrong operation (e.g. "delete a collection" vs "delete items from a collection").
- It contradicts the gold on a key fact (e.g. a feature exists vs does not exist).
- It provides outdated or clearly wrong information.

Leniency: the generated answer does NOT need to reproduce every step from the gold. Partial coverage of the correct solution is acceptable as long as the main direction is right and no incorrect information is given.

Respond with ONLY 'correct' or 'incorrect'.\
"""

_GENERIC_SYSTEM_PROMPT = "You are an expert evaluator."

_GENERIC_USER_PROMPT = """\
Please evaluate if the generated answer is correct by comparing it with the gold answer.
Generated answer: {pred}
Gold answer: {gold}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

Respond with ONLY 'correct' or 'incorrect'.
Response:\
"""


class Evaluator:
    def __init__(self, llm_model, predictions_path, dataset_name: str = ""):
        self.llm_model = llm_model
        self.predictions_path = predictions_path
        self.dataset_name = dataset_name.lower()
        self.prediction_results = self.load_predictions()

    def load_predictions(self):
        prediction_results = json.load(open(self.predictions_path))
        return prediction_results

    def _get_prompts(self, pre_answer: str, gold_ans: str):
        if self.dataset_name == "wixqa":
            system = _WIXQA_SYSTEM_PROMPT
            user = _WIXQA_USER_PROMPT.format(pred=pre_answer, gold=gold_ans)
        else:
            system = _GENERIC_SYSTEM_PROMPT
            user = _GENERIC_USER_PROMPT.format(pred=pre_answer, gold=gold_ans)
        return system, user

    def calculate_llm_accuracy(self, pre_answer, gold_ans):
        system_prompt, user_prompt = self._get_prompts(pre_answer, gold_ans)
        response, _, _, _, _ = self.llm_model.infer(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        return 1.0 if response.strip().lower() == "correct" else 0.0

    def calculate_contain(self,pre_answers,gold_ans):
        if pre_answers is None or pre_answers == "" or (isinstance(pre_answers, str) and pre_answers.strip() == ""):
            return 0
        if gold_ans is None or gold_ans == "" or (isinstance(gold_ans, str) and gold_ans.strip() == ""):
            return 0
        s1 = normalize_answer(pre_answers)
        s2 = normalize_answer(gold_ans)
        if s2 in s1:
            return 1
        else:
            return 0

    def evaluate_sig_sample(self,idx,prediction):
        pre_answer = prediction["pred_answer"]
        gold_ans = prediction["gold_answer"]

        llm_acc = self.calculate_llm_accuracy(pre_answer, gold_ans)
        contain_acc = self.calculate_contain(pre_answer, gold_ans)
        return idx, llm_acc, contain_acc

    def evaluate(self,max_workers):
        llm_scores = [0.0] * len(self.prediction_results)
        contain_scores = [0.0] * len(self.prediction_results)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.evaluate_sig_sample, idx, pred): idx
                for idx, pred in enumerate(self.prediction_results)
            }

            completed = 0
            total_llm_score = 0.0
            total_contain_score = 0.0
            pbar = tqdm(total=len(futures), desc="Evaluating samples", unit="sample", ncols=80)
            for future in as_completed(futures):
                idx, llm_acc, contain_acc  = future.result()
                llm_scores[idx] = llm_acc
                contain_scores[idx] = contain_acc
                self.prediction_results[idx]["llm_accuracy"] = llm_acc
                self.prediction_results[idx]["contain_accuracy"] = contain_acc
                total_llm_score += llm_acc
                total_contain_score += contain_acc
                completed += 1
                current_llm_acc = total_llm_score / completed
                current_contain_acc = total_contain_score / completed
                pbar.set_postfix({
                    'LLM_Acc': f'{current_llm_acc:.3f}',
                    'Contain_Acc': f'{current_contain_acc:.3f}'
                })
                pbar.update(1)
            pbar.close()

        llm_accuracy = sum(llm_scores) / len(llm_scores)
        contain_accuracy = sum(contain_scores) / len(contain_scores)


        parse_stats = {}
        if all("answer_parsed" in r for r in self.prediction_results):
            n_parsed = sum(1 for r in self.prediction_results if r["answer_parsed"])
            n_total = len(self.prediction_results)
            answer_parse_rate = n_parsed / n_total
            parse_stats["answer_parse_rate"] = round(answer_parse_rate, 4)
            parse_stats["answer_parse_count"] = n_parsed
            parse_stats["answer_parse_fail_count"] = n_total - n_parsed
            logger.info(
                f"  Answer Parse Rate: {answer_parse_rate:.4f} "
                f"({n_parsed}/{n_total} responses returned non-empty content)"
            )


        if all("prompt_tokens" in r for r in self.prediction_results):
            total_prompt = sum(r["prompt_tokens"] for r in self.prediction_results)
            total_completion = sum(r["completion_tokens"] for r in self.prediction_results)
            total_reasoning = sum(r["reasoning_tokens"] for r in self.prediction_results)
            n = len(self.prediction_results)
            parse_stats["avg_prompt_tokens"] = round(total_prompt / n, 1)
            parse_stats["avg_completion_tokens"] = round(total_completion / n, 1)
            parse_stats["avg_reasoning_tokens"] = round(total_reasoning / n, 1)
            parse_stats["total_prompt_tokens"] = total_prompt
            parse_stats["total_completion_tokens"] = total_completion
            parse_stats["total_reasoning_tokens"] = total_reasoning
            logger.info(
                f"  Token Usage (avg per question): "
                f"prompt={total_prompt/n:.0f}  "
                f"completion={total_completion/n:.0f}  "
                f"reasoning={total_reasoning/n:.0f}  "
                f"(reasoning ratio={total_reasoning/max(total_completion,1)*100:.1f}%)"
            )

        logger.info(f"Evaluation Results:")
        logger.info(f"  LLM Accuracy: {llm_accuracy:.4f} ({sum(llm_scores)}/{len(llm_scores)})")
        logger.info(f"  Contain Accuracy: {contain_accuracy:.4f} ({sum(contain_scores)}/{len(contain_scores)})")

        with open(self.predictions_path, "w", encoding="utf-8") as f:
            json.dump(self.prediction_results, f, ensure_ascii=False, indent=4)

        eval_results = {
            "llm_accuracy": llm_accuracy,
            "contain_accuracy": contain_accuracy,
            **parse_stats,
        }
        with open(os.path.join(os.path.dirname(self.predictions_path), "evaluation_results.json"), "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=4)
        return llm_accuracy, contain_accuracy
