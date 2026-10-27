from hashlib import md5
import httpx
from openai import OpenAI
import re
import string
import logging
import numpy as np
import os
import time

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()

class LLM_Model:
    def __init__(self, llm_model, max_tokens=5000, reasoning_enabled=True, reasoning_effort="xhigh"):
        http_client = httpx.Client(timeout=60.0, trust_env=True)
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            http_client=http_client
        )
        reasoning_config = {"enabled": reasoning_enabled}
        if reasoning_effort is not None:
            reasoning_config["effort"] = reasoning_effort
        self.llm_config = {
            "model": llm_model,
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "extra_body": {"reasoning": reasoning_config},
        }

    def infer(self, messages, max_retries=5):
        logger = logging.getLogger(__name__)
        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(**self.llm_config, messages=messages)
                msg = response.choices[0].message
                content = msg.content or ""
                reasoning = getattr(msg, "reasoning", None)
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                reasoning_tokens = (
                    usage.completion_tokens_details.reasoning_tokens
                    if usage and usage.completion_tokens_details
                    else 0
                )
                return content, reasoning, prompt_tokens, completion_tokens, reasoning_tokens
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"LLM request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait)
        raise RuntimeError("LLM request failed after all retries; check API access and model availability.")


def normalize_answer(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def setup_logging(log_file):
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    handlers = [logging.StreamHandler()]
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val


    if range_val == 0:
        return np.ones_like(x)

    return (x - min_val) / range_val
