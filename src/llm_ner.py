from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import md5
from pathlib import Path
from typing import Optional

import httpx
from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _naive_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 10]


_PASSAGE_SYSTEM = (
    "You are an entity extraction assistant for Wix product documentation. "
    "Extract ALL of the following from the given text:\n\n"
    "1. Named product / feature / UI names "
    "(e.g. 'Wix Events', 'Event Details Page', 'Dashboard', 'DNS records', 'CMS collection')\n"
    "2. The main topic or action concept of this article as 1–3 short phrases that would match "
    "a user's question about this content "
    "(e.g. 'Wix Payments account verification', 'adding funds to cover refund', "
    "'publishing pages to live site', 'staff working hours', 'CMS collection item order')\n\n"
    "Return ONLY a JSON array combining all extracted terms. No explanation, no markdown fences.\n"
    "Example — article about verifying a Wix Payments account:\n"
    "[\"Wix Payments\", \"Wix Payments account verification\", \"account verification status\", "
    "\"Accept Payments\", \"Adyen\", \"Stripe\"]"
)

_QUESTION_SYSTEM = (
    "You are a Wix documentation entity extractor. "
    "Extract 3–5 key terms from the user's question that best identify the relevant help article.\n\n"
    "CRITICAL: Your terms must match the vocabulary used in Wix help article entity labels — "
    "phrase them as they appear in article titles or article-level tags, NOT as the user phrased them.\n\n"
    "Priority order:\n"
    "1. Exact canonical Wix product / feature name — use the official Wix name, not user paraphrase\n"
    "   ('Google Workspace' not 'Gmail business'; 'Wix Bookings' not 'booking system'; "
    "'Email Marketing Plan' not 'email plan')\n"
    "2. A specific action-topic phrase describing the article's main subject\n"
    "   Good: 'resetting Workspace user password', 'failed payout', 'blocking off time in Booking Calendar',\n"
    "         'Publishing Issues', 'Premium Vouchers page', 'upgrade Email Marketing Plan'\n"
    "   Bad:  'password reset', 'payment', 'block time', 'page issue', 'voucher', 'upgrade'\n"
    "3. Precise technical terms only if clearly stated (e.g. 'TXT record', 'iCal URL', 'DNS settings')\n\n"
    "Rules:\n"
    "- Match article vocabulary: 'Google Workspace' → 'Google Workspace'; "
    "'Gmail' in an admin/business context → 'Google Workspace'\n"
    "- Use full noun phrases, not single generic nouns: "
    "'Email Marketing Plan upgrade' > 'email'; 'payout delay' > 'payment'\n"
    "- Do NOT include ultra-generic terms: 'site', 'page', 'account', 'email', 'help', "
    "'settings', 'Payments' (alone), 'Gmail' (alone for business email questions)\n"
    "- Do NOT invent irrelevant terms not directly related to the user's problem\n"
    "- Return ONLY a JSON array of strings, max 5 items. No explanation, no markdown fences.\n\n"
    "Examples:\n"
    "Q: 'I forgot the password for my second Gmail business account and I have an admin account.'\n"
    "Output: [\"Google Workspace\", \"resetting Workspace user password\", "
    "\"Google Workspace admin console\"]\n\n"
    "Q: 'I have not received my payment in my bank account yet.'\n"
    "Output: [\"Wix Payments\", \"payout delay\", \"failed payout\", \"Payouts On Hold\"]\n\n"
)


class LlmNER:

    def __init__(
        self,
        model: str = "qwen/qwen3.5-9b",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        cache_path: Optional[str | Path] = None,
        request_interval: float = 0.05,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._request_interval = request_interval
        self._last_request_time = 0.0

        http_client = httpx.Client(timeout=60.0, trust_env=True)
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            http_client=http_client,
        )


        self._cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, list[str]] = {}
        if self._cache_path and self._cache_path.exists():
            try:
                with open(self._cache_path, encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info("[LlmNER] Loaded %d cached entries from %s", len(self._cache), self._cache_path)
            except Exception as e:
                logger.warning("[LlmNER] Could not load cache: %s", e)

        logger.info("[LlmNER] Initialised — model=%s", model)


    def _cache_key(self, text: str) -> str:
        return md5(text.encode()).hexdigest()

    def _save_cache(self) -> None:
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed)
        self._last_request_time = time.time()

    def _call_llm(self, system: str, user_text: str, max_retries: int = 4) -> list[str]:
        key = self._cache_key(system[:30] + user_text)
        if key in self._cache:
            return self._cache[key]

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    extra_body={"reasoning": {"enabled": False}},
                )
                raw = (response.choices[0].message.content or "").strip()
                entities = self._parse_json_list(raw)
                self._cache[key] = entities
                return entities
            except Exception as e:
                wait = 2 ** attempt
                logger.warning("[LlmNER] API error (attempt %d/%d): %s — retrying in %ds", attempt + 1, max_retries, e, wait)
                if attempt < max_retries - 1:
                    time.sleep(wait)

        logger.error("[LlmNER] All retries exhausted for input: %s…", user_text[:60])
        raise RuntimeError("NER request failed after all retries; check API access and the NER model.")

    @staticmethod
    def _parse_json_list(raw: str) -> list[str]:

        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()


        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return [str(e).strip() for e in result if str(e).strip()]
        except json.JSONDecodeError:
            pass


        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
                if isinstance(result, list):
                    return [str(e).strip() for e in result if str(e).strip()]
            except json.JSONDecodeError:
                pass


        return re.findall(r'"([^"]+)"', raw)

    def _process_passage(self, args: tuple[str, str]) -> tuple[str, list[str], dict[str, list[str]]]:
        hash_id, passage_text = args


        passage_entities = self._call_llm(_PASSAGE_SYSTEM, passage_text)


        sentence_to_entities: dict[str, list[str]] = {}
        if passage_entities:
            for sent in _naive_sentences(passage_text):
                sent_lower = sent.lower()
                sent_ents = [e for e in passage_entities if e.lower() in sent_lower]
                if sent_ents:
                    sentence_to_entities[sent] = sent_ents

        return hash_id, passage_entities, sentence_to_entities


    def batch_ner(
        self,
        hash_id_to_passage: dict[str, str],
        max_workers: int,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        passage_hash_id_to_entities: dict[str, list[str]] = {}
        sentence_to_entities: defaultdict[str, list[str]] = defaultdict(list)

        items = list(hash_id_to_passage.items())


        uncached = [(hid, txt) for hid, txt in items if self._cache_key(_PASSAGE_SYSTEM[:30] + txt) not in self._cache]
        cached_count = len(items) - len(uncached)
        if cached_count:
            logger.info("[LlmNER] %d/%d passages already cached, skipping LLM calls", cached_count, len(items))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_passage, item): item[0] for item in uncached}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Indexing LlmNER", ncols=80):
                hash_id, p_ents, s_ents = future.result()
                passage_hash_id_to_entities[hash_id] = p_ents
                for sent, ents in s_ents.items():
                    for e in ents:
                        if e not in sentence_to_entities[sent]:
                            sentence_to_entities[sent].append(e)


        if cached_count:
            for hash_id, passage_text in tqdm(
                [(hid, txt) for hid, txt in items if hid not in passage_hash_id_to_entities],
                desc="Loading cached passages",
                ncols=80,
            ):
                _, p_ents, s_ents = self._process_passage((hash_id, passage_text))
                passage_hash_id_to_entities[hash_id] = p_ents
                for sent, ents in s_ents.items():
                    for e in ents:
                        if e not in sentence_to_entities[sent]:
                            sentence_to_entities[sent].append(e)

        self._save_cache()
        return passage_hash_id_to_entities, dict(sentence_to_entities)

    def question_ner(self, question: str) -> set[str]:
        entities = self._call_llm(_QUESTION_SYSTEM, question)
        return set(e.lower() for e in entities)

    def extract_entities_from_text(self, text: str) -> list[str]:
        entities = self._call_llm(_QUESTION_SYSTEM, text)
        seen: set[str] = set()
        result: list[str] = []
        for e in entities:
            el = e.lower()
            if el not in seen:
                seen.add(el)
                result.append(el)
        return result
