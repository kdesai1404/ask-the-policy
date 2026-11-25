"""OpenAI-compatible chat client with retry + on-disk cache (saves quota on re-runs)."""
import hashlib
import json
import time

from openai import OpenAI

from . import config

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY is not set. Copy .env.example to .env and fill it in.")
        _client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    return _client


def _model_kwargs(max_tokens: int) -> tuple[dict, int]:
    """gpt-oss models 'think' before answering, and the thinking counts against max_tokens.
    Keep the thinking short and leave headroom, otherwise answers can come back empty."""
    if "gpt-oss" in config.LLM_MODEL:
        return {"reasoning_effort": "low"}, max(max_tokens, 1500)
    return {}, max_tokens


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 700,
         use_cache: bool = True) -> str:
    key = hashlib.sha256(
        json.dumps([config.LLM_MODEL, messages, temperature, max_tokens],
                   ensure_ascii=False).encode()
    ).hexdigest()
    path = config.CACHE_DIR / f"{key}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["text"]

    extra, budget = _model_kwargs(max_tokens)
    for attempt in range(5):
        try:
            resp = _get_client().chat.completions.create(
                model=config.LLM_MODEL, messages=messages,
                temperature=temperature, max_tokens=budget, **extra,
            )
            text = resp.choices[0].message.content or ""
            if use_cache:
                path.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
            return text
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)  # simple exponential backoff for rate limits
    return ""
