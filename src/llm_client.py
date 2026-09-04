"""
Phase 4 — cached LLM client (shared helper for llm_generator + llm_verifier).

NOTE: not in the CLAUDE.md section 14 layout — a small shared client so the two
LLM modules don't duplicate the Groq + cache plumbing. Flagged deliberately.

Provider is Groq (CLAUDE.md section 11 names Anthropic; using Groq per user
direction). Every response is cached to data/llm_cache/ keyed by a SHA-256 hash
of the full input (model + messages + params). A cache HIT needs no key and no
network, so a fully-warmed demo never depends on a live API call. The API key is
read from .env (GROQ_API_KEY) and is only required on a cache MISS.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

with open(ROOT / "config" / "llm.yaml", "r", encoding="utf-8") as _fh:
    LLM_CFG = yaml.safe_load(_fh)

CACHE_DIR = ROOT / LLM_CFG["cache_dir"]
try:                                 # create once at import, not on every call
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
_client = None                       # lazily created only on a cache miss


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not set and response not cached — a live call is "
                "needed but no key is available. Add it to .env.")
        _client = Groq(api_key=key)
    return _client


def _cache_key(model, messages, temperature, max_tokens, reasoning_effort) -> str:
    blob = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature,
         "max_tokens": max_tokens, "reasoning_effort": reasoning_effort},
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def chat(model, messages, temperature, max_tokens, reasoning_effort=None,
         force_refresh=False) -> dict:
    """Return {'content': str, 'cached': bool, 'key': str}. Cached by input hash."""
    key = _cache_key(model, messages, temperature, max_tokens, reasoning_effort)
    path = CACHE_DIR / f"{key}.json"

    if path.exists() and not force_refresh:
        rec = json.loads(path.read_text(encoding="utf-8"))
        return {"content": rec["content"], "cached": True, "key": key}

    client = _get_client()
    kwargs = dict(model=model, messages=messages, temperature=temperature,
                  max_tokens=max_tokens, timeout=LLM_CFG["request_timeout_s"])
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    last_err = None
    for attempt in range(LLM_CFG["max_retries"]):
        try:
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            content = msg.content or getattr(msg, "reasoning", None) or ""
            if not content.strip():                       # empty -> retry, don't cache
                raise RuntimeError("empty model response")
            path.write_text(json.dumps(
                {"model": model, "messages": messages, "temperature": temperature,
                 "max_tokens": max_tokens, "reasoning_effort": reasoning_effort,
                 "content": content}, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"content": content, "cached": False, "key": key}
        except Exception as exc:                          # noqa: BLE001 - retry any API error
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Groq call failed after retries: {last_err}")


def extract_json(text: str):
    """Parse a JSON object from a model response, tolerating ```json fences and
    surrounding prose. Fails loudly (no silent fallback) if none is found."""
    t = text.strip()
    if "```" in t:                                        # strip code fences
        parts = t.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") or p.startswith("["):
                t = p
                break
    start = min([i for i in (t.find("{"), t.find("[")) if i != -1], default=-1)
    if start == -1:
        raise ValueError(f"no JSON found in model response: {text[:200]!r}")
    # find the matching close by scanning
    end = t.rfind("}") if t[start] == "{" else t.rfind("]")
    return json.loads(t[start:end + 1])
