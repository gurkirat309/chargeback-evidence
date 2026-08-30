"""
Phase 4 — LLM letter generator (CLAUDE.md section 11).

Receives ONLY the assembled evidence bundle and the reason code. Returns
structured JSON: a list of {claim, artifact_id} plus a short framing paragraph.
It does NOT receive p_win, does NOT decide anything, and cannot cite an artifact
that is absent (it only ever sees present artifacts). Response cached by input.
"""
from __future__ import annotations

import json

import llm_client as LC
from evidence_agent import Bundle, bundle_for_generator

CFG = LC.LLM_CFG["generator"]

SYSTEM = (
    "You draft chargeback representment (dispute rebuttal) letters for a merchant. "
    "You are defense-only. You will be given a reason code and a numbered list of "
    "evidence artifacts, each with an artifact_id and a factual statement. "
    "Write ONLY claims that are directly supported by a specific artifact, and cite "
    "that artifact_id. Never invent facts, never cite an artifact_id that is not in "
    "the list, and do not assert anything an artifact contradicts (e.g. if AVS is "
    "'no match', do not claim the address matched). If the evidence is weak, write "
    "fewer claims — never pad. Respond with STRICT JSON only, no prose outside JSON."
)

USER_TEMPLATE = (
    "Reason code: {reason_code}\n\n"
    "Evidence artifacts:\n{artifacts}\n\n"
    "Return JSON with this exact shape:\n"
    '{{\n'
    '  "framing": "<=2 sentence framing paragraph, no specific factual claims>",\n'
    '  "claims": [{{"claim": "<one supported sentence>", "artifact_id": "<ART-...>"}}]\n'
    '}}'
)


def _format_artifacts(payload) -> str:
    return "\n".join(
        f"- {a['artifact_id']}: {a['statement']}" for a in payload["artifacts"])


def generate_letter(bundle: Bundle, force_refresh=False) -> dict:
    """Return {'framing': str, 'claims': [{claim, artifact_id}], 'cached': bool}."""
    payload = bundle_for_generator(bundle)     # reason_code + present artifacts only
    user = USER_TEMPLATE.format(
        reason_code=payload["reason_code"], artifacts=_format_artifacts(payload))
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]

    resp = LC.chat(CFG["model"], messages, CFG["temperature"], CFG["max_tokens"],
                   reasoning_effort=CFG.get("reasoning_effort"),
                   force_refresh=force_refresh)
    data = LC.extract_json(resp["content"])

    framing = str(data.get("framing", "")).strip()
    claims = []
    for c in data.get("claims", []):
        if isinstance(c, dict) and "claim" in c and "artifact_id" in c:
            claims.append({"claim": str(c["claim"]).strip(),
                           "artifact_id": str(c["artifact_id"]).strip()})
    return {"framing": framing, "claims": claims, "cached": resp["cached"]}
