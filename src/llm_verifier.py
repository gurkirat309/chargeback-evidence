"""
Phase 4 — LLM claim verifier (CLAUDE.md section 11).

A SEPARATE call (different, smaller model). For each claim it receives the claim
and its cited artifact only, and returns {supported: bool, reason: str}. A claim
whose artifact_id does not resolve is unsupported by construction (no artifact to
check). Unsupported claims are STRIPPED before the packet is assembled — the
letter gets shorter, never invented. Responses cached by input.
"""
from __future__ import annotations

import llm_client as LC
from evidence_agent import Bundle

CFG = LC.LLM_CFG["verifier"]

SYSTEM = (
    "You are a strict claim verifier for chargeback letters. You are given ONE "
    "claim and the ONE evidence artifact it cites. Decide whether the artifact "
    "directly supports the claim. Be strict: if the artifact contradicts the claim, "
    "is unrelated, or only partially supports it, mark it unsupported. Respond with "
    'STRICT JSON only: {"supported": true|false, "reason": "<short reason>"}.'
)

USER_TEMPLATE = (
    "Claim: {claim}\n"
    "Cited artifact ({artifact_id}): {statement}\n\n"
    'Return JSON: {{"supported": true|false, "reason": "<short>"}}'
)


def _verify_one(claim_text: str, artifact_id: str, statement: str,
                force_refresh=False) -> dict:
    user = USER_TEMPLATE.format(claim=claim_text, artifact_id=artifact_id,
                                statement=statement)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
    resp = LC.chat(CFG["model"], messages, CFG["temperature"], CFG["max_tokens"],
                   reasoning_effort=CFG.get("reasoning_effort"),
                   force_refresh=force_refresh)
    data = LC.extract_json(resp["content"])
    return {"supported": bool(data.get("supported", False)),
            "reason": str(data.get("reason", "")).strip(),
            "cached": resp["cached"]}


def verify_claims(claims, bundle: Bundle, force_refresh=False) -> list:
    """Return per-claim records with support verdicts. A claim citing a
    non-existent artifact is unsupported without an API call."""
    results = []
    for c in claims:
        art = bundle.resolve(c["artifact_id"])
        if art is None:
            results.append({**c, "resolves": False, "supported": False,
                            "reason": "cited artifact_id does not resolve",
                            "cached": True})
            continue
        v = _verify_one(c["claim"], c["artifact_id"], art.statement, force_refresh)
        results.append({**c, "resolves": True, **v})
    return results


def build_packet(bundle: Bundle, framing: str, verified: list) -> dict:
    """Assemble the final packet: framing + only the SUPPORTED, resolving claims."""
    kept = [v for v in verified if v["resolves"] and v["supported"]]
    return {
        "dispute_id": bundle.dispute_id,
        "reason_code": bundle.reason_code,
        "framing": framing,
        "claims": [{"claim": v["claim"], "artifact_id": v["artifact_id"],
                    "statement": bundle.resolve(v["artifact_id"]).statement}
                   for v in kept],
        "n_generated": len(verified),
        "n_kept": len(kept),
    }
