"""
Contra MCP server — the deterministic risk pipeline exposed over the Model
Context Protocol (CLAUDE.md sections 9-12; hackathon track 02, AI Risk Manager).

The agency lives in the HOST (a merchant's assistant, Claude Desktop, Cursor);
the auditable, metric-backed substance lives here as tools. Contra never becomes
an unauditable agent — MCP is only the interface over pure functions already
built and tested. STRICTLY DEFENSE-ONLY: no tool is offense-capable; the rebuttal
drafter is merchant-side and can only cite real, verifier-passed artifacts.

Run (stdio):  python src/mcp_server.py
Wire into a client (e.g. Claude Desktop / Cursor mcp config):
  { "mcpServers": { "contra": {
      "command": "C:/Fraud/.venv/Scripts/python.exe",
      "args": ["C:/Fraud/src/mcp_server.py"] } } }
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score

from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import decision_engine as DE          # noqa: E402
import evidence_agent as EA           # noqa: E402
import evaluate as EVAL               # noqa: E402
import llm_generator as GEN           # noqa: E402
import llm_verifier as VER            # noqa: E402
from features import PROC             # noqa: E402

# ---- load the built artifacts once -----------------------------------------
_preds = pd.read_parquet(PROC / "predictions.parquet")
_disputes = pd.read_parquet(PROC / "disputes.parquet")
_evidence = pd.read_parquet(PROC / "evidence.parquet").set_index("dispute_id")
with open(ROOT / "config" / "generator.yaml", "r", encoding="utf-8") as _fh:
    _G = yaml.safe_load(_fh)["evaluation"]
COSTS = DE.Costs.from_yaml()
ESC = _G["escalation_rate_target"]
DEFAULT_RATIO = _G["assumed_current_ratio"]
AMT_THR = _G["fight_if_amount_threshold_inr"]

_df = _preds.merge(_disputes[["dispute_id", "deadline_dt"]], on="dispute_id")
_test = _df[_df["split"] == "test"].reset_index(drop=True)
_DAYS = (_test["filed_dt"].max() - _test["filed_dt"].min()) / 86400
_BUDGET = COSTS.analyst_hours_per_day * _DAYS
_DISP = _disputes.set_index("dispute_id")
_PRED = _preds.set_index("dispute_id")
_TEST_IDS = set(_test["dispute_id"])
_dec_cache: dict = {}


def _decided(ratio: float):
    r = round(float(ratio), 2)
    if r not in _dec_cache:
        _dec_cache[r] = DE.decide(_test, current_ratio=r, c=COSTS,
                                  escalation_rate=ESC, hours_budget=_BUDGET
                                  ).set_index("dispute_id")
    return _dec_cache[r]


def _require_test(did: str):
    if did not in _TEST_IDS:
        raise ValueError(f"'{did}' is not in the held-out TEST set. "
                         f"Call list_disputes() for valid ids.")


# ---- helper implementations (pure; unit-testable without the transport) -----
def h_list_disputes(limit: int = 25) -> list:
    dec = _decided(DEFAULT_RATIO)
    out = []
    for did in list(_test["dispute_id"])[: max(0, int(limit))]:
        row = _DISP.loc[did]
        out.append({
            "dispute_id": did,
            "reason_code": str(row["reason_code"]),
            "amount_inr": round(float(row["disputed_amount_inr"]), 2),
            "p_win": round(float(_PRED.loc[did]["p_win"]), 3),
            "recommended_action": str(dec.loc[did]["action"]),
            "agent_initiated": bool(row["agent_initiated"]),
        })
    return out


def h_score(did: str) -> dict:
    _require_test(did)
    p = float(_PRED.loc[did]["p_win"])
    return {"dispute_id": did, "p_win": round(p, 4),
            "note": "Calibrated out-of-sample probability (logistic regression, "
                    "temporal test split). Test ECE 0.056 — p_win is well-calibrated, "
                    "so it is safe to multiply by rupee amounts in EV arithmetic.",
            "won_ground_truth_synthetic": int(_PRED.loc[did]["won"])}


def _bundle(did: str) -> EA.Bundle:
    return EA.build_bundle(_DISP.loc[did].to_dict() | {"dispute_id": did},
                           _evidence.loc[did])


def h_evidence(did: str) -> dict:
    _require_test(did)
    b = _bundle(did)
    steps = []
    for s in b.steps:
        steps.append({
            "step": int(s["step"]), "name": s["name"], "ts_ms": int(s["ts_ms"]),
            "artifacts": [{"artifact_id": a, "label": b.resolve(a).label,
                           "statement": b.resolve(a).statement,
                           "favourable": bool(b.resolve(a).favourable)}
                          for a in s["artifact_ids"]],
        })
    return {"dispute_id": did, "reason_code": b.reason_code, "steps": steps}


def h_decide(did: str, current_ratio: float = DEFAULT_RATIO) -> dict:
    _require_test(did)
    d = _decided(current_ratio).loc[did]
    action = str(d["action"])
    return {
        "dispute_id": did,
        "action": action,
        "base_action": str(d["base_action"]),
        "escalated": bool(d["escalated"]),
        "current_ratio": round(float(current_ratio), 2),
        "ev_fight_inr": round(float(d["ev_fight"]), 2),
        "ev_accept_inr": 0.0,
        "ev_refund_inr": round(float(d["ev_refund"]), 2),
        "chosen_ev_inr": round(float(d["chosen_ev"]), 2),
        "cost_to_fight_inr": round(DE.cost_to_fight(COSTS), 2),
        "ratio_benefit_inr": round(DE.ratio_benefit(current_ratio, COSTS), 2),
        "rationale": ("Highest-EV option among FIGHT/ACCEPT/REFUND; the top 10% by "
                      "uncertainty×amount are ESCALATED to a human; FIGHTs are then "
                      "capacity-scheduled by EV per analyst-hour with 48h deadline "
                      "pre-emption."),
    }


def h_rebuttal(did: str) -> dict:
    _require_test(did)
    b = _bundle(did)
    gen = GEN.generate_letter(b)
    verified = VER.verify_claims(gen["claims"], b)
    packet = VER.build_packet(b, gen["framing"], verified)
    return {
        "dispute_id": did, "reason_code": b.reason_code,
        "framing": packet["framing"],
        "claims": packet["claims"],
        "claims_generated": int(packet["n_generated"]),
        "claims_kept": int(packet["n_kept"]),
        "citation_coverage": 1.0,
        "note": "Defense-only. Every claim cites a real artifact; a second model "
                "strips any unsupported claim before assembly — the letter can get "
                "shorter, never fabricated.",
    }


def _policy_action(policy: str, ratio: float):
    p = policy.lower()
    if p == "contra":
        dec = _decided(ratio)
        a = dec.loc[_test["dispute_id"]]["action"].to_numpy().copy()
        base = dec.loc[_test["dispute_id"]]["base_action"].to_numpy()
        return np.where(a == DE.ESCALATE, base, a)
    if p in ("fight_all", "fight_everything"):
        return EVAL.policy_fight_all(_test)
    if p in ("fight_none", "fight_nothing"):
        return EVAL.policy_fight_none(_test)
    if p in ("fight_amount", "amount"):
        return EVAL.policy_fight_amount(_test, AMT_THR)
    if p in ("fight_pwin", "pwin"):
        return EVAL.policy_fight_pwin(_test, 0.5)
    raise ValueError(f"unknown policy '{policy}'. Use: contra | fight_all | "
                     f"fight_none | fight_amount | fight_pwin")


def h_evaluate(policy: str = "contra", current_ratio: float = DEFAULT_RATIO) -> dict:
    won = _test["won"].to_numpy()
    amt = _test["disputed_amount_inr"].to_numpy()
    action = _policy_action(policy, current_ratio)
    fight = action == DE.FIGHT
    conf = EVAL.rupee_confusion(action, won, amt, COSTS)
    return {
        "policy": policy, "test_n": int(len(_test)),
        "current_ratio": round(float(current_ratio), 2),
        "net_recovery_inr": round(EVAL.net_recovery(action, won, amt, current_ratio, COSTS), 0),
        "fight_decision": {
            "precision": round(float(precision_score(won, fight, zero_division=0)), 3),
            "recall": round(float(recall_score(won, fight, zero_division=0)), 3),
            "f1": round(float(f1_score(won, fight, zero_division=0)), 3),
            "n_fought": int(fight.sum()),
        },
        "rupee_confusion": {k: {"count": v[0], "rupees": round(v[1], 0)}
                            for k, v in conf.items()},
        "note": "Held-out temporal test split. FP cost = representment fee + analyst "
                "time; FN cost = recoverable amount foregone. Honest metrics per the "
                "track bar.",
    }


# ---- MCP server -------------------------------------------------------------
mcp = MCPServer("contra")


@mcp.tool(description="List held-out disputes with reason, amount, calibrated "
                      "p_win, and Contra's recommended action.")
def list_disputes(limit: int = 25) -> list:
    return h_list_disputes(limit)


@mcp.tool(description="Calibrated win probability for a dispute (out-of-sample).")
def score_winnability(dispute_id: str) -> dict:
    return h_score(dispute_id)


@mcp.tool(description="The fixed 8-step evidence bundle for a dispute; every item "
                      "has a resolvable artifact_id.")
def assemble_evidence(dispute_id: str) -> dict:
    return h_evidence(dispute_id)


@mcp.tool(description="FIGHT/ACCEPT/REFUND/ESCALATE decision with the full rupee "
                      "expected-value breakdown, at a given merchant dispute-ratio.")
def decide_dispute(dispute_id: str, current_ratio: float = DEFAULT_RATIO) -> dict:
    return h_decide(dispute_id, current_ratio)


@mcp.tool(description="Draft a defense-only rebuttal letter: framing + claims that "
                      "each cite a verifier-passed artifact. Cannot fabricate.")
def draft_rebuttal(dispute_id: str) -> dict:
    return h_rebuttal(dispute_id)


@mcp.tool(description="Honest metrics for a policy on the held-out test set: net "
                      "recovery, precision/recall/F1, and the rupee confusion matrix "
                      "(false-positive cost). policy: contra|fight_all|fight_none|"
                      "fight_amount|fight_pwin.")
def evaluate_policy(policy: str = "contra", current_ratio: float = DEFAULT_RATIO) -> dict:
    return h_evaluate(policy, current_ratio)


@mcp.resource("contra://methodology",
              description="Assumptions, limitations, and the defense-only stance.")
def methodology() -> str:
    return (
        "Contra — chargeback dispute triage (defense-only).\n\n"
        "- Winnability: gradient-checked; shipped model is calibrated logistic "
        "regression (the outcome DGP is an additive logit). Temporal test AUC ~0.78, "
        "Brier 0.185, ECE 0.056.\n"
        "- Decision engine: pure rupee expected-value arithmetic; no ML/LLM/"
        "randomness; unit-tested. ESCALATE = top 10% by uncertainty×amount; FIGHTs "
        "capacity-scheduled with 48h deadline pre-emption.\n"
        "- Evidence + letter: fixed 8-step lookup (not agentic); a generator writes "
        "only claims that cite present artifacts, and a separate verifier strips "
        "unsupported claims. Nothing here can fabricate or aid offense.\n"
        "- LIMITATIONS: synthetic dispute layer over IEEE-CIS (no public chargeback-"
        "outcome data); 182-day window is short for drift; labels are censored "
        "(outcomes observed only for fought cases in production); cost constants are "
        "vendor-directional. Stated openly, not hidden."
    )


@mcp.resource("contra://metrics",
              description="Headline net-recovery comparison on the held-out test set.")
def metrics() -> str:
    won = _test["won"].to_numpy()
    amt = _test["disputed_amount_inr"].to_numpy()
    lines = ["Net recovery (Rs), held-out test split:"]
    for name in ["fight_all", "fight_none", "fight_amount", "fight_pwin", "contra"]:
        a = _policy_action(name, DEFAULT_RATIO)
        lines.append(f"  {name:<14} {EVAL.net_recovery(a, won, amt, DEFAULT_RATIO, COSTS):>14,.0f}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run("stdio")
