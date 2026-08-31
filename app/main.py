"""
Phase 6 — dashboard backend (CLAUDE.md sections 13, 15). FastAPI + uvicorn.

Serves the single-page UI (app/static/) and a small JSON API. Everything the UI
shows is assembled from the already-built artifacts:
  * decision  -> decision_engine.decide() on the full test batch (pure, live)
  * evidence  -> evidence_agent.build_bundle() (pure, live)
  * letter    -> llm_generator/verifier, served from data/llm_cache (no live call)

A curated, stratified set of test-split disputes is exposed (variety of actions
and reason codes for the demo). Their letters are pre-warmed into the cache by
`python app/main.py --warm` so the server never needs a live LLM call.

Run:  python app/main.py --warm         # one-time, populates the LLM cache
      uvicorn app.main:app --reload     # or: python app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import decision_engine as DE          # noqa: E402
import evidence_agent as EA           # noqa: E402
import llm_generator as GEN           # noqa: E402
import llm_verifier as VER            # noqa: E402

PROC = ROOT / "data" / "processed"
STATIC = Path(__file__).resolve().parent / "static"

# ---- load data / config once ------------------------------------------------
_preds = pd.read_parquet(PROC / "predictions.parquet")
_disputes = pd.read_parquet(PROC / "disputes.parquet")
_evidence = pd.read_parquet(PROC / "evidence.parquet").set_index("dispute_id")
with open(ROOT / "config" / "generator.yaml", "r", encoding="utf-8") as _fh:
    _gcfg = yaml.safe_load(_fh)["evaluation"]
COSTS = DE.Costs.from_yaml()
RATIO = _gcfg["assumed_current_ratio"]
ESC_RATE = _gcfg["escalation_rate_target"]

# ---- decision on the full test batch (consistent with the Phase 5 harness) --
_df = _preds.merge(_disputes[["dispute_id", "deadline_dt"]], on="dispute_id")
_test = _df[_df["split"] == "test"].reset_index(drop=True)
_days = (_test["filed_dt"].max() - _test["filed_dt"].min()) / 86400
_HOURS_BUDGET = COSTS.analyst_hours_per_day * _days
_DISPUTE_BY_ID = _disputes.set_index("dispute_id")
_PRED_BY_ID = _preds.set_index("dispute_id")

# decisions depend on the merchant's current dispute-ratio (drives ratio_benefit
# / REFUND). Memoise the batch decision per ratio so the UI slider is instant.
_DECIDED_CACHE: dict = {}


def decided_at(ratio: float):
    r = round(float(ratio), 2)
    if r not in _DECIDED_CACHE:
        _DECIDED_CACHE[r] = DE.decide(
            _test, current_ratio=r, c=COSTS, escalation_rate=ESC_RATE,
            hours_budget=_HOURS_BUDGET).set_index("dispute_id")
    return _DECIDED_CACHE[r]


_DECIDED = decided_at(RATIO)      # default, used to build the stable curated queue


def _curated_ids(n_per_bucket=4) -> list:
    """Deterministic, stratified selection for the demo queue."""
    d = _DECIDED.copy()
    picks = []
    # a few of each final action, and some agent-initiated cases
    for action in [DE.FIGHT, DE.ACCEPT, DE.REFUND, DE.ESCALATE]:
        ids = d[d["action"] == action].sort_values("chosen_ev", ascending=False)
        picks += list(ids.head(n_per_bucket).index)
    agent = _test[_test["reason_code"] == "agent_initiated"]["dispute_id"]
    picks += list(agent.head(3))
    # de-dup preserving order
    seen, out = set(), []
    for p in picks:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


CURATED = _curated_ids()


def _letter_payload(did: str) -> dict:
    row = _DISPUTE_BY_ID.loc[did]
    bundle = EA.build_bundle(row.to_dict() | {"dispute_id": did}, _evidence.loc[did])
    gen = GEN.generate_letter(bundle)
    verified = VER.verify_claims(gen["claims"], bundle)
    packet = VER.build_packet(bundle, gen["framing"], verified)
    stripped = [{"claim": v["claim"], "artifact_id": v["artifact_id"],
                 "reason": v["reason"]}
                for v in verified if not (v["resolves"] and v["supported"])]
    return {"bundle": bundle, "gen": gen, "packet": packet, "stripped": stripped}


def _bundle_json(bundle: EA.Bundle) -> list:
    steps = []
    for s in bundle.steps:
        arts = [{"artifact_id": a,
                 "label": bundle.resolve(a).label,
                 "statement": bundle.resolve(a).statement,
                 "favourable": bool(bundle.resolve(a).favourable)}
                for a in s["artifact_ids"]]
        steps.append({"step": s["step"], "name": s["name"],
                      "latency_ms": s["latency_ms"], "ts_ms": s["ts_ms"],
                      "artifacts": arts})
    return steps


def build_case(did: str, ratio: float = RATIO) -> dict:
    row = _DISPUTE_BY_ID.loc[did]
    pred = _PRED_BY_ID.loc[did]
    dec = decided_at(ratio).loc[did]
    lp = _letter_payload(did)
    return {
        "dispute_id": did,
        "reason_code": row["reason_code"],
        "disputed_amount_inr": float(row["disputed_amount_inr"]),
        "days_txn_to_dispute": int(row["days_txn_to_dispute"]),
        "agent_initiated": bool(row["agent_initiated"]),
        "p_win": float(pred["p_win"]),
        "won": int(pred["won"]),
        "decision": {
            "action": dec["action"], "base_action": dec["base_action"],
            "escalated": bool(dec["escalated"]),
            "ev_fight": float(dec["ev_fight"]), "ev_accept": float(dec["ev_accept"]),
            "ev_refund": float(dec["ev_refund"]), "chosen_ev": float(dec["chosen_ev"]),
            "cost_to_fight": float(DE.cost_to_fight(COSTS)),
            "ratio_benefit": float(DE.ratio_benefit(ratio, COSTS)),
        },
        "evidence_steps": _bundle_json(lp["bundle"]),
        "letter": {
            "framing": lp["packet"]["framing"],
            "claims": lp["packet"]["claims"],
            "stripped": lp["stripped"],
            "n_generated": lp["packet"]["n_generated"],
            "n_kept": lp["packet"]["n_kept"],
        },
    }


# ---- warmed case store ------------------------------------------------------
_CASES: dict = {}


def warm():
    print(f"warming {len(CURATED)} cases (populates LLM cache) ...")
    for i, did in enumerate(CURATED, 1):
        _CASES[did] = build_case(did)
        c = _CASES[did]
        print(f"  [{i:>2}/{len(CURATED)}] {did}  {c['reason_code']:<16} "
              f"{c['decision']['action']:<9} kept {c['letter']['n_kept']}/"
              f"{c['letter']['n_generated']}")
    print("done.")


# ---- API --------------------------------------------------------------------
app = FastAPI(title="Contra — Chargeback Evidence")


@app.get("/api/warm_status")
def warm_status():
    return {"warmed": len(_CASES), "curated": len(CURATED)}


@app.get("/api/summary")
def summary(ratio: float = RATIO):
    return {
        "cost_to_fight": DE.cost_to_fight(COSTS),
        "assumed_ratio": RATIO, "ratio": round(float(ratio), 2),
        "vamp_threshold": COSTS.vamp_threshold_pct,
        "vamp_band": COSTS.vamp_warning_band_pct,
        "escalation_rate": ESC_RATE,
        "test_n": int(len(_test)), "curated_n": len(CURATED),
        "hours_budget": round(_HOURS_BUDGET),
        "baseline_table": _baseline_table(ratio),
    }


@app.get("/api/queue")
def queue(ratio: float = RATIO):
    dec = decided_at(ratio)
    out = []
    for did in CURATED:
        row = _DISPUTE_BY_ID.loc[did]
        out.append({"dispute_id": did, "reason_code": row["reason_code"],
                    "disputed_amount_inr": float(row["disputed_amount_inr"]),
                    "p_win": float(_PRED_BY_ID.loc[did]["p_win"]),
                    "action": dec.loc[did]["action"],
                    "agent_initiated": bool(row["agent_initiated"])})
    return out


@app.get("/api/dispute/{did}")
def dispute(did: str, ratio: float = RATIO):
    if did not in CURATED:
        raise HTTPException(404, "dispute not in curated demo set")
    return build_case(did, ratio)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _baseline_table(ratio: float = RATIO):
    """Recompute the Phase 5 headline for the summary bar (net recovery, rupees)."""
    import numpy as np
    won = _test["won"].to_numpy()
    amt = _test["disputed_amount_inr"].to_numpy()
    ctf = DE.cost_to_fight(COSTS)
    rb = DE.ratio_benefit(ratio, COSTS)

    def net(action):
        a = np.asarray(action)
        v = np.zeros(len(a))
        v[a == DE.FIGHT] = (won * amt - ctf)[a == DE.FIGHT]
        v[a == DE.REFUND] = (-amt + rb)[a == DE.REFUND]
        return float(v.sum())

    dec = decided_at(ratio)
    contra = dec.loc[_test["dispute_id"]]["action"].to_numpy().copy()
    base = dec.loc[_test["dispute_id"]]["base_action"].to_numpy()
    contra = np.where(contra == DE.ESCALATE, base, contra)
    thr = _gcfg["fight_if_amount_threshold_inr"]
    return {
        "fight everything": net(np.full(len(_test), DE.FIGHT)),
        "fight nothing": 0.0,
        f"fight if amount > Rs{thr}": net(np.where(amt > thr, DE.FIGHT, DE.ACCEPT)),
        "fight if p_win > 0.5": net(np.where(_test["p_win"].to_numpy() > 0.5, DE.FIGHT, DE.ACCEPT)),
        "Contra": net(contra),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if args.warm:
        warm()
    if args.serve or not args.warm:
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=args.port)
