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

import hashlib
import hmac
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                    # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler     # noqa: E402

import abuse_rings as AR              # noqa: E402
import decision_engine as DE          # noqa: E402
import evidence_agent as EA           # noqa: E402
import features as F                  # noqa: E402
import llm_client as LC               # noqa: E402
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
app = FastAPI(title="RokdaDaav — Chargeback Evidence")


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


# ---------------------------------------------------------------------------- #
# In-dashboard live agent: a Groq model DECIDES which RokdaDaav tools to call.
# These are the same tools the MCP server exposes — here they run in-process.
# ---------------------------------------------------------------------------- #
load_dotenv(ROOT / ".env")
_GROQ = None


def _groq():
    global _GROQ
    if _GROQ is None:
        from groq import Groq
        _GROQ = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _GROQ


def _chk(did):
    if did not in CURATED:
        raise ValueError(f"'{did}' is not in the demo set — call list_disputes first")


def _t_list(ratio, limit=12):
    dec = decided_at(ratio)
    return [{"dispute_id": d, "reason": _DISPUTE_BY_ID.loc[d]["reason_code"],
             "amount_inr": round(float(_DISPUTE_BY_ID.loc[d]["disputed_amount_inr"])),
             "p_win": round(float(_PRED_BY_ID.loc[d]["p_win"]), 2),
             "action": dec.loc[d]["action"]} for d in CURATED[: int(limit)]]


def _t_score(did):
    _chk(did)
    return {"dispute_id": did, "p_win": round(float(_PRED_BY_ID.loc[did]["p_win"]), 3)}


def _t_decide(did, ratio):
    _chk(did)
    d = decided_at(ratio).loc[did]
    return {"dispute_id": did, "action": d["action"],
            "ev_fight_inr": round(float(d["ev_fight"])),
            "ev_refund_inr": round(float(d["ev_refund"])),
            "chosen_ev_inr": round(float(d["chosen_ev"])),
            "cost_to_fight_inr": round(DE.cost_to_fight(COSTS))}


def _t_rebuttal(did, ratio):
    _chk(did)
    lp = build_case(did, ratio)["letter"]
    return {"dispute_id": did, "framing": lp["framing"], "claims_kept": lp["n_kept"],
            "claims": [x["claim"] for x in lp["claims"]]}


def _t_evaluate(ratio):
    return {"net_recovery_by_policy_inr": {k: round(v) for k, v in _baseline_table(ratio).items()}}


_TOOL_FN = {
    "list_disputes": lambda a, r: _t_list(r, a.get("limit", 12)),
    "score_winnability": lambda a, r: _t_score(a["dispute_id"]),
    "decide_dispute": lambda a, r: _t_decide(a["dispute_id"], a.get("current_ratio", r)),
    "draft_rebuttal": lambda a, r: _t_rebuttal(a["dispute_id"], r),
    "evaluate_policy": lambda a, r: _t_evaluate(a.get("current_ratio", r)),
}


def _tool(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


_TOOLS = [
    _tool("list_disputes", "List the demo disputes with reason, amount, p_win and "
          "RokdaDaav's recommended action.", {"limit": {"type": "integer"}}),
    _tool("score_winnability", "Calibrated win probability for one dispute.",
          {"dispute_id": {"type": "string"}}, ["dispute_id"]),
    _tool("decide_dispute", "FIGHT/ACCEPT/REFUND/ESCALATE with the rupee EV breakdown.",
          {"dispute_id": {"type": "string"}}, ["dispute_id"]),
    _tool("draft_rebuttal", "The drafted rebuttal letter (framing + kept claims).",
          {"dispute_id": {"type": "string"}}, ["dispute_id"]),
    _tool("evaluate_policy", "Net recovery (Rs) per policy on the held-out test set.", {}),
]

_AGENT_SYS = ("You are a merchant's chargeback risk assistant. You have no knowledge "
              "of specific disputes yourself — you MUST use the RokdaDaav tools to look "
              "things up and decide. Keep it short; end with a clear recommendation, "
              "in rupees where relevant.")


class _Ask(BaseModel):
    question: str
    ratio: float = RATIO


@app.post("/api/ask")
def ask(body: _Ask):
    msgs = [{"role": "system", "content": _AGENT_SYS},
            {"role": "user", "content": body.question}]
    steps = []
    for _ in range(6):
        resp = _groq().chat.completions.create(
            model="openai/gpt-oss-120b", messages=msgs, tools=_TOOLS,
            tool_choice="auto", temperature=0.2, max_tokens=1400, reasoning_effort="low")
        m = resp.choices[0].message
        if not m.tool_calls:
            return {"steps": steps, "answer": (m.content or "").strip()}
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [tc.model_dump() for tc in m.tool_calls]})
        for tc in m.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = _TOOL_FN[tc.function.name](args, body.ratio)
            except Exception as exc:                       # noqa: BLE001
                args = args if "args" in dir() else {}
                result = {"error": str(exc)}
            steps.append({"tool": tc.function.name, "args": args, "result": result})
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return {"steps": steps, "answer": "(the assistant did not converge)"}


# ---------------------------------------------------------------------------- #
# Razorpay-native: consume a dispute webhook, auto-triage, return a contest
# packet in Razorpay's format. SIMULATED — uses Razorpay's real payload/API
# shapes so it is drop-in, but events are generated locally (no live account).
# ---------------------------------------------------------------------------- #
RZP_WEBHOOK_SECRET = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")  # optional

# our reason_code -> a Razorpay-style dispute reason
_RZP_REASON = {
    "fraud": "fraudulent", "inr": "goods_or_services_not_received",
    "nad": "goods_or_services_not_as_described",
    "subscription": "recurring_transaction_cancelled",
    "agent_initiated": "unrecognized_transaction",
}


def _short_id(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:14]


def _razorpay_event(did: str) -> dict:
    """Build a Razorpay-shaped `dispute.created` webhook payload for one dispute."""
    row = _DISPUTE_BY_ID.loc[did]
    amt_paise = int(round(float(row["disputed_amount_inr"]) * 100))
    now = int(time.time())
    return {
        "entity": "event", "event": "dispute.created", "created_at": now,
        "payload": {"dispute": {"entity": {
            "id": "disp_" + _short_id(did),
            "payment_id": "pay_" + _short_id(did[::-1]),
            "amount": amt_paise, "currency": "INR", "amount_deducted": amt_paise,
            "reason_code": _RZP_REASON.get(row["reason_code"], "chargeback"),
            "respond_by": now + 7 * 86400, "status": "open", "phase": "chargeback",
            "created_at": now, "notes": {"rokdadaav_dispute_id": did},
        }}},
    }


def _verify_signature(raw_body: bytes, sig: str) -> bool:
    """Razorpay signs webhooks HMAC-SHA256. If no secret is configured we skip
    (documented) — with a secret set, this is the real verification."""
    if not RZP_WEBHOOK_SECRET:
        return True
    expected = hmac.new(RZP_WEBHOOK_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig or "")


def _map_evidence(evidence_steps) -> dict:
    """Map our artifacts to Razorpay contest-evidence categories."""
    buckets = {"access_activity_log": [], "shipping_proof": [],
               "customer_communication": [], "billing_proof": [], "other": []}
    where = {"AVS result": "access_activity_log", "CVV result": "access_activity_log",
             "3-D Secure": "access_activity_log", "IP origin": "access_activity_log",
             "Device match": "access_activity_log", "Delivery proof": "shipping_proof",
             "Support log": "customer_communication", "Consent record": "customer_communication",
             "Refund policy": "customer_communication", "Product photos": "shipping_proof",
             "Account tenure": "billing_proof", "Prior dispute record": "billing_proof"}
    for s in evidence_steps:
        for a in s["artifacts"]:
            buckets[where.get(a["label"], "other")].append(
                {"artifact_id": a["artifact_id"], "statement": a["statement"]})
    return {k: v for k, v in buckets.items() if v}


_ROUTING = {"FIGHT": "ready_to_submit", "ACCEPT": "auto_accepted",
            "REFUND": "auto_refunded", "ESCALATE": "human_review"}


def _triage(did: str, ratio: float) -> dict:
    case = build_case(did, ratio)
    dec = case["decision"]
    action = "ESCALATE" if dec["escalated"] else dec["action"]
    packet = None
    if action == "FIGHT":
        letter = case["letter"]
        packet = {                                   # Razorpay "contest dispute" API shape
            "action": "draft",                       # human approves before submit
            "amount": int(round(case["disputed_amount_inr"] * 100)),
            "summary": letter["framing"],
            "explanation_letter": [c["claim"] for c in letter["claims"]],
            "evidence": _map_evidence(case["evidence_steps"]),
        }
    return {
        "dispute_id": did, "amount_inr": case["disputed_amount_inr"],
        "reason": case["reason_code"], "p_win": round(case["p_win"], 2),
        "action": action, "routing": _ROUTING[action],
        "ev_fight_inr": round(float(dec["ev_fight"])),
        "contest_packet": packet,
    }


@app.post("/webhook/razorpay/dispute")
async def razorpay_webhook(request: Request, ratio: float = RATIO):
    raw = await request.body()
    if not _verify_signature(raw, request.headers.get("X-Razorpay-Signature", "")):
        raise HTTPException(400, "invalid webhook signature")
    evt = json.loads(raw or b"{}")
    ent = evt.get("payload", {}).get("dispute", {}).get("entity", {})
    did = ent.get("notes", {}).get("rokdadaav_dispute_id")
    if not did or did not in CURATED:
        raise HTTPException(404, "dispute not recognised in the demo set")
    return {"received": True, "razorpay_dispute_id": ent.get("id"),
            "respond_by": ent.get("respond_by"), "triage": _triage(did, ratio)}


@app.get("/api/sim/events")
def sim_events(n: int = 6):
    """A batch of Razorpay-shaped dispute events for the dashboard to replay
    through the webhook — the streaming 'incoming disputes' demo."""
    ids = random.sample(CURATED, min(int(n), len(CURATED)))
    return {"events": [_razorpay_event(d) for d in ids]}


# ---- REAL Razorpay (test-mode) connection ----------------------------------
_RZP_CLIENT = None


def _rzp_client():
    kid = os.environ.get("RAZORPAY_KEY_ID")
    ks = os.environ.get("RAZORPAY_KEY_SECRET")
    if not kid or not ks:
        return None
    global _RZP_CLIENT
    if _RZP_CLIENT is None:
        import razorpay
        _RZP_CLIENT = razorpay.Client(auth=(kid, ks))
    return _RZP_CLIENT


@app.get("/api/razorpay/status")
def razorpay_status():
    c = _rzp_client()
    if c is None:
        return {"connected": False, "mode": "simulated"}
    try:
        orders = c.order.all({"count": 1})
        kid = os.environ.get("RAZORPAY_KEY_ID", "")
        # never surface the full key — mask to prefix + last 4 (it is only the
        # public key id, but no credential belongs in the UI / a demo video)
        masked = (kid[:8] + "…" + kid[-4:]) if len(kid) > 12 else "test"
        return {"connected": True, "mode": "test", "key_id_masked": masked,
                "orders_count": orders.get("count", 0)}
    except Exception as exc:                                # noqa: BLE001
        return {"connected": False, "mode": "error", "error": str(exc)[:120]}


class _Submit(BaseModel):
    dispute_id: str
    ratio: float = RATIO


@app.post("/api/razorpay/submit")
def razorpay_submit(body: _Submit):
    """Record a triaged FIGHT to the merchant's REAL Razorpay account. Test mode
    has no dispute-contest-without-a-dispute, so we write a tagged order carrying
    RokdaDaav's decision — a real, dashboard-visible artifact of the integration."""
    tri = _triage(body.dispute_id, body.ratio)
    c = _rzp_client()
    if c is None:
        return {"submitted": False, "mode": "simulated", "triage": tri}
    try:
        o = c.order.create({
            "amount": int(round(tri["amount_inr"] * 100)), "currency": "INR",
            "receipt": f"rd_contest_{body.dispute_id}"[:40],
            "notes": {"source": "RokdaDaav", "dispute_id": body.dispute_id,
                      "decision": tri["action"], "p_win": str(tri["p_win"]),
                      "ev_fight_inr": str(tri["ev_fight_inr"])}})
        return {"submitted": True, "mode": "test",
                "razorpay_order_id": o["id"], "triage": tri}
    except Exception as exc:                                # noqa: BLE001
        return {"submitted": False, "mode": "error", "error": str(exc)[:160], "triage": tri}


# ---------------------------------------------------------------------------- #
# Abuse-ring detective (defense-only): cluster the dispute stream by shared
# signature, flag coordinated rings, and have the AI narrate each one.
# ---------------------------------------------------------------------------- #
_RINGS_DF = AR.build_dataset()
_RINGS, _RING_METRICS = AR.detect(_RINGS_DF)


def _narrate_ring(r: dict) -> str:
    prompt = (
        "A fraud-ops clusterer flagged a suspected chargeback abuse ring. Stats:\n"
        f"- members: {r['size']} disputes sharing one device fingerprint\n"
        f"- product: {r['product']}, average amount Rs {r['amount_avg_inr']}\n"
        f"- shipping city: {r['ship_city']}, card BIN {r['card_bin']}\n"
        f"- {int(r['datacenter_pct']*100)}% from datacenter/proxy IPs; all filed "
        f"within {r['filing_burst_days']} days\n\n"
        "In at most 2 sentences: assess whether this looks coordinated, and give one "
        "concrete DEFENSIVE action (e.g. hold payouts, manual review, block the BIN/"
        "device, tighten 3DS). Defense-only. Plain text.")
    resp = LC.chat("openai/gpt-oss-120b",
                   [{"role": "system", "content": "You are a payments fraud-ops analyst. "
                     "Defense-only. Be concise and specific."},
                    {"role": "user", "content": prompt}],
                   0.2, 400, reasoning_effort="low")
    return resp["content"].strip().replace("**", "")


@app.get("/api/rings")
def rings():
    out = []
    for r in _RINGS:
        rr = dict(r)
        rr["device_fp"] = r["device_fp"][:12] + "…"        # mask
        rr["assessment"] = _narrate_ring(r)
        out.append(rr)
    return {"metrics": _RING_METRICS, "dataset_size": int(len(_RINGS_DF)), "rings": out}


# ---------------------------------------------------------------------------- #
# What-if evidence advisor: re-score the model on counterfactual evidence to tell
# the merchant the cheapest documents to gather to make a case winnable.
# ---------------------------------------------------------------------------- #
_FB = F.load_features()
_XOHE = _FB.X_ohe
_OHE_COLS = list(_XOHE.columns)
_order = np.argsort(_FB.filed_dt.to_numpy(), kind="stable")
_cut = int(len(_order) * 0.70)
_SCALER = StandardScaler().fit(_XOHE.iloc[_order[:_cut]])
_SCORER = LogisticRegression(max_iter=2000).fit(
    _SCALER.transform(_XOHE.iloc[_order[:_cut]]), _FB.y.iloc[_order[:_cut]])
_ID_TO_ROW = {d: i for i, d in enumerate(
    pd.read_parquet(PROC / "disputes.parquet")["dispute_id"].tolist())}

# gatherable evidence the merchant can still produce (field, favourable value,
# label, how-to). Fixed facts (AVS/CVV/3DS/tenure/priors/device) are excluded.
_LEVERS = [
    ("delivery_proof_type", "signature", "Signed delivery proof",
     "obtain a signed delivery confirmation or carrier proof-of-delivery"),
    ("product_photos_on_file", 1, "Product photos",
     "attach photos of the shipped item and the product listing"),
    ("refund_policy_shown", 1, "Refund-policy acceptance",
     "attach the refund policy the customer accepted at checkout"),
    ("consent_record_exists", 1, "Consent / terms record",
     "attach the terms-and-consent acceptance record for the order"),
    ("customer_contacted_support", 1, "Support conversation",
     "attach the customer's support-conversation history"),
]


def _score_native(row: dict) -> float:
    oh = pd.get_dummies(pd.DataFrame([row]), columns=F.CATEGORICAL, dtype=int)
    oh = oh.reindex(columns=_OHE_COLS, fill_value=0)
    return float(_SCORER.predict_proba(_SCALER.transform(oh))[0, 1])


def _is_off(field, cur):
    return cur == "none" if field == "delivery_proof_type" else not bool(cur)


def _whatif(dispute_id: str, ratio: float) -> dict:
    i = _ID_TO_ROW[dispute_id]
    base = _FB.X.iloc[i].to_dict()
    base_p = _score_native(base)
    amt = float(_DISPUTE_BY_ID.loc[dispute_id, "disputed_amount_inr"])
    ctf = DE.cost_to_fight(COSTS)
    base_ev = base_p * amt - ctf
    base_action = str(decided_at(ratio).loc[dispute_id]["action"])

    levers, row_all = [], dict(base)
    for field, val, label, howto in _LEVERS:
        if not _is_off(field, base[field]):
            continue
        row2 = dict(base); row2[field] = val
        p2 = _score_native(row2)
        if p2 - base_p <= 0.003:                 # only recommend real gains
            continue
        row_all[field] = val
        ev2 = p2 * amt - ctf
        levers.append({"label": label, "howto": howto,
                       "lift": round(p2 - base_p, 3), "new_p_win": round(p2, 3),
                       "new_ev_inr": round(ev2), "flips_to_fight": bool(base_ev <= 0 < ev2)})
    levers.sort(key=lambda x: x["lift"], reverse=True)

    p_all = _score_native(row_all)
    combined = {"p_win": round(p_all, 3), "ev_inr": round(p_all * amt - ctf),
                "flips_to_fight": bool(base_ev <= 0 < p_all * amt - ctf)}
    return {"dispute_id": dispute_id, "base_p_win": round(base_p, 3),
            "base_action": base_action, "amount_inr": amt,
            "levers": levers, "combined": combined,
            "recommendation": _whatif_narration(base_p, base_action, amt, levers)}


def _whatif_narration(base_p, base_action, amt, levers) -> str:
    if not levers:
        return ("This case already carries strong evidence — no additional document "
                "would materially raise the win probability.")
    top = levers[0]
    lines = "; ".join(f"{l['label']} (+{int(l['lift']*100)} pts -> {int(l['new_p_win']*100)}%)"
                      for l in levers[:4])
    prompt = (
        f"A chargeback case is at {int(base_p*100)}% win probability (current decision: "
        f"{base_action}, amount Rs {int(amt)}). The merchant can still gather these "
        f"documents, each with the model's win-probability lift: {lines}. "
        f"In at most 2 sentences, tell the merchant the single highest-impact document "
        f"to gather ({top['label']}) and what it does to the decision / expected value. "
        f"Defense-only, concrete, plain text.")
    resp = LC.chat("openai/gpt-oss-120b",
                   [{"role": "system", "content": "You advise merchants on chargeback "
                     "evidence. Defense-only. Concise and actionable."},
                    {"role": "user", "content": prompt}], 0.2, 400, reasoning_effort="low")
    return resp["content"].strip().replace("**", "")


@app.get("/api/whatif/{dispute_id}")
def whatif(dispute_id: str, ratio: float = RATIO):
    if dispute_id not in _ID_TO_ROW:
        raise HTTPException(404, "unknown dispute")
    return _whatif(dispute_id, ratio)


@app.get("/")
def index():
    # no-store so edits to index.html always show on reload (demo convenience)
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


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
        "RokdaDaav": net(contra),
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
