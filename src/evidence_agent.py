"""
Phase 4 — evidence agent (CLAUDE.md section 10).

Deliberately NOT agentic: a FIXED 8-step sequence of lookups, executed in order,
each step logged with a simulated timestamp so the UI can animate it. Produces a
structured numbered bundle whose every item carries an artifact_id that resolves
to a source record. Pure and deterministic — no model chooses control flow here.

Presence rule: only artifacts that actually EXIST are emitted (no delivery proof
-> no delivery artifact; no support contact -> no support artifact; a missing
consent record is simply absent). Check-results that always run (AVS/CVV/3DS/
device/IP) are always present, carrying their true state — favourable or not.
The LLM generator later sees only this bundle, so it can never cite an absent
artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Per-step simulated latency (ms), only so the UI can animate step timing.
STEP_LATENCY_MS = {1: 120, 2: 260, 3: 180, 4: 220, 5: 300, 6: 160, 7: 140, 8: 90}

STEP_NAMES = {
    1: "Order record", 2: "Payment / 3DS check", 3: "Customer history",
    4: "Delivery record", 5: "Device & IP match", 6: "Past disputes on instrument",
    7: "Support conversations", 8: "Bundle assembly",
}


@dataclass
class Artifact:
    artifact_id: str
    step: int
    label: str
    statement: str          # the factual claim the verifier checks against
    favourable: bool        # whether it supports the merchant (for diagnostics only)


@dataclass
class Bundle:
    dispute_id: str
    reason_code: str
    steps: list = field(default_factory=list)      # [{step, name, latency_ms, ts_ms, artifact_ids}]
    artifacts: dict = field(default_factory=dict)  # artifact_id -> Artifact

    def resolve(self, artifact_id: str) -> Optional[Artifact]:
        return self.artifacts.get(artifact_id)

    def statements(self) -> dict:
        return {a.artifact_id: a.statement for a in self.artifacts.values()}


def _yn(b):
    return bool(b)


def build_bundle(dispute_row, evidence_row) -> Bundle:
    """Run the fixed 8-step sequence over one dispute's records."""
    d, e = dispute_row, evidence_row
    did = d["dispute_id"]
    b = Bundle(dispute_id=did, reason_code=d["reason_code"])
    counter = [0]

    def add(step, label, statement, favourable):
        counter[0] += 1
        aid = f"ART-{did}-{counter[0]:02d}"
        b.artifacts[aid] = Artifact(aid, step, label, statement, favourable)
        return aid

    ts = 0
    step_defs = []  # (step, [artifact_ids])

    # 1. Order record ------------------------------------------------------
    ids = [add(1, "Order record",
               f"Order of Rs {d['disputed_amount_inr']:.0f} under reason code "
               f"'{d['reason_code']}'; dispute filed {int(d['days_txn_to_dispute'])} "
               f"days after the transaction.", favourable=True)]
    step_defs.append((1, ids))

    # 2. Payment / 3DS check ----------------------------------------------
    ids = []
    ids.append(add(2, "AVS result",
                   f"Address Verification (AVS): {'match' if _yn(e['avs_match']) else 'no match'}.",
                   favourable=_yn(e["avs_match"])))
    ids.append(add(2, "CVV result",
                   f"Card security code (CVV): {'match' if _yn(e['cvv_match']) else 'no match'}.",
                   favourable=_yn(e["cvv_match"])))
    ids.append(add(2, "3-D Secure",
                   f"3-D Secure authentication: {e['three_ds_status']}.",
                   favourable=(e["three_ds_status"] == "passed")))
    step_defs.append((2, ids))

    # 3. Customer history --------------------------------------------------
    ids = [add(3, "Account tenure",
               f"Cardholder account age {int(e['account_age_days'])} days; "
               f"{int(e['prior_txn_count'])} prior transactions on file.",
               favourable=e["account_age_days"] >= 180)]
    step_defs.append((3, ids))

    # 4. Delivery record (only if proof exists) ---------------------------
    ids = []
    if e["delivery_proof_type"] != "none":
        otp = " (OTP verified)" if _yn(e["delivery_otp_verified"]) else ""
        ids.append(add(4, "Delivery proof",
                       f"Delivery proof on file: {e['delivery_proof_type']}{otp}.",
                       favourable=True))
    step_defs.append((4, ids))

    # 5. Device & IP match -------------------------------------------------
    ids = []
    ids.append(add(5, "Device match",
                   f"Device fingerprint vs. customer history: {e['device_match_status']}.",
                   favourable=(e["device_match_status"] == "matched")))
    ids.append(add(5, "IP origin",
                   f"Originating IP is {'a datacenter/hosting provider' if _yn(e['ip_is_datacenter']) else 'residential'}; "
                   f"{e['ip_to_ship_km']:.0f} km from the shipping address.",
                   favourable=not _yn(e["ip_is_datacenter"])))
    step_defs.append((5, ids))

    # 6. Past disputes on instrument --------------------------------------
    ids = [add(6, "Prior dispute record",
               f"{int(e['prior_undisputed_count'])} of {int(e['prior_txn_count'])} "
               f"prior transactions on this instrument were undisputed.",
               favourable=e["prior_undisputed_count"] >= 3)]
    step_defs.append((6, ids))

    # 7. Support conversations (only if contacted) ------------------------
    ids = []
    if _yn(e["customer_contacted_support"]):
        ids.append(add(7, "Support log",
                       f"Customer support contact on record: {int(e['support_ticket_count'])} ticket(s).",
                       favourable=True))
    step_defs.append((7, ids))

    # 8. Bundle assembly — supplementary records (only present ones) ------
    ids = []
    if _yn(e["consent_record_exists"]):
        ids.append(add(8, "Consent record",
                       "Consent / terms-acceptance record on file for this order.", True))
    if _yn(e["product_photos_on_file"]):
        ids.append(add(8, "Product photos",
                       "Product photographs on file for the shipped item.", True))
    if _yn(e["refund_policy_shown"]):
        ids.append(add(8, "Refund policy",
                       "Refund policy was shown to the customer at checkout.", True))
    if _yn(e["agent_mandate_on_file"]):
        ids.append(add(8, "Agent mandate",
                       "Authorised autonomous-purchase mandate on file for this order.", True))
    step_defs.append((8, ids))

    # assemble steps with simulated timestamps
    for step, aids in step_defs:
        lat = STEP_LATENCY_MS[step]
        ts += lat
        b.steps.append({"step": step, "name": STEP_NAMES[step],
                        "latency_ms": lat, "ts_ms": ts, "artifact_ids": aids})
    return b


def bundle_for_generator(b: Bundle) -> dict:
    """The exact payload the generator is allowed to see: reason_code + the
    numbered artifacts (id, label, statement). No p_win, no decision, no absent
    artifacts, no favourability flags."""
    return {
        "reason_code": b.reason_code,
        "artifacts": [
            {"artifact_id": a.artifact_id, "label": a.label, "statement": a.statement}
            for a in b.artifacts.values()
        ],
    }
