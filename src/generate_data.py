"""
Phase 1 — synthetic dispute generator (CLAUDE.md sections 5-7).

Builds a dispute layer over the IEEE-CIS transaction data. Driven ENTIRELY by
config/generator.yaml and config/costs.yaml — no magic numbers here. Outputs
four parquet files to data/processed/: disputes, evidence, outcomes, dataset.

Pipeline is reproducible from a single seed. Fails loudly, never silently
(CLAUDE.md section 16).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# CLAUDE.md section 4 columns actually used (R_emaildomain dropped: 76.75% null).
TXN_COLS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "isFraud",
]
IDN_USE = ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"]
IDN_COLS = ["TransactionID"] + IDN_USE

SECONDS_PER_DAY = 86400


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def draw_bernoulli(rng, is_fraud, p_map):
    """Per-row Bernoulli, prob chosen by isFraud."""
    p = np.where(is_fraud, p_map["fraud"], p_map["genuine"])
    return rng.random(len(is_fraud)) < p


def draw_categorical(rng, is_fraud, genuine_dist, fraud_dist):
    """Per-row categorical over the same keys; distribution chosen by isFraud."""
    cats = list(genuine_dist.keys())
    pg = np.array([genuine_dist[c] for c in cats], dtype=float)
    pf = np.array([fraud_dist[c] for c in cats], dtype=float)
    P = np.where(is_fraud[:, None], pf[None, :], pg[None, :])
    P = P / P.sum(axis=1, keepdims=True)
    u = rng.random(len(is_fraud))
    cum = np.cumsum(P, axis=1)
    idx = (u[:, None] > cum).sum(axis=1)
    idx = np.clip(idx, 0, len(cats) - 1)
    return np.asarray(cats, dtype=object)[idx]


def draw_gamma(rng, is_fraud, g_params, f_params):
    k = np.where(is_fraud, f_params["shape_k"], g_params["shape_k"])
    theta = np.where(is_fraud, f_params["scale_theta"], g_params["scale_theta"])
    return rng.gamma(k, theta)


def solve_intercept(spread, target, lo=-12.0, hi=12.0, iters=64):
    """Bisection: find intercept b so mean(sigmoid(b + spread)) == target.
    Corrects the Jensen gap — with a wide logit spread, mean(sigmoid) drifts
    toward 0.5, so a naive intercept of logit(target) misses the target mean."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if sigmoid(mid + spread).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def load_subset(cfg, costs):
    """Load the §4 columns, take a contiguous TransactionDT subset, add INR + id flag."""
    txn = pd.read_csv(RAW / "train_transaction.csv", usecols=TXN_COLS)
    idn = pd.read_csv(RAW / "train_identity.csv", usecols=IDN_COLS)

    txn = txn.sort_values("TransactionDT").reset_index(drop=True)
    n = cfg["sampling"]["subset_size"]
    mode = cfg["sampling"]["mode"]
    if n > len(txn):
        raise ValueError(f"subset_size {n} exceeds available rows {len(txn)}")

    # data_window_end = true 182-day boundary over ALL transactions; the
    # censoring test in section 12 is against this, not the subset's own end.
    data_window_end = int(txn["TransactionDT"].max())

    if mode == "head":
        sub = txn.iloc[:n].copy()
    elif mode == "span":
        stride = len(txn) // n            # evenly-strided, covers full window
        sub = txn.iloc[::stride].iloc[:n].copy()
    else:
        raise ValueError(f"unknown sampling.mode: {mode!r}")

    sub["has_identity"] = sub["TransactionID"].isin(set(idn["TransactionID"]))
    sub = sub.merge(idn, on="TransactionID", how="left")
    sub["amount_inr"] = sub["TransactionAmt"] * costs["usd_to_inr"]
    return sub.reset_index(drop=True), data_window_end


def select_disputes(rng, sub, cfg):
    """Weighted sampling without replacement: over-represent fraud / high-value / no-identity."""
    ds = cfg["dispute_selection"]
    n_disputes = int(round(len(sub) * ds["dispute_rate_target"]))

    amt_pct = sub["amount_inr"].rank(pct=True).to_numpy()
    w = np.ones(len(sub))
    w *= np.where(sub["isFraud"].to_numpy() == 1, ds["weight_fraud"], 1.0)
    w *= np.where(sub["has_identity"].to_numpy(), 1.0, ds["weight_no_identity"])
    w *= 1.0 + ds["amount_tilt"] * amt_pct           # 1 .. 1+tilt with amount
    p = w / w.sum()

    idx = rng.choice(len(sub), size=n_disputes, replace=False, p=p)
    d = sub.iloc[np.sort(idx)].copy().reset_index(drop=True)
    return d


def apply_lag_and_censor(rng, d, cfg, data_window_end):
    """Right-skewed filing lag; drop disputes filed beyond the 182-day data window."""
    lg = cfg["lag_distribution"]
    days = rng.gamma(lg["shape_k"], lg["scale_theta"], size=len(d))
    days = np.clip(np.round(days), lg["min_days"], None).astype(int)
    d["days_txn_to_dispute"] = days
    d["filed_dt"] = d["TransactionDT"] + days * SECONDS_PER_DAY

    # A dispute filed after the 182-day data window is right-censored: we never
    # observe its outcome. DROP, not clip (clipping piles mass at the boundary
    # and distorts the lag tail). Boundary is the true data max (section 12).
    keep = d["filed_dt"] <= data_window_end
    n_dropped = int((~keep).sum())
    d = d[keep].copy().reset_index(drop=True)
    return d, n_dropped, data_window_end


def assign_reason_codes(rng, d, cfg):
    """Carve agent-initiated first (legit + identity), then conditional draw for the rest."""
    rc = cfg["reason_code_mix"]
    agent_sig = cfg["agent_initiated"]
    n = len(d)
    is_fraud = d["isFraud"].to_numpy() == 1

    reason = np.empty(n, dtype=object)
    is_agent = np.zeros(n, dtype=bool)

    # --- carve agent-initiated: genuinely legit AND has an identity record ---
    target_agent = int(round(n * rc["target_share"]["agent_initiated"]))
    eligible = np.where((~is_fraud) & d["has_identity"].to_numpy())[0]
    if len(eligible) < target_agent:
        raise ValueError(
            f"only {len(eligible)} legit+identity disputes available, "
            f"need {target_agent} for agent_initiated — loosen weight_no_identity"
        )
    agent_idx = rng.choice(eligible, size=target_agent, replace=False)
    is_agent[agent_idx] = True
    reason[agent_idx] = "agent_initiated"

    # --- remaining disputes: conditional categorical over the 4 non-agent codes
    rest = np.where(~is_agent)[0]
    codes4 = ["fraud", "inr", "nad", "subscription"]
    pf = np.array([rc["conditional_when_fraud"][c] for c in codes4])
    pg = np.array([rc["conditional_when_legit"][c] for c in codes4])
    for i in rest:
        probs = pf if is_fraud[i] else pg
        reason[i] = codes4[rng.choice(len(codes4), p=probs / probs.sum())]

    d["reason_code"] = reason
    d["agent_initiated"] = is_agent
    return d


def generate_evidence(rng, d, cfg):
    """One row per dispute. Presence conditioned on isFraud so evidence carries signal."""
    ev = cfg["evidence"]
    is_fraud = d["isFraud"].to_numpy() == 1
    has_id = d["has_identity"].to_numpy()
    is_agent = d["agent_initiated"].to_numpy()
    n = len(d)

    e = pd.DataFrame({"dispute_id": d["dispute_id"].to_numpy()})

    e["avs_match"] = draw_bernoulli(rng, is_fraud, ev["avs_match_p"])
    e["cvv_match"] = draw_bernoulli(rng, is_fraud, ev["cvv_match_p"])
    e["three_ds_status"] = draw_categorical(
        rng, is_fraud, ev["three_ds_dist"]["genuine"], ev["three_ds_dist"]["fraud"])

    # device_match_status: 'unknown' if no identity record; else matched/mismatched.
    dm = draw_categorical(
        rng, is_fraud, ev["device_match_when_identity"]["genuine"],
        ev["device_match_when_identity"]["fraud"])
    dm = np.where(has_id, dm, "unknown").astype(object)
    e["device_match_status"] = dm

    prior = rng.poisson(
        np.where(is_fraud, ev["prior_txn_count"]["fraud_lam"],
                 ev["prior_txn_count"]["genuine_lam"]))
    e["prior_txn_count"] = prior
    frac = np.where(is_fraud, ev["prior_undisputed_frac"]["fraud"],
                    ev["prior_undisputed_frac"]["genuine"])
    e["prior_undisputed_count"] = rng.binomial(prior, frac)

    e["account_age_days"] = draw_gamma(
        rng, is_fraud, ev["account_age_days"]["genuine"],
        ev["account_age_days"]["fraud"]).round().astype(int)

    proof = draw_categorical(
        rng, is_fraud, ev["delivery_proof_dist"]["genuine"],
        ev["delivery_proof_dist"]["fraud"])
    e["delivery_proof_type"] = proof
    otp_p = np.where(is_fraud, ev["delivery_otp_verified_p"]["fraud"],
                     ev["delivery_otp_verified_p"]["genuine"])
    e["delivery_otp_verified"] = (proof == "otp") & (rng.random(n) < otp_p)

    e["ip_to_ship_km"] = draw_gamma(
        rng, is_fraud, ev["ip_to_ship_km"]["genuine"],
        ev["ip_to_ship_km"]["fraud"]).round(1)
    e["ip_is_datacenter"] = draw_bernoulli(rng, is_fraud, ev["ip_is_datacenter_p"])

    contacted = draw_bernoulli(rng, is_fraud, ev["customer_contacted_support_p"])
    e["customer_contacted_support"] = contacted
    tickets = rng.poisson(
        np.where(is_fraud, ev["support_ticket_count"]["fraud_lam"],
                 ev["support_ticket_count"]["genuine_lam"]))
    e["support_ticket_count"] = np.where(contacted, np.maximum(tickets, 1), 0)

    e["product_photos_on_file"] = draw_bernoulli(rng, is_fraud, ev["product_photos_on_file_p"])
    e["refund_policy_shown"] = draw_bernoulli(rng, is_fraud, ev["refund_policy_shown_p"])
    e["consent_record_exists"] = draw_bernoulli(rng, is_fraud, ev["consent_record_exists_p"])
    e["agent_mandate_on_file"] = draw_bernoulli(rng, is_fraud, ev["agent_mandate_on_file_p"])

    # --- agent-initiated signature override (CLAUDE.md section 7) ---
    sig = cfg["agent_initiated"]
    e.loc[is_agent, "device_match_status"] = sig["device_match_status"]
    e.loc[is_agent, "ip_is_datacenter"] = sig["ip_is_datacenter"]
    e.loc[is_agent, "customer_contacted_support"] = sig["customer_contacted_support"]
    e.loc[is_agent, "support_ticket_count"] = 0   # no support contact -> no tickets
    e.loc[is_agent, "three_ds_status"] = sig["three_ds_status"]
    e.loc[is_agent, "agent_mandate_on_file"] = sig["agent_mandate_on_file"]

    return e


def generate_outcomes(rng, d, e, sub_card1, cfg):
    """Additive logit + unobserved issuer_leniency + Gaussian noise -> Bernoulli. Section 6."""
    oc = cfg["outcome"]
    w = oc["weights"]
    caps = oc["caps"]
    n = len(d)
    reason = d["reason_code"].to_numpy()

    # Evidence contribution is accumulated separately, then CENTERED within each
    # reason code before adding base_logit. This makes base_logit[r] the mean
    # logit for reason r (so realised base win rates match section 7 targets),
    # while evidence still drives within-reason spread. Without centering the
    # mostly-positive evidence on an 86%-legit dispute pool inflates every win
    # rate toward 1.
    ev = np.zeros(len(d), dtype=float)

    dm = e["device_match_status"].to_numpy()
    proof = e["delivery_proof_type"].to_numpy()
    prior_undisp = np.minimum(e["prior_undisputed_count"].to_numpy(), caps["prior_undisputed_count"])
    age = np.minimum(e["account_age_days"].to_numpy(), caps["account_age_days"])

    ev += w["avs_match"] * e["avs_match"].to_numpy()
    ev += w["cvv_match"] * e["cvv_match"].to_numpy()
    ev += w["three_ds_passed"] * (e["three_ds_status"].to_numpy() == "passed")
    ev += w["device_matched"] * (dm == "matched")
    ev += w["device_mismatched"] * (dm == "mismatched")
    ev += w["prior_undisputed_count"] * prior_undisp
    ev += w["account_age_days"] * age
    ev += w["delivery_proof_strong"] * np.isin(proof, ["signature", "otp", "photo"])
    ev += w["delivery_proof_tracking"] * (proof == "tracking")
    ev += w["delivery_otp_verified"] * e["delivery_otp_verified"].to_numpy()
    ev += w["ip_is_datacenter"] * e["ip_is_datacenter"].to_numpy()
    ev += w["customer_contacted_support"] * e["customer_contacted_support"].to_numpy()
    ev += w["product_photos_on_file"] * e["product_photos_on_file"].to_numpy()
    ev += w["refund_policy_shown"] * e["refund_policy_shown"].to_numpy()
    ev += w["consent_record_exists"] * e["consent_record_exists"].to_numpy()
    ev += w["agent_mandate_on_file"] * e["agent_mandate_on_file"].to_numpy()

    ev *= oc["weight_scale"]          # single knob to tune achievable AUC

    # center evidence within each reason code (see note above)
    for r in np.unique(reason):
        m = reason == r
        ev[m] -= ev[m].mean()

    # unobserved issuer leniency, keyed off a card1-derived group. NOT a feature.
    il = oc["issuer_leniency"]
    group = (sub_card1.to_numpy().astype(np.int64)) % il["n_groups"]
    group_effect = rng.normal(0.0, il["sigma"], size=il["n_groups"])

    # spread = everything except the per-reason intercept
    spread = ev + group_effect[group] + rng.normal(0.0, oc["noise_sigma"], size=n)

    # per-reason intercept calibrated so realised mean win rate == section 7
    # target (base_logit is the target expressed as a logit; Jensen-corrected).
    logit = np.empty(n, dtype=float)
    for r in np.unique(reason):
        m = reason == r
        target = sigmoid(oc["base_logit"][r])
        logit[m] = solve_intercept(spread[m], target) + spread[m]

    p_win = sigmoid(logit)
    won = rng.binomial(1, p_win)

    out = pd.DataFrame({
        "dispute_id": d["dispute_id"].to_numpy(),
        "p_win_true": p_win,
        "won": won.astype(int),
        "issuer_group": group,          # kept for diagnostics only, NOT a feature
    })
    return out


def add_fought(rng, d, cfg):
    """Naive historical merchant: fight above an amount threshold, with occasional flips."""
    fp = cfg["fought_policy"]
    naive = d["disputed_amount_inr"].to_numpy() > fp["amount_threshold_inr"]
    flip = rng.random(len(d)) < fp["flip_prob"]
    return (naive ^ flip)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    cfg = load_yaml(ROOT / "config" / "generator.yaml")
    costs = load_yaml(ROOT / "config" / "costs.yaml")
    rng = np.random.default_rng(cfg["seed"])
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading subset ...")
    sub, data_window_end = load_subset(cfg, costs)
    print(f"      subset: {len(sub):,} txns, "
          f"isFraud {sub['isFraud'].mean()*100:.2f}%, "
          f"identity {sub['has_identity'].mean()*100:.2f}%")

    print("[2/6] selecting disputes ...")
    d = select_disputes(rng, sub, cfg)
    n_selected = len(d)
    print(f"      selected {len(d):,} disputes "
          f"(fraud {d['isFraud'].mean()*100:.1f}%, "
          f"identity {d['has_identity'].mean()*100:.1f}%)")

    print("[3/6] filing lag + right-censor drop ...")
    d, n_dropped, obs_end = apply_lag_and_censor(rng, d, cfg, data_window_end)
    print(f"      dropped {n_dropped:,} right-censored; "
          f"{len(d):,} remain (data_window_end={obs_end:,}s = {obs_end/SECONDS_PER_DAY:.0f}d)")

    print("[4/6] assigning reason codes ...")
    d = assign_reason_codes(rng, d, cfg)

    # identifiers + amounts now that the dispute set is final
    d["dispute_id"] = [f"DSP{ i:06d}" for i in range(len(d))]
    d["disputed_amount_inr"] = d["amount_inr"].round(2)
    # deadline_dt: representment window after filing. Placeholder span (mean lag,
    # in days) until Phase 3 sets the real network deadline; documented as such.
    deadline_days = int(cfg["lag_distribution"]["scale_theta"] * cfg["lag_distribution"]["shape_k"])
    d["deadline_dt"] = d["filed_dt"] + deadline_days * SECONDS_PER_DAY

    print("[5/6] generating evidence ...")
    e = generate_evidence(rng, d, cfg)

    print("[6/6] generating outcomes + fought ...")
    out = generate_outcomes(rng, d, e, d["card1"], cfg)
    d["fought"] = add_fought(rng, d, cfg)

    # ---- assemble the four tables ----
    disputes = d[[
        "dispute_id", "TransactionID", "filed_dt", "days_txn_to_dispute",
        "reason_code", "disputed_amount_inr", "deadline_dt", "agent_initiated",
        "fought",
    ]].copy()

    dataset = (
        disputes
        .merge(e, on="dispute_id")
        .merge(out[["dispute_id", "p_win_true", "won", "issuer_group"]], on="dispute_id")
        .merge(d[["dispute_id", "TransactionDT", "isFraud", "has_identity",
                  "ProductCD", "card4", "card6"]], on="dispute_id")
    )

    disputes.to_parquet(OUT / "disputes.parquet", index=False)
    e.to_parquet(OUT / "evidence.parquet", index=False)
    out[["dispute_id", "p_win_true", "won"]].to_parquet(OUT / "outcomes.parquet", index=False)
    dataset.to_parquet(OUT / "dataset.parquet", index=False)

    # meta for the verifier: things not recoverable from the processed tables
    import json
    meta = {
        "seed": cfg["seed"],
        "subset_size": int(len(sub)),
        "n_selected": int(n_selected),
        "n_dropped_right_censored": int(n_dropped),
        "n_disputes": int(len(disputes)),
        "data_window_end": int(obs_end),
        "train_fraction": cfg["evaluation"]["train_fraction"],
        "target_share": cfg["reason_code_mix"]["target_share"],
        "base_win_rate": {  # section 7 targets (sigmoid of base_logit)
            r: float(sigmoid(v)) for r, v in cfg["outcome"]["base_logit"].items()
        },
        "leakage_auc_band": cfg["evaluation"]["leakage_auc_band"],
        "leakage_auc_fail_above": cfg["evaluation"]["leakage_auc_fail_above"],
    }
    with open(OUT / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nwrote {len(disputes):,} disputes to {OUT}")
    print(f"  realised win rate: {out['won'].mean()*100:.1f}%")
    print(f"  reason mix: " + ", ".join(
        f"{k} {v*100:.0f}%" for k, v in disputes['reason_code'].value_counts(normalize=True).items()))
    print(f"[done in {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
