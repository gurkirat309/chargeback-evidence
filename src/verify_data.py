"""
Phase 1 — dispute-data verifier (CLAUDE.md sections 6, 7, 12).

Exists to catch us fooling ourselves. Reads data/processed/ and prints the eight
checks from the Phase 1 prompt. The headline is the leakage check: a plain
logistic regression on the evidence features must land test AUC in 0.72-0.80.
Above 0.85 means the generator is leaking -> regenerate with looser weights.

Read-only. Never regenerates or edits data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
END = "\033[0m"


def hr(title):
    print("\n" + "=" * 72)
    print(BOLD + title + END)
    print("=" * 72)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric evidence features ONLY. No p_win_true, isFraud, issuer_group, won."""
    proof = df["delivery_proof_type"]
    tds = df["three_ds_status"]
    dm = df["device_match_status"]
    X = pd.DataFrame({
        "avs_match": df["avs_match"].astype(int),
        "cvv_match": df["cvv_match"].astype(int),
        "three_ds_passed": (tds == "passed").astype(int),
        "three_ds_attempted": (tds == "attempted").astype(int),
        "device_matched": (dm == "matched").astype(int),
        "device_mismatched": (dm == "mismatched").astype(int),
        "prior_txn_count": df["prior_txn_count"],
        "prior_undisputed_count": df["prior_undisputed_count"],
        "account_age_days": df["account_age_days"],
        "delivery_tracking": (proof == "tracking").astype(int),
        "delivery_signature": (proof == "signature").astype(int),
        "delivery_otp": (proof == "otp").astype(int),
        "delivery_photo": (proof == "photo").astype(int),
        "delivery_otp_verified": df["delivery_otp_verified"].astype(int),
        "ip_to_ship_km": df["ip_to_ship_km"],
        "ip_is_datacenter": df["ip_is_datacenter"].astype(int),
        "customer_contacted_support": df["customer_contacted_support"].astype(int),
        "support_ticket_count": df["support_ticket_count"],
        "product_photos_on_file": df["product_photos_on_file"].astype(int),
        "refund_policy_shown": df["refund_policy_shown"].astype(int),
        "consent_record_exists": df["consent_record_exists"].astype(int),
        "agent_mandate_on_file": df["agent_mandate_on_file"].astype(int),
    })
    return X


def main():
    df = pd.read_parquet(PROC / "dataset.parquet")
    meta = json.loads((PROC / "meta.json").read_text())

    # ---------------------------------------------------------------- 1
    hr("1. DISPUTE COUNT & RATE")
    print(f"disputes (after right-censor drop): {len(df):,}")
    print(f"transaction subset               : {meta['subset_size']:,}")
    print(f"selected before censor drop      : {meta['n_selected']:,}")
    print(f"dispute rate vs subset           : {len(df)/meta['subset_size']*100:.3f}%  "
          f"(target 1.00%)")

    # ---------------------------------------------------------------- 2
    hr("2. REASON CODE DISTRIBUTION  (generated vs target; observed drifts by censoring)")
    observed = df["reason_code"].value_counts(normalize=True)
    gen = meta.get("generated_reason_share", {})
    print(f"{'code':<16}{'target':>9}{'generated':>11}{'observed':>10}{'obs-tgt':>9}")
    for code, tgt in meta["target_share"].items():
        g = gen.get(code, 0.0)
        o = observed.get(code, 0.0)
        print(f"{code:<16}{tgt*100:>8.1f}%{g*100:>10.1f}%{o*100:>9.1f}%{(o-tgt)*100:>+9.1f}")
    print("\n  'generated' = pre-censor mix (should match target). 'observed' drifts")
    print("  toward fast-filing reasons (fraud/agent) because slow-filing reasons")
    print("  (inr/subscription) are censored more often — realistic, by design.")

    # ---------------------------------------------------------------- 3
    hr("3. REALISED WIN RATE PER REASON vs BASE RATE")
    print(f"{'code':<16}{'won':>8}{'n':>7}{'target':>10}{'delta':>10}")
    for code, tgt in meta["base_win_rate"].items():
        sub = df[df["reason_code"] == code]
        wr = sub["won"].mean()
        print(f"{code:<16}{wr*100:>7.1f}%{len(sub):>7}{tgt*100:>9.1f}%{(wr-tgt)*100:>+9.1f}")
    print(f"{'OVERALL':<16}{df['won'].mean()*100:>7.1f}%{len(df):>7}")

    # ---------------------------------------------------------------- 4
    hr("4. EVIDENCE FEATURE CORRELATION WITH `won`  (sorted; >0.5 flagged)")
    X = build_feature_matrix(df)
    corr = X.apply(lambda c: np.corrcoef(c, df["won"])[0, 1]).sort_values(
        key=lambda s: s.abs(), ascending=False)
    for feat, c in corr.items():
        line = f"  {feat:<28}{c:>+7.3f}"
        if abs(c) > 0.5:
            line = RED + line + f"   <-- LEAK (|r|>0.5)" + END
        print(line)
    print(f"\nmax |correlation| = {corr.abs().max():.3f}")

    # ---------------------------------------------------------------- 4b
    hr("4b. consent_record_exists BY reason_code  (proxy check)")
    tab = df.groupby("reason_code")["consent_record_exists"].agg(["mean", "sum", "count"])
    print(f"{'reason_code':<16}{'present %':>10}{'n_present':>11}{'n':>7}")
    for code in meta["base_win_rate"]:
        if code in tab.index:
            r = tab.loc[code]
            print(f"{code:<16}{r['mean']*100:>9.1f}%{int(r['sum']):>11}{int(r['count']):>7}")
    lo, hi = tab["mean"].min(), tab["mean"].max()
    if lo > 0.2:
        print(f"\n  {GREEN}present at {lo*100:.0f}-{hi*100:.0f}% across ALL reasons -> "
              f"NOT a reason-code proxy.{END} Its correlation reflects outcome weight.")
    else:
        print(f"\n  {YELLOW}near-zero for some reasons -> partly a reason proxy; "
              f"see README segment-model note.{END}")

    # ---------------------------------------------------------------- 5
    hr("5. p_win_true — MEAN PER REASON + SPREAD (want spread, not a spike)")
    print(f"{'code':<16}{'mean':>8}{'std':>8}{'p10':>8}{'p50':>8}{'p90':>8}")
    for code in meta["base_win_rate"]:
        s = df.loc[df["reason_code"] == code, "p_win_true"]
        print(f"{code:<16}{s.mean():>8.3f}{s.std():>8.3f}"
              f"{s.quantile(.1):>8.3f}{s.quantile(.5):>8.3f}{s.quantile(.9):>8.3f}")
    allp = df["p_win_true"]
    print(f"{'ALL':<16}{allp.mean():>8.3f}{allp.std():>8.3f}"
          f"{allp.quantile(.1):>8.3f}{allp.quantile(.5):>8.3f}{allp.quantile(.9):>8.3f}")
    # crude histogram to confirm it is not a spike
    hist, edges = np.histogram(allp, bins=10, range=(0, 1))
    print("\n  p_win_true histogram:")
    for i in range(10):
        bar = "#" * int(hist[i] / max(hist) * 40)
        print(f"   [{edges[i]:.1f},{edges[i+1]:.1f})  {hist[i]:>5}  {bar}")

    # ---------------------------------------------------------------- 5b
    hr("5b. FILING LAG (days_txn_to_dispute) — distribution + by reason")
    SPD = 86400
    lag = df["days_txn_to_dispute"]
    print(f"OVERALL   mean {lag.mean():>5.1f}  median {lag.median():>4.0f}  "
          f"p90 {lag.quantile(.9):>4.0f}  p99 {lag.quantile(.99):>4.0f}  max {lag.max():>4}")
    print(f"{'by reason':<16}{'mean':>7}{'median':>8}{'p90':>7}{'p99':>7}{'max':>7}")
    for code in meta["base_win_rate"]:
        s = df.loc[df["reason_code"] == code, "days_txn_to_dispute"]
        print(f"  {code:<14}{s.mean():>7.1f}{s.median():>8.0f}"
              f"{s.quantile(.9):>7.0f}{s.quantile(.99):>7.0f}{s.max():>7}")

    hr("5c. filed_dt vs OBSERVATION BOUNDARY  (right-censoring)")
    obs_end_day = meta["observation_end"] / SPD
    fd = df["filed_dt"] / SPD
    print(f"observation_end = day {obs_end_day:.0f}  "
          f"(data max day {meta['data_window_end']/SPD:.0f} + "
          f"{meta['observation_buffer_days']}d buffer)")
    print(f"right-censored dropped = {meta['n_dropped_right_censored']:,} "
          f"({meta['censor_rate_of_selected']*100:.1f}% of {meta['n_selected']:,} selected)")
    print("\n  filed_dt histogram (observed disputes; day bins):")
    hist, edges = np.histogram(fd, bins=list(range(0, int(obs_end_day) + 15, 15)))
    for i in range(len(hist)):
        bar = "#" * int(hist[i] / max(hist) * 40)
        near = "  <- near boundary" if edges[i + 1] > obs_end_day - 15 else ""
        print(f"   day [{edges[i]:>3.0f},{edges[i+1]:>3.0f})  {hist[i]:>4}  {bar}{near}")

    # ---------------------------------------------------------------- 6
    hr("6. TEMPORAL SPLIT (on filed_dt, section 12)")
    tf = meta["train_fraction"]
    d_sorted = df.sort_values("filed_dt").reset_index(drop=True)
    cut = int(len(d_sorted) * tf)
    filed_boundary = int(d_sorted.loc[cut, "filed_dt"])
    # what the boundary WOULD be if we naively split on TransactionDT instead
    txndt_boundary = int(df["TransactionDT"].sort_values().reset_index(drop=True).loc[cut])
    train, test = d_sorted.iloc[:cut], d_sorted.iloc[cut:]
    SPD = 86400
    print(f"split on filed_dt @ {tf:.0%}:")
    print(f"  filed_dt boundary     : {filed_boundary:,}s  (~day {filed_boundary/SPD:.0f})")
    print(f"  TransactionDT boundary: {txndt_boundary:,}s  (~day {txndt_boundary/SPD:.0f})"
          f"   <- differs; splitting on this would leak")
    print(f"  right-censored dropped : {meta['n_dropped_right_censored']:,}")
    print(f"  train: {len(train):>5} disputes, win rate {train['won'].mean()*100:.1f}%")
    print(f"  test : {len(test):>5} disputes, win rate {test['won'].mean()*100:.1f}%")

    # ---------------------------------------------------------------- 7
    hr("7. AGENT-INITIATED SIGNATURE (agent vs non-agent)")
    df = df.copy()
    df["three_ds_passed"] = (df["three_ds_status"] == "passed").astype(int)
    df["device_matches"] = (df["device_match_status"] == "matched").astype(int)
    ag = df[df["agent_initiated"]]
    non = df[~df["agent_initiated"]]
    feats = ["device_matches", "ip_is_datacenter", "three_ds_passed",
             "customer_contacted_support", "agent_mandate_on_file"]
    print(f"{'feature':<28}{'agent':>10}{'non-agent':>12}")
    for f in feats:
        print(f"  {f:<26}{ag[f].mean():>10.3f}{non[f].mean():>12.3f}")

    print("\n  agent-initiated `device_match_status` value counts "
          "(must be all 'mismatched'):")
    print("   ", ag["device_match_status"].value_counts().to_dict())

    # additional requirement: agent signature broken out BY device_match_status,
    # to confirm distinctiveness on OBSERVED features, not just missingness.
    hr("7b. SIGNATURE BY device_match_status  (observed-feature distinctiveness)")
    print(f"{'device_match_status':<20}{'group':<12}{'n':>6}"
          f"{'ip_dc':>8}{'3ds_pass':>10}{'mandate':>9}{'no_supp':>9}")
    for dmv in ["matched", "mismatched", "unknown"]:
        for label, grp in [("agent", ag), ("non-agent", non)]:
            g = grp[grp["device_match_status"] == dmv]
            if len(g) == 0:
                continue
            print(f"{dmv:<20}{label:<12}{len(g):>6}"
                  f"{g['ip_is_datacenter'].mean():>8.2f}"
                  f"{g['three_ds_passed'].mean():>10.2f}"
                  f"{g['agent_mandate_on_file'].mean():>9.2f}"
                  f"{(1-g['customer_contacted_support']).mean():>9.2f}")
    # the key contrast: agent (mismatched) vs non-agent (mismatched)
    a_mis = ag  # agents are all mismatched
    n_mis = non[non["device_match_status"] == "mismatched"]
    print("\n  KEY CONTRAST — among OBSERVED-mismatched-device cases:")
    print(f"    agent      (n={len(a_mis):>4}): "
          f"ip_dc={a_mis['ip_is_datacenter'].mean():.2f}  "
          f"3ds_pass={a_mis['three_ds_passed'].mean():.2f}  "
          f"mandate={a_mis['agent_mandate_on_file'].mean():.2f}")
    print(f"    non-agent  (n={len(n_mis):>4}): "
          f"ip_dc={n_mis['ip_is_datacenter'].mean():.2f}  "
          f"3ds_pass={n_mis['three_ds_passed'].mean():.2f}  "
          f"mandate={n_mis['agent_mandate_on_file'].mean():.2f}")

    # ---------------------------------------------------------------- 8
    hr("8. LEAKAGE CHECK — logistic regression, temporal test AUC")
    X = build_feature_matrix(d_sorted)
    y = d_sorted["won"].to_numpy()
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ytr, yte = y[:cut], y[cut:]
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(scaler.transform(Xtr), ytr)
    auc = roc_auc_score(yte, clf.predict_proba(scaler.transform(Xte))[:, 1])

    lo, hi = meta["leakage_auc_band"]
    fail_above = meta["leakage_auc_fail_above"]
    print(f"  temporal test AUC = {auc:.3f}")
    print(f"  target band       = {lo:.2f} - {hi:.2f}")
    print(f"  leak threshold    = {fail_above:.2f}")
    if lo <= auc <= hi:
        verdict = GREEN + BOLD + f"PASS — AUC {auc:.3f} in [{lo:.2f}, {hi:.2f}]" + END
    elif auc > fail_above:
        verdict = RED + BOLD + f"FAIL — AUC {auc:.3f} > {fail_above:.2f}: LEAKING, regenerate" + END
    else:
        verdict = YELLOW + BOLD + (
            f"WARN — AUC {auc:.3f} outside band but not leaking "
            f"(<{fail_above:.2f}); consider tightening weights" + END)
    print("\n  " + verdict)


if __name__ == "__main__":
    main()
