"""
Phase 5 — evaluation harness (CLAUDE.md section 12). The headline deliverable.

Evaluates on the TEMPORAL TEST split only (out-of-sample p_win from
predictions.parquet). Reports, in rupees:
  * precision / recall / F1 on the FIGHT decision
  * calibration (Brier, ECE) on test
  * segmented metrics by reason code and amount band
  * rupee-denominated confusion matrix (FP = fee+analyst, FN = recoverable amount)
  * threshold sweep with net-recovery curve -> reports/net_recovery_curve.png
  * the baseline comparison table (the headline): RokdaDaav vs fight-everything,
    fight-nothing, fight-if-amount>Rs2000, fight-if-p_win>0.5

Money accounting (relative to the fight-nothing baseline, consistent with the
section 9 EVs):
  FIGHT  realized = won*amount - cost_to_fight
  ACCEPT realized = 0
  REFUND realized = -amount + ratio_benefit(current_ratio)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decision_engine as DE
from features import PROC

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def ece(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    e, N = 0.0, len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        if m.sum():
            e += m.sum() / N * abs(y[m].mean() - p[m].mean())
    return e


# --------------------------------------------------------------------------- #
# money
# --------------------------------------------------------------------------- #
def realized_net(action, won, amount, ratio, c: DE.Costs):
    """Per-dispute realised recovery (rupees) given the true outcome `won`."""
    ctf = DE.cost_to_fight(c)
    out = np.zeros(len(action), dtype=float)
    out[action == DE.FIGHT] = (won * amount - ctf)[action == DE.FIGHT]
    # ACCEPT -> 0
    rb = DE.ratio_benefit(ratio, c)
    out[action == DE.REFUND] = (-amount + rb)[action == DE.REFUND]
    return out


def net_recovery(action, won, amount, ratio, c):
    return realized_net(np.asarray(action), won, amount, ratio, c).sum()


# --------------------------------------------------------------------------- #
# policies -> action arrays
# --------------------------------------------------------------------------- #
def policy_fight_all(df):
    return np.full(len(df), DE.FIGHT)


def policy_fight_none(df):
    return np.full(len(df), DE.ACCEPT)


def policy_fight_amount(df, thr):
    return np.where(df["disputed_amount_inr"].to_numpy() > thr, DE.FIGHT, DE.ACCEPT)


def policy_fight_pwin(df, thr):
    return np.where(df["p_win"].to_numpy() > thr, DE.FIGHT, DE.ACCEPT)


def policy_contra(df, ratio, c, escalation_rate, hours_budget):
    out = DE.decide(df, current_ratio=ratio, c=c, escalation_rate=escalation_rate,
                    hours_budget=hours_budget)
    action = out["action"].to_numpy().copy()
    # ESCALATE goes to a human; for realised accounting, the human follows the
    # engine's pre-escalation EV choice (no oracle boost).
    esc = action == DE.ESCALATE
    action[esc] = out["base_action"].to_numpy()[esc]
    return action


# --------------------------------------------------------------------------- #
# reporting pieces
# --------------------------------------------------------------------------- #
def hr(t):
    print("\n" + "=" * 74 + "\n" + t + "\n" + "=" * 74)


def rupee_confusion(action, won, amount, c):
    """FIGHT-decision confusion in rupees (section 12)."""
    ctf = DE.cost_to_fight(c)
    fight = action == DE.FIGHT
    won = won.astype(bool)
    tp = fight & won
    fp = fight & ~won
    fn = ~fight & won
    tn = ~fight & ~won
    return {
        "TP (fought, won)":  (int(tp.sum()), float((amount[tp] - ctf).sum())),
        "FP (fought, lost)": (int(fp.sum()), float(-(ctf * fp.sum()))),
        "FN (missed a win)": (int(fn.sum()), float(-(amount[fn]).sum())),
        "TN (correctly skipped)": (int(tn.sum()), 0.0),
    }


def net_recovery_curve(df, won, amount, ratio, c):
    ts = np.linspace(0, 1, 101)
    nets = [net_recovery(policy_fight_pwin(df, t), won, amount, ratio, c) for t in ts]
    nets = np.array(nets)
    best_i = int(np.argmax(nets))
    plt.figure(figsize=(7, 4.5))
    plt.plot(ts, nets / 1000, lw=2)
    plt.axvline(ts[best_i], color="crimson", ls="--",
                label=f"optimum t={ts[best_i]:.2f}  (Rs {nets[best_i]/1000:.0f}k)")
    plt.xlabel("fight if p_win > t")
    plt.ylabel("net recovery (Rs thousands)")
    plt.title("Net-recovery threshold sweep (test set)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    REPORTS.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORTS / "net_recovery_curve.png", dpi=120); plt.close()
    return ts[best_i], nets[best_i]


def main():
    with open(ROOT / "config" / "generator.yaml", "r", encoding="utf-8") as fh:
        g = yaml.safe_load(fh)["evaluation"]
    c = DE.Costs.from_yaml()
    ratio = g["assumed_current_ratio"]
    escalation_rate = g["escalation_rate_target"]
    amt_thr = g["fight_if_amount_threshold_inr"]

    preds = pd.read_parquet(PROC / "predictions.parquet")
    disputes = pd.read_parquet(PROC / "disputes.parquet")[["dispute_id", "deadline_dt"]]
    df = preds.merge(disputes, on="dispute_id")
    test = df[df["split"] == "test"].reset_index(drop=True)
    won = test["won"].to_numpy()
    amount = test["disputed_amount_inr"].to_numpy()

    days = (test["filed_dt"].max() - test["filed_dt"].min()) / 86400
    hours_budget = c.analyst_hours_per_day * days      # capacity over the test window

    hr(f"EVALUATION — test split (n={len(test)}), assumed dispute-ratio {ratio}")
    print(f"  cost_to_fight = Rs {DE.cost_to_fight(c):.0f}  | escalation_rate "
          f"{escalation_rate} | analyst-hour budget {hours_budget:.0f}h "
          f"(6h/day x {days:.0f} days)")

    # --- calibration on test ---
    hr("CALIBRATION (test)")
    p = test["p_win"].to_numpy()
    print(f"  Brier {brier_score_loss(won, p):.4f}   ECE {ece(won, p):.4f}   "
          f"mean p_win {p.mean():.3f}   base win rate {won.mean():.3f}")

    # --- RokdaDaav actions (used for P/R/F1, confusion, segments) ---
    contra = policy_contra(test, ratio, c, escalation_rate, hours_budget)
    fight = contra == DE.FIGHT

    hr("FIGHT DECISION — precision / recall / F1  (positive = winnable, won==1)")
    print(f"  precision {precision_score(won, fight):.3f}   "
          f"recall {recall_score(won, fight):.3f}   F1 {f1_score(won, fight):.3f}")
    print(f"  ({fight.sum()} fought of {len(test)}; {won.sum()} winnable)")

    hr("RUPEE CONFUSION MATRIX — RokdaDaav FIGHT decision")
    print(f"  {'cell':<26}{'count':>8}{'rupees':>16}")
    for k, (n, v) in rupee_confusion(contra, won, amount, c).items():
        print(f"  {k:<26}{n:>8}{v:>16,.0f}")

    # --- segments ---
    hr("SEGMENTED NET RECOVERY (RokdaDaav) — by reason code")
    for rc in ["fraud", "inr", "nad", "subscription", "agent_initiated"]:
        m = test["reason_code"].to_numpy() == rc
        if m.sum():
            nr = net_recovery(contra[m], won[m], amount[m], ratio, c)
            print(f"  {rc:<16} n={m.sum():>4}  net Rs {nr:>12,.0f}  "
                  f"fought {int((contra[m]==DE.FIGHT).sum()):>3}  won {int(won[m].sum()):>3}")

    hr("SEGMENTED NET RECOVERY (RokdaDaav) — by amount band (INR)")
    bands = [(0, 2000), (2000, 6000), (6000, 15000), (15000, 1e12)]
    for lo, hi in bands:
        m = (amount >= lo) & (amount < hi)
        if m.sum():
            nr = net_recovery(contra[m], won[m], amount[m], ratio, c)
            label = f"{lo:,.0f}-{hi:,.0f}" if hi < 1e12 else f">{lo:,.0f}"
            print(f"  {label:<18} n={m.sum():>4}  net Rs {nr:>12,.0f}  "
                  f"fought {int((contra[m]==DE.FIGHT).sum()):>3}")

    # --- threshold sweep curve ---
    hr("THRESHOLD SWEEP — net recovery vs p_win cutoff")
    best_t, best_net = net_recovery_curve(test, won, amount, ratio, c)
    print(f"  optimum: fight if p_win > {best_t:.2f}  ->  net Rs {best_net:,.0f}")
    print(f"  curve saved -> {REPORTS / 'net_recovery_curve.png'}")

    # --- THE HEADLINE TABLE ---
    hr("BASELINE COMPARISON — net recovery (rupees), test split  [HEADLINE]")
    policies = {
        "fight everything": policy_fight_all(test),
        "fight nothing": policy_fight_none(test),
        f"fight if amount > Rs{amt_thr}": policy_fight_amount(test, amt_thr),
        "fight if p_win > 0.5": policy_fight_pwin(test, 0.5),
        "RokdaDaav (EV + ratio + capacity)": contra,
    }
    results = {name: net_recovery(a, won, amount, ratio, c) for name, a in policies.items()}
    print(f"  {'policy':<34}{'net recovery Rs':>18}")
    for name, nr in results.items():
        star = "  <--" if name.startswith("RokdaDaav") else ""
        print(f"  {name:<34}{nr:>18,.0f}{star}")

    baseline = results[f"fight if amount > Rs{amt_thr}"]
    contra_net = results["RokdaDaav (EV + ratio + capacity)"]
    hr("VERDICT")
    diff = contra_net - baseline
    if diff > 0:
        print(f"  RokdaDaav beats 'fight if amount > Rs{amt_thr}' by Rs {diff:,.0f} "
              f"(+{diff/abs(baseline)*100:.1f}%). There is a product.")
    else:
        print(f"  RokdaDaav does NOT beat 'fight if amount > Rs{amt_thr}' "
              f"(Rs {diff:,.0f}). Report honestly: no product yet.")

    # --- ratio & capacity sensitivity (why RokdaDaav is more than a p_win threshold) ---
    hr("SENSITIVITY — dispute-ratio (refunds) and capacity")
    for r in [0.50, 0.85, 0.95]:
        a = policy_contra(test, r, c, escalation_rate, hours_budget)
        n_ref = int((a == DE.REFUND).sum())
        print(f"  ratio {r:.2f}: net Rs {net_recovery(a, won, amount, r, c):>12,.0f}  "
              f"refunds {n_ref:>3}  ratio_benefit Rs {DE.ratio_benefit(r, c):.0f}")
    tight = c.analyst_hours_per_case if hasattr(c, "analyst_hours_per_case") else DE.analyst_hours_per_case(c)
    for frac, lbl in [(0.25, "tight"), (1.0, "full")]:
        a = policy_contra(test, ratio, c, escalation_rate, hours_budget * frac)
        print(f"  capacity {lbl} ({hours_budget*frac:.0f}h): net Rs "
              f"{net_recovery(a, won, amount, ratio, c):>12,.0f}  "
              f"fought {int((a==DE.FIGHT).sum()):>3}")

    hr("CAVEAT — censored labels (section 12)")
    print("  We only observe outcomes for cases that were (synthetically) resolved.")
    print("  In production, outcomes exist only for FOUGHT cases, so real training")
    print("  data is biased; correcting it needs a randomized holdout where some")
    print("  low-p_win cases are fought anyway. Stated in the README and the video.")


if __name__ == "__main__":
    main()
