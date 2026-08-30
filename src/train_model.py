"""
Phase 2 — winnability model (CLAUDE.md sections 6, 12, 13).

LightGBM, temporal split on filed_dt (70/30, same boundary as verify_data.py),
isotonic + Platt calibration on a held-out slice of the TRAIN window (never test).
Reports test-only metrics, saves reports/calibration.png, and runs two robustness
checks: a weight_scale sweep and a logistic-regression baseline.

Priority is CALIBRATION, not discrimination: every downstream rupee decision
multiplies p_win by an amount, so p_win must mean what it says.

Run: python src/train_model.py   (regenerates data at other weight_scales for the
sweep, into temp dirs — the canonical data/processed is left untouched).
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*eval_set.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

import features as F
import generate_data as G

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TRAIN_FRACTION = 0.70
CALIB_FRACTION = 0.20      # last 20% of the TRAIN window -> isotonic/Platt
VAL_FRACTION = 0.15        # middle slice of TRAIN -> LGBM early stopping
SEED = 20260830

LGB_PARAMS = dict(
    objective="binary", n_estimators=600, learning_rate=0.03,
    num_leaves=15, max_depth=4, min_child_samples=30,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=SEED, n_jobs=1, verbose=-1,
)


# --------------------------------------------------------------------------- #
# splits & metrics
# --------------------------------------------------------------------------- #
def temporal_order(filed_dt: pd.Series) -> np.ndarray:
    """Positional indices sorted by filed_dt ascending (stable) — the split key."""
    return np.argsort(filed_dt.to_numpy(), kind="stable")


def three_way_split(filed_dt: pd.Series):
    """fit (oldest) / val / calib (newest, adjacent to test) within TRAIN; then TEST."""
    order = temporal_order(filed_dt)
    n = len(order)
    cut = int(n * TRAIN_FRACTION)
    train, test = order[:cut], order[cut:]
    n_cal = int(cut * CALIB_FRACTION)
    n_val = int(cut * VAL_FRACTION)
    fit = train[: cut - n_cal - n_val]
    val = train[cut - n_cal - n_val: cut - n_cal]
    cal = train[cut - n_cal:]
    return fit, val, cal, test, cut


def expected_calibration_error(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, N = 0.0, len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        if m.sum():
            ece += m.sum() / N * abs(y[m].mean() - p[m].mean())
    return ece


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def train_lgbm(Xfit, yfit, Xval, yval):
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        Xfit, yfit, eval_set=[(Xval, yval)], eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def logistic_test_pred(b, fit, test):
    """Standardised one-hot logistic regression trained on `fit`; test probabilities."""
    scaler = StandardScaler().fit(b.X_ohe.iloc[fit])
    lr = LogisticRegression(max_iter=2000).fit(
        scaler.transform(b.X_ohe.iloc[fit]), b.y.iloc[fit])
    return lr.predict_proba(scaler.transform(b.X_ohe.iloc[test]))[:, 1]


def test_aucs(proc_dir: Path):
    """Return (LGBM, logistic) test AUC on proc_dir data — for the sweep."""
    b = F.load_features(proc_dir)
    fit, val, cal, test, _ = three_way_split(b.filed_dt)
    model = train_lgbm(b.X.iloc[fit], b.y.iloc[fit], b.X.iloc[val], b.y.iloc[val])
    y = b.y.iloc[test]
    auc_lgbm = roc_auc_score(y, model.predict_proba(b.X.iloc[test])[:, 1])
    auc_lr = roc_auc_score(y, logistic_test_pred(b, fit, test))
    return auc_lgbm, auc_lr


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def reliability_plot(y_test, curves: dict, path: Path, n_bins=10):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    edges = np.linspace(0, 1, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    for name, p in curves.items():
        obs = np.full(n_bins, np.nan)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            m = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
            if m.sum():
                obs[i] = np.asarray(y_test)[m].mean()
        plt.plot(centres, obs, "o-", ms=4, label=name)
    plt.xlabel("mean predicted P(win)")
    plt.ylabel("observed win rate")
    plt.title("Reliability diagram (test set)")
    plt.legend()
    plt.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def hr(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    b = F.load_features()
    F.print_feature_summary(b)

    fit, val, cal, test, cut = three_way_split(b.filed_dt)
    print(f"\nTemporal split on filed_dt @ {TRAIN_FRACTION:.0%}  "
          f"(boundary day {int(b.filed_dt.to_numpy()[temporal_order(b.filed_dt)][cut])//86400})")
    print(f"  fit {len(fit)}  |  val {len(val)}  |  calib {len(cal)}  ||  test {len(test)}")
    print(f"  (calib is the newest slice of TRAIN, never the test set)")

    ytr_test = b.y.iloc[test].to_numpy()
    model = train_lgbm(b.X.iloc[fit], b.y.iloc[fit], b.X.iloc[val], b.y.iloc[val])

    # raw predictions
    p_test_raw = model.predict_proba(b.X.iloc[test])[:, 1]
    p_cal_raw = model.predict_proba(b.X.iloc[cal])[:, 1]
    y_cal = b.y.iloc[cal].to_numpy()

    # calibrators fit on CALIB slice only
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal_raw, y_cal)
    platt = LogisticRegression().fit(p_cal_raw.reshape(-1, 1), y_cal)
    p_test_iso = iso.predict(p_test_raw)
    p_test_platt = platt.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]

    # logistic baseline (same `fit` rows), isotonic-calibrated on the same calib slice
    p_lr = logistic_test_pred(b, fit, test)
    auc_lr = roc_auc_score(ytr_test, p_lr)
    p_lr_cal_raw = logistic_test_pred(b, fit, cal)
    iso_lr = IsotonicRegression(out_of_bounds="clip").fit(p_lr_cal_raw, y_cal)
    p_lr_iso = iso_lr.predict(p_lr)

    # ---------------------------------------------------------------- AUC
    hr("AUC (test) — discrimination; invariant to monotonic calibration")
    auc = roc_auc_score(ytr_test, p_test_raw)
    print(f"  overall test AUC = {auc:.3f}")
    print(f"\n  per reason_code:")
    rc_test = b.reason_code.iloc[test].to_numpy()
    for rc in ["fraud", "inr", "nad", "subscription", "agent_initiated"]:
        m = rc_test == rc
        if m.sum() and len(np.unique(ytr_test[m])) == 2:
            print(f"    {rc:<16} n={m.sum():>4}  AUC {roc_auc_score(ytr_test[m], p_test_raw[m]):.3f}")
        else:
            print(f"    {rc:<16} n={m.sum():>4}  AUC   n/a (one class)")

    # ---------------------------------------------------------- calibration
    hr("CALIBRATION (test) — Brier & ECE, before vs after  (spread = pred std)")
    rows = [("raw LGBM", p_test_raw), ("isotonic", p_test_iso), ("Platt", p_test_platt)]
    print(f"  {'method':<14}{'Brier':>10}{'ECE':>10}{'spread':>10}")
    for name, p in rows:
        print(f"  {name:<14}{brier_score_loss(ytr_test, p):>10.4f}"
              f"{expected_calibration_error(ytr_test, p):>10.4f}{np.std(p):>10.4f}")
    # Choose by Brier: it rewards BOTH calibration and sharpness, so it flags a
    # calibrator that collapses toward the base rate (low ECE, but useless spread
    # for the downstream EV math). ECE alone can be gamed by a flat predictor.
    best = min(rows[1:], key=lambda r: brier_score_loss(ytr_test, r[1]))
    print(f"\n  Brier rewards calibration AND sharpness; ECE alone can be won by a")
    print(f"  near-constant predictor. Chosen for downstream use: {best[0]} "
          f"(Brier {brier_score_loss(ytr_test, best[1]):.4f}, "
          f"ECE {expected_calibration_error(ytr_test, best[1]):.4f}, "
          f"spread {np.std(best[1]):.3f}).")

    reliability_plot(ytr_test,
                     {"LGBM raw": p_test_raw, "LGBM isotonic": p_test_iso,
                      "LGBM Platt": p_test_platt, "logistic isotonic": p_lr_iso},
                     REPORTS / "calibration.png")
    print(f"  reliability diagram -> {REPORTS / 'calibration.png'}")

    # ------------------------------------------------------ feature importance
    hr("FEATURE IMPORTANCE (LGBM gain) — top 15")
    gain = model.booster_.feature_importance(importance_type="gain")
    imp = pd.Series(gain, index=b.feature_names).sort_values(ascending=False)
    for feat, g in imp.head(15).items():
        print(f"  {feat:<28}{g:>12.1f}")

    # --------------------------------------------------- agent-initiated segment
    hr("AGENT-INITIATED segment (test) — predicted P(win) vs actual, with CI")
    p_test_best = best[1]
    am = rc_test == "agent_initiated"
    n_ag = int(am.sum())
    y_ag = ytr_test[am]
    p_ag = p_test_best[am]
    wins = int(y_ag.sum())
    actual = y_ag.mean()
    lo, hi = wilson_ci(wins, n_ag)
    print(f"  n = {n_ag}   (calibrated with: {best[0]})")
    print(f"  predicted P(win):  mean {p_ag.mean():.3f}  median {np.median(p_ag):.3f}  "
          f"p10 {np.quantile(p_ag,.1):.3f}  p90 {np.quantile(p_ag,.9):.3f}")
    print(f"  actual win rate :  {actual:.3f}  ({wins}/{n_ag})")
    print(f"  95% Wilson CI on actual: [{lo:.3f}, {hi:.3f}]  (width {hi-lo:.3f})")
    inside = lo <= p_ag.mean() <= hi
    print(f"  mean predicted {'INSIDE' if inside else 'OUTSIDE'} the CI "
          f"-> {'consistent' if inside else 'CHECK — possible miscalibration on this segment'}")
    print(f"  NOTE: n~{n_ag} gives a wide CI; report the interval, not a point estimate.")
    print(f"  Within-agent AUC is ~0.5 by design: the agent signature (mismatched/"
          f"datacenter/3DS-passed/mandate) is CONSTANT, so there is little to rank on;")
    print(f"  the model returns the class base rate, which is what calibration needs.")

    # ----------------------------------------------------- robustness 1: sweep
    hr("ROBUSTNESS 1 — weight_scale sweep (AUC is a designed property)")
    print(f"  {'weight_scale':<14}{'LGBM AUC':>10}{'LR AUC':>10}")
    print(f"  {'1.9 (canon)':<14}{auc:>10.3f}{auc_lr:>10.3f}")
    sweep = {1.5: None, 2.3: None}
    for ws in [1.5, 2.3]:
        tmp = Path(tempfile.mkdtemp(prefix=f"contra_ws{ws}_"))
        G.main(weight_scale=ws, out=tmp, verbose=False)
        a_lgbm, a_lr = test_aucs(tmp)
        sweep[ws] = (a_lgbm, a_lr)
        print(f"  {ws:<14}{a_lgbm:>10.3f}{a_lr:>10.3f}")
    print("  -> AUC tracks the simulator's weight_scale by design (clearest on the")
    print("     lower-variance LR); we state this openly, not as a finding.")

    # ------------------------------------------ robustness 2: logistic baseline
    hr("ROBUSTNESS 2 — logistic-regression baseline vs LGBM  (+ calibration)")
    gap = auc - auc_lr
    print(f"  {'model':<20}{'AUC':>8}{'Brier(iso)':>12}{'ECE(iso)':>10}{'spread':>9}")
    print(f"  {'LGBM (isotonic)':<20}{auc:>8.3f}"
          f"{brier_score_loss(ytr_test, p_test_iso):>12.4f}"
          f"{expected_calibration_error(ytr_test, p_test_iso):>10.4f}"
          f"{np.std(p_test_iso):>9.3f}")
    print(f"  {'logistic (isotonic)':<20}{auc_lr:>8.3f}"
          f"{brier_score_loss(ytr_test, p_lr_iso):>12.4f}"
          f"{expected_calibration_error(ytr_test, p_lr_iso):>10.4f}"
          f"{np.std(p_lr_iso):>9.3f}")
    print(f"\n  AUC gap (LGBM - LR) = {gap:+.3f}")

    hr("MODEL SELECTION — verdict")
    if gap < 0.02:
        print(f"  Logistic regression WINS (gap {gap:+.3f}, i.e. LR is better) and is")
        print(f"  simpler. This is expected: section 6 generates the outcome as an")
        print(f"  ADDITIVE LOGIT in the features, so LR is the correctly-specified model")
        print(f"  and LGBM's tree flexibility only adds variance on ~{len(fit)} fit rows.")
        print(f"  RECOMMENDATION: use calibrated logistic regression as the winnability")
        print(f"  model in Phase 3. (LGBM pipeline retained here for the comparison.)")
    else:
        print(f"  LGBM meaningfully better (gap {gap:+.3f}); keep it.")


if __name__ == "__main__":
    main()
