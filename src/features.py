"""
Phase 2 — feature matrix for the winnability model (CLAUDE.md sections 5, 13).

Builds features from evidence.parquet + dispute-level fields. Everything the
model must NOT see (ground-truth generator internals, the label, identifiers) is
listed in FORBIDDEN and asserted absent. Import `load_features`; run as a script
to print the final feature list.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# Dispute-level numeric/boolean features.
DISPUTE_NUMERIC = ["days_txn_to_dispute", "disputed_amount_inr", "agent_initiated"]

# Evidence numeric/boolean artifacts.
EVIDENCE_NUMERIC = [
    "avs_match", "cvv_match", "prior_txn_count", "prior_undisputed_count",
    "account_age_days", "delivery_otp_verified", "ip_to_ship_km",
    "ip_is_datacenter", "customer_contacted_support", "support_ticket_count",
    "product_photos_on_file", "refund_policy_shown", "consent_record_exists",
    "agent_mandate_on_file",
]

# Categoricals (native for LightGBM, one-hot for linear models).
CATEGORICAL = ["reason_code", "three_ds_status", "device_match_status",
               "delivery_proof_type"]

NUMERIC = DISPUTE_NUMERIC + EVIDENCE_NUMERIC
ALL_FEATURES = NUMERIC + CATEGORICAL

# Must NEVER be a feature: generator internals, the label, identifiers, or any
# field that encodes the split/time directly. Asserted below.
FORBIDDEN = {
    "p_win_true", "issuer_leniency", "issuer_group",   # generator ground truth
    "isFraud",                                          # underlying fraud label
    "fought",                                           # censoring policy, not evidence
    "won",                                              # the target
    "dispute_id", "TransactionID",                      # identifiers
    "filed_dt", "deadline_dt", "TransactionDT",         # time / leakage
    "has_identity",                                     # implied by device_match_status
    "amount_inr",                                       # pre-round duplicate
}


@dataclass
class FeatureBundle:
    X: pd.DataFrame          # native dtypes; categoricals as pandas 'category'
    X_ohe: pd.DataFrame      # one-hot numeric version for linear models
    y: pd.Series             # won (0/1)
    filed_dt: pd.Series      # split key ONLY — never a feature
    reason_code: pd.Series   # for segmented reporting ONLY
    feature_names: list


def load_features(proc_dir: Path = PROC) -> FeatureBundle:
    disputes = pd.read_parquet(proc_dir / "disputes.parquet")
    evidence = pd.read_parquet(proc_dir / "evidence.parquet")
    outcomes = pd.read_parquet(proc_dir / "outcomes.parquet")

    df = disputes.merge(evidence, on="dispute_id").merge(outcomes, on="dispute_id")

    X = df[ALL_FEATURES].copy()
    X["agent_initiated"] = X["agent_initiated"].astype(int)
    for col in EVIDENCE_NUMERIC:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    for col in CATEGORICAL:
        X[col] = X[col].astype("category")

    # hard guarantee: nothing forbidden leaked into the matrix
    leaked = FORBIDDEN & set(X.columns)
    assert not leaked, f"FORBIDDEN feature(s) leaked into X: {leaked}"
    assert "won" not in X.columns and "p_win_true" not in X.columns

    X_ohe = pd.get_dummies(X, columns=CATEGORICAL, dtype=int)

    return FeatureBundle(
        X=X,
        X_ohe=X_ohe,
        y=df["won"].astype(int),
        filed_dt=df["filed_dt"],
        reason_code=df["reason_code"],
        feature_names=list(X.columns),
    )


def print_feature_summary(b: FeatureBundle) -> None:
    print("=" * 70)
    print(f"FEATURE MATRIX — {b.X.shape[0]:,} rows x {b.X.shape[1]} features")
    print("=" * 70)
    print(f"\nNumeric / boolean ({len(NUMERIC)}):")
    for c in NUMERIC:
        print(f"  - {c:<28} {str(b.X[c].dtype):<10}")
    print(f"\nCategorical ({len(CATEGORICAL)}) — native for LGBM, "
          f"one-hot ({b.X_ohe.shape[1]} cols) for linear:")
    for c in CATEGORICAL:
        cats = list(b.X[c].cat.categories)
        print(f"  - {c:<28} {cats}")
    print(f"\nTarget: won  (win rate {b.y.mean()*100:.1f}%)")
    print(f"Split key (NOT a feature): filed_dt")
    print(f"\nExcluded & asserted absent: {sorted(FORBIDDEN)}")


if __name__ == "__main__":
    print_feature_summary(load_features())
