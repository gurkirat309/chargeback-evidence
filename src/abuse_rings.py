"""
Abuse-ring detective (defense-only) — CLAUDE.md track direction "Abuse-ring
sentinel".

IEEE-CIS is anonymised, so it carries no device fingerprint / card BIN / shipping
address to link disputes into coordinated rings. So this module builds its own
small SYNTHETIC signature layer — a background of independent disputes plus a few
PLANTED rings whose members share a device fingerprint, a card BIN and a shipping
city and file in a tight window — and detects them with a real graph clusterer.

Because the rings are planted, we can report honest DETECTION metrics
(precision / recall / F1 on ring membership) — a known-answer eval, in keeping
with the project's honest-metrics stance. Deterministic from a seed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# --- synthetic-signature layer knobs (self-contained; commented for the README) ---
SEED = 71
N_BACKGROUND = 320          # independent, non-ring disputes
N_BINS = 60                 # distinct card BINs in the background
N_CITIES = 45               # distinct shipping cities
BACKGROUND_DC_RATE = 0.06   # datacenter-IP rate for ordinary disputes
N_NOISE_PAIRS = 12          # background pairs that coincidentally share a device
                            # (family/office device) — the detector must NOT
                            # call these rings -> keeps precision honest (<1.0)
RINGS = [                   # planted OBVIOUS rings (shared device, burst, proxies)
    {"size": 7, "amount": 16500, "product": "electronics"},
    {"size": 6, "amount": 8900,  "product": "gift_cards"},
    {"size": 5, "amount": 24000, "product": "electronics"},
    {"size": 4, "amount": 4200,  "product": "fashion"},
]
# One STEALTH ring: members avoid a shared device, spread filing over weeks, use
# residential IPs. It SHOULD evade a velocity/device detector — so recall drops
# and we report the limitation honestly rather than faking a perfect score.
STEALTH_RINGS = [{"size": 6, "amount": 30000, "product": "electronics"}]
MIN_RING_SIZE = 3           # a detected cluster must have >= this many members

_CITIES = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Pune",
           "Kolkata", "Ahmedabad", "Jaipur", "Surat", "Lucknow", "Indore"]
_PRODUCTS = ["electronics", "gift_cards", "fashion", "home", "beauty"]


def _rng():
    return np.random.default_rng(SEED)


def build_dataset() -> pd.DataFrame:
    """Background disputes + planted rings, with a shared-attribute signature."""
    rng = _rng()
    rows = []
    dev_ctr = [0]

    def dev():
        dev_ctr[0] += 1
        return f"dev_{rng.integers(16**8):08x}{dev_ctr[0]:04d}"

    # --- background: independent disputes, mostly-unique devices ---
    bins = [f"{rng.integers(400000, 999999)}" for _ in range(N_BINS)]
    for i in range(N_BACKGROUND):
        rows.append({
            "id": f"BG{i:04d}", "device_fp": dev(),
            "card_bin": rng.choice(bins),
            "ship_city": rng.choice(_CITIES[: N_CITIES] if N_CITIES < len(_CITIES) else _CITIES),
            "product": rng.choice(_PRODUCTS),
            "amount_inr": float(round(rng.gamma(2.0, 4000), -1)),
            "ip_is_datacenter": bool(rng.random() < BACKGROUND_DC_RATE),
            "filed_day": int(rng.integers(0, 180)),
            "is_ring": False, "ring_id": -1,
        })

    # --- noise: coincidental shared devices (NOT rings) ---
    for _ in range(N_NOISE_PAIRS):
        a, b = rng.choice(N_BACKGROUND, size=2, replace=False)
        rows[b]["device_fp"] = rows[a]["device_fp"]

    # --- planted rings: shared device + BIN + city, tight window, high risk ---
    for r_idx, spec in enumerate(RINGS):
        d = dev()
        b = f"{rng.integers(400000, 999999)}"
        city = rng.choice(_CITIES)
        start = int(rng.integers(0, 170))
        for j in range(spec["size"]):
            rows.append({
                "id": f"RING{r_idx}_{j}", "device_fp": d, "card_bin": b,
                "ship_city": city, "product": spec["product"],
                "amount_inr": float(round(spec["amount"] * rng.uniform(0.9, 1.1), -1)),
                "ip_is_datacenter": bool(rng.random() < 0.85),   # rings ride proxies
                "filed_day": int(start + rng.integers(0, 5)),     # tight burst
                "is_ring": True, "ring_id": r_idx,
            })

    # --- stealth rings: shared BIN+city only, unique devices, slow burn, residential
    for s_idx, spec in enumerate(STEALTH_RINGS):
        b = f"{rng.integers(400000, 999999)}"
        city = rng.choice(_CITIES)
        for j in range(spec["size"]):
            rows.append({
                "id": f"STEALTH{s_idx}_{j}", "device_fp": dev(),   # unique per member
                "card_bin": b, "ship_city": city, "product": spec["product"],
                "amount_inr": float(round(spec["amount"] * rng.uniform(0.9, 1.1), -1)),
                "ip_is_datacenter": bool(rng.random() < 0.10),      # residential
                "filed_day": int(rng.integers(0, 60)),              # spread over weeks
                "is_ring": True, "ring_id": 100 + s_idx,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# detector: union-find over shared signatures
# --------------------------------------------------------------------------- #
class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def detect(df: pd.DataFrame):
    """Link disputes that share a strong signature, cluster, flag coordinated rings.

    Linking rule: same device fingerprint (strong), OR same card BIN *and* same
    shipping city (corroborating). Connected components of >= MIN_RING_SIZE that
    look coordinated (shared device + burst filing / datacenter IP) are flagged."""
    n = len(df)
    uf = _UF(n)
    by_dev, by_bincity = {}, {}
    for i, row in df.iterrows():
        by_dev.setdefault(row["device_fp"], []).append(i)
        by_bincity.setdefault((row["card_bin"], row["ship_city"]), []).append(i)
    for members in list(by_dev.values()) + list(by_bincity.values()):
        for k in range(1, len(members)):
            uf.union(members[0], members[k])

    comps = {}
    for i in range(n):
        comps.setdefault(uf.find(i), []).append(i)

    rings, flagged_idx = [], set()
    for members in comps.values():
        if len(members) < MIN_RING_SIZE:
            continue
        g = df.iloc[members]
        shared_dev = g["device_fp"].nunique() <= max(1, len(g) // 3)
        dc = float(g["ip_is_datacenter"].mean())
        burst = int(g["filed_day"].max() - g["filed_day"].min())
        # coordinated if a device is shared across the cluster AND it bursts / rides proxies
        if not (shared_dev and (burst <= 10 or dc >= 0.4)):
            continue
        flagged_idx.update(members)
        risk = round(min(1.0, 0.4 * (len(g) / 6) + 0.4 * dc + 0.2 * (burst <= 7)), 2)
        rings.append({
            "size": len(g),
            "device_fp": g["device_fp"].mode().iat[0],
            "card_bin": g["card_bin"].mode().iat[0],
            "ship_city": g["ship_city"].mode().iat[0],
            "product": g["product"].mode().iat[0],
            "amount_avg_inr": round(float(g["amount_inr"].mean())),
            "datacenter_pct": round(dc, 2),
            "filing_burst_days": burst,
            "risk": risk,
            "members": g["id"].tolist(),
        })
    rings.sort(key=lambda r: (r["risk"], r["size"]), reverse=True)

    # honest metrics vs the planted labels
    truth = set(df.index[df["is_ring"]].tolist())
    tp = len(flagged_idx & truth)
    fp = len(flagged_idx - truth)
    fn = len(truth - flagged_idx)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    metrics = {"planted_ring_disputes": len(truth), "flagged_disputes": len(flagged_idx),
               "rings_detected": len(rings), "precision": round(prec, 2),
               "recall": round(rec, 2), "f1": round(f1, 2)}
    return rings, metrics


if __name__ == "__main__":
    df = build_dataset()
    rings, m = detect(df)
    print(f"dataset: {len(df)} disputes ({df['is_ring'].sum()} planted ring members)")
    print("metrics:", m)
    for r in rings:
        print(f"  RING size {r['size']} · {r['product']} ~Rs{r['amount_avg_inr']} · "
              f"{r['ship_city']} · {r['datacenter_pct']*100:.0f}% datacenter · "
              f"burst {r['filing_burst_days']}d · risk {r['risk']} · dev {r['device_fp'][:12]}")
