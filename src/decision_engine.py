"""
Phase 3 — decision engine (CLAUDE.md section 9).

Pure functions. No ML, no LLM, no randomness. Given a calibrated p_win and the
rupee constants, decide FIGHT / ACCEPT / REFUND / ESCALATE by explicit expected
value, then apply the capacity constraint. Every number traces to config/costs.yaml
(money) and config/generator.yaml (escalation_rate_target). Unit-tested.

Design notes
- EV_fight = p_win * amount - cost_to_fight
- EV_accept = 0
- EV_refund = -amount + ratio_benefit(current_ratio)   # sometimes refund a winnable
                                                        # case for dispute-ratio relief
- ESCALATE overrides the EV choice for the top N% by uncertainty*amount.
- Capacity: FIGHT cases are taken greedily by EV-per-analyst-hour until the hour
  budget is spent, with deadline pre-emption for cases due within 48h; the rest
  fall back to their next-best EV option.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

FIGHT, ACCEPT, REFUND, ESCALATE = "FIGHT", "ACCEPT", "REFUND", "ESCALATE"
SECONDS_PER_DAY = 86400
URGENT_WINDOW_S = 48 * 3600          # deadline pre-emption window (section 9)


@dataclass(frozen=True)
class Costs:
    representment_fee_inr: float
    analyst_minutes_per_case: float
    analyst_cost_per_hour_inr: float
    analyst_hours_per_day: float
    vamp_threshold_pct: float
    vamp_warning_band_pct: float
    vamp_fine_per_dispute_inr: float

    @classmethod
    def from_yaml(cls, path: Path = ROOT / "config" / "costs.yaml") -> "Costs":
        with open(path, "r", encoding="utf-8") as fh:
            c = yaml.safe_load(fh)
        return cls(
            representment_fee_inr=c["representment_fee_inr"],
            analyst_minutes_per_case=c["analyst_minutes_per_case"],
            analyst_cost_per_hour_inr=c["analyst_cost_per_hour_inr"],
            analyst_hours_per_day=c["analyst_hours_per_day"],
            vamp_threshold_pct=c["vamp_threshold_pct"],
            vamp_warning_band_pct=c["vamp_warning_band_pct"],
            vamp_fine_per_dispute_inr=c["vamp_fine_per_dispute_inr"],
        )


# --------------------------------------------------------------------------- #
# per-case arithmetic
# --------------------------------------------------------------------------- #
def analyst_hours_per_case(c: Costs) -> float:
    return c.analyst_minutes_per_case / 60.0


def cost_to_fight(c: Costs) -> float:
    """Representment fee + analyst time to assemble and file one representment."""
    return c.representment_fee_inr + analyst_hours_per_case(c) * c.analyst_cost_per_hour_inr


def ratio_benefit(current_ratio: float, c: Costs) -> float:
    """Dispute-ratio relief from refunding, piecewise & monotonic in current_ratio.

    Below (threshold - warning_band): 0. Across the warning band: ramps linearly
    0 -> fine_per_dispute. At/above threshold: fine_per_dispute * 3 (a merchant in
    the program pays the fine on every dispute, so relief is worth much more)."""
    threshold = c.vamp_threshold_pct
    band = c.vamp_warning_band_pct
    fine = c.vamp_fine_per_dispute_inr
    if current_ratio < threshold - band:
        return 0.0
    if current_ratio < threshold:
        scale = (current_ratio - (threshold - band)) / band     # 0 -> 1 across band
        return fine * scale
    return fine * 3.0


def ev_fight(p_win, amount, c: Costs):
    return p_win * amount - cost_to_fight(c)


def ev_accept():
    return 0.0


def ev_refund(amount, current_ratio, c: Costs):
    return -amount + ratio_benefit(current_ratio, c)


def uncertainty(p_win):
    """Peaks at p_win=0.5 (=1), zero at the extremes. Used for escalation ranking."""
    return 1.0 - 2.0 * np.abs(p_win - 0.5)


# --------------------------------------------------------------------------- #
# batch decision
# --------------------------------------------------------------------------- #
def _best_non_fight(ev_accept_v, ev_refund_v):
    """Next-best action when a FIGHT case does not fit the capacity budget."""
    return np.where(ev_refund_v > ev_accept_v, REFUND, ACCEPT)


def capacity_allocate(ev_fight_v, hours_per_case, hours_budget, urgent):
    """Greedy selection of FIGHT cases under an analyst-hour budget.

    urgent (bool array) cases are pre-empted first (deadline within 48h), each
    group ranked by EV-per-analyst-hour (== EV, since hours_per_case is constant).
    Returns a boolean 'selected' array aligned to the inputs."""
    n = len(ev_fight_v)
    selected = np.zeros(n, dtype=bool)
    # rank key: urgent first, then higher EV-per-hour
    order = sorted(range(n), key=lambda i: (not urgent[i], -ev_fight_v[i] / hours_per_case))
    spent = 0.0
    for i in order:
        if spent + hours_per_case <= hours_budget + 1e-9:
            selected[i] = True
            spent += hours_per_case
    return selected


def decide(df: pd.DataFrame, current_ratio: float, c: Costs,
           escalation_rate: float, hours_budget: float, now_dt=None) -> pd.DataFrame:
    """Decide an action for every dispute in df.

    df needs: p_win, disputed_amount_inr, deadline_dt (for capacity pre-emption).
    Returns df + ev columns, escalation flags, and the final `action`.
    hours_budget is the analyst-hour capacity for this decision batch.
    now_dt (int seconds) is the decision time; defaults to the latest filing."""
    out = df.copy().reset_index(drop=True)
    p = out["p_win"].to_numpy()
    amt = out["disputed_amount_inr"].to_numpy()

    out["ev_fight"] = ev_fight(p, amt, c)
    out["ev_accept"] = 0.0
    out["ev_refund"] = ev_refund(amt, current_ratio, c)

    evs = np.column_stack([out["ev_fight"], out["ev_accept"], out["ev_refund"]])
    base = np.array([FIGHT, ACCEPT, REFUND])[np.argmax(evs, axis=1)]
    out["base_action"] = base
    out["chosen_ev"] = evs.max(axis=1)

    # --- ESCALATE override: top N% by uncertainty * amount ---
    out["escalation_score"] = uncertainty(p) * amt
    if escalation_rate > 0 and len(out) > 0:
        k = int(round(len(out) * escalation_rate))
        if k > 0:
            cutoff = np.sort(out["escalation_score"].to_numpy())[-k]
            escalated = out["escalation_score"].to_numpy() >= cutoff
            # keep exactly k by breaking ties deterministically on index
            if escalated.sum() > k:
                idx = np.argsort(-out["escalation_score"].to_numpy(), kind="stable")[:k]
                escalated = np.zeros(len(out), dtype=bool)
                escalated[idx] = True
        else:
            escalated = np.zeros(len(out), dtype=bool)
    else:
        escalated = np.zeros(len(out), dtype=bool)
    out["escalated"] = escalated

    action = np.where(escalated, ESCALATE, base)

    # --- capacity constraint on the FIGHT queue (escalated cases already removed) ---
    if now_dt is None:
        now_dt = int(out["deadline_dt"].to_numpy().max() - 30 * SECONDS_PER_DAY) \
            if "deadline_dt" in out else 0
    hpc = analyst_hours_per_case(c)
    fight_mask = action == FIGHT
    fight_idx = np.where(fight_mask)[0]
    out["capacity_selected"] = False
    if len(fight_idx):
        urgent = (out["deadline_dt"].to_numpy()[fight_idx] - now_dt) <= URGENT_WINDOW_S
        sel = capacity_allocate(out["ev_fight"].to_numpy()[fight_idx], hpc, hours_budget, urgent)
        out.loc[fight_idx, "capacity_selected"] = sel
        # unselected FIGHT cases fall back to their next-best EV option
        dropped = fight_idx[~sel]
        if len(dropped):
            fallback = _best_non_fight(out["ev_accept"].to_numpy()[dropped],
                                       out["ev_refund"].to_numpy()[dropped])
            action[dropped] = fallback

    out["action"] = action
    return out


def load_config():
    """Convenience: (Costs, escalation_rate_target) from the config files."""
    c = Costs.from_yaml()
    with open(ROOT / "config" / "generator.yaml", "r", encoding="utf-8") as fh:
        g = yaml.safe_load(fh)
    return c, g["evaluation"]["escalation_rate_target"]
