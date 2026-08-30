"""
Phase 3 — decision-engine unit tests (CLAUDE.md sections 9, 15).

Pure arithmetic, so everything here is exact and deterministic. Run: pytest -q.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import decision_engine as de


@pytest.fixture
def c():
    # explicit constants so the tests do not depend on config edits
    return de.Costs(
        representment_fee_inr=2500,
        analyst_minutes_per_case=35,
        analyst_cost_per_hour_inr=400,
        analyst_hours_per_day=6,
        vamp_threshold_pct=0.90,
        vamp_warning_band_pct=0.15,
        vamp_fine_per_dispute_inr=800,
    )


# ---------------------------------------------------------------- cost_to_fight
def test_cost_to_fight(c):
    # 2500 + (35/60)*400 = 2500 + 233.333...
    assert de.cost_to_fight(c) == pytest.approx(2500 + 35 / 60 * 400)
    assert de.analyst_hours_per_case(c) == pytest.approx(35 / 60)


# ---------------------------------------------------------------- ratio_benefit
def test_ratio_benefit_below_band_is_zero(c):
    assert de.ratio_benefit(0.50, c) == 0.0
    # just below the warning band start (0.90 - 0.15 = 0.75)
    assert de.ratio_benefit(0.7499, c) == 0.0


def test_ratio_benefit_ramps_across_band(c):
    assert de.ratio_benefit(0.75, c) == pytest.approx(0.0)          # band start
    assert de.ratio_benefit(0.825, c) == pytest.approx(400.0)       # halfway -> 0.5*800
    assert de.ratio_benefit(0.8999, c) == pytest.approx(800 * (0.8999 - 0.75) / 0.15)


def test_ratio_benefit_above_threshold_is_triple(c):
    assert de.ratio_benefit(0.90, c) == pytest.approx(2400.0)       # 800 * 3
    assert de.ratio_benefit(0.99, c) == pytest.approx(2400.0)


def test_ratio_benefit_monotonic_nondecreasing(c):
    xs = np.linspace(0.0, 1.0, 200)
    ys = [de.ratio_benefit(x, c) for x in xs]
    assert all(b >= a - 1e-9 for a, b in zip(ys, ys[1:]))


# ------------------------------------------------------------------------- EVs
def test_ev_fight_monotonic_in_pwin(c):
    amt = 10000
    xs = np.linspace(0, 1, 50)
    evs = [de.ev_fight(x, amt, c) for x in xs]
    assert all(b > a for a, b in zip(evs, evs[1:]))
    assert de.ev_fight(0.0, amt, c) == pytest.approx(-de.cost_to_fight(c))


def test_ev_refund_formula(c):
    assert de.ev_refund(5000, 0.50, c) == pytest.approx(-5000 + 0.0)
    assert de.ev_refund(5000, 0.99, c) == pytest.approx(-5000 + 2400)


def test_uncertainty(c):
    assert de.uncertainty(0.5) == pytest.approx(1.0)
    assert de.uncertainty(0.0) == pytest.approx(0.0)
    assert de.uncertainty(1.0) == pytest.approx(0.0)
    assert de.uncertainty(0.75) == pytest.approx(0.5)


# ------------------------------------------------------------------- decisions
def _df(rows):
    return pd.DataFrame(rows)


def test_decision_picks_max_ev_fight(c):
    # high p_win, large amount -> FIGHT dominates
    df = _df([{"p_win": 0.9, "disputed_amount_inr": 50000, "deadline_dt": 10**9}])
    out = de.decide(df, current_ratio=0.5, c=c, escalation_rate=0.0,
                    hours_budget=100)
    assert out.loc[0, "action"] == de.FIGHT


def test_decision_accept_when_all_ev_nonpositive(c):
    # tiny amount, low p_win, far from ratio threshold -> ACCEPT (EV 0 is best)
    df = _df([{"p_win": 0.1, "disputed_amount_inr": 1000, "deadline_dt": 10**9}])
    out = de.decide(df, current_ratio=0.5, c=c, escalation_rate=0.0,
                    hours_budget=100)
    assert out.loc[0, "action"] == de.ACCEPT


def test_decision_refund_near_threshold_small_amount(c):
    # near/over threshold: ratio_benefit(2400) - amount(2000) = +400 > 0 = accept,
    # and fight EV is negative (recovery < cost_to_fight) -> REFUND
    df = _df([{"p_win": 0.6, "disputed_amount_inr": 2000, "deadline_dt": 10**9}])
    out = de.decide(df, current_ratio=0.95, c=c, escalation_rate=0.0,
                    hours_budget=100)
    assert out.loc[0, "action"] == de.REFUND


# ------------------------------------------------------------------ escalation
def test_escalation_selects_top_n_by_uncertainty_times_amount(c):
    # 10 disputes; escalation_rate 0.2 -> exactly 2 escalated, the highest
    # uncertainty*amount. p_win=0.5 (max uncertainty) with the largest amounts.
    rows = []
    for i in range(10):
        rows.append({"p_win": 0.5 if i < 3 else 0.95,
                     "disputed_amount_inr": 1000 * (i + 1),
                     "deadline_dt": 10**9})
    out = de.decide(_df(rows), current_ratio=0.5, c=c, escalation_rate=0.2,
                    hours_budget=100)
    assert out["escalated"].sum() == 2
    # the two escalated must be among the high-uncertainty (i<3), largest amounts
    esc = out.index[out["escalated"]].tolist()
    assert set(esc) == {1, 2}   # i=2 (amt 3000) and i=1 (amt 2000), both p=0.5


def test_escalation_zero_rate_escalates_nothing(c):
    rows = [{"p_win": 0.5, "disputed_amount_inr": 5000, "deadline_dt": 10**9}
            for _ in range(5)]
    out = de.decide(_df(rows), current_ratio=0.5, c=c, escalation_rate=0.0,
                    hours_budget=100)
    assert out["escalated"].sum() == 0


# -------------------------------------------------------------------- capacity
def test_capacity_limits_fights_and_falls_back(c):
    # 3 attractive FIGHT cases but budget only fits 1 (0.5833 h each; budget 0.6h)
    rows = [{"p_win": 0.95, "disputed_amount_inr": 50000, "deadline_dt": 10**9}
            for _ in range(3)]
    out = de.decide(_df(rows), current_ratio=0.5, c=c, escalation_rate=0.0,
                    hours_budget=de.analyst_hours_per_case(c) + 1e-6)
    assert (out["action"] == de.FIGHT).sum() == 1
    # the other two fall back to next-best EV (ACCEPT here, ratio far from threshold)
    assert (out["action"] == de.ACCEPT).sum() == 2


def test_capacity_deadline_preemption(c):
    # two FIGHT cases, budget for one. The lower-EV case is URGENT (due in 24h)
    # and must pre-empt the higher-EV non-urgent one.
    now = 1_000_000
    rows = [
        {"p_win": 0.99, "disputed_amount_inr": 90000, "deadline_dt": now + 10 * de.SECONDS_PER_DAY},  # high EV, not urgent
        {"p_win": 0.80, "disputed_amount_inr": 40000, "deadline_dt": now + 24 * 3600},                # lower EV, urgent
    ]
    out = de.decide(_df(rows), current_ratio=0.5, c=c, escalation_rate=0.0,
                    hours_budget=de.analyst_hours_per_case(c) + 1e-6, now_dt=now)
    assert out.loc[1, "action"] == de.FIGHT      # urgent one selected
    assert out.loc[0, "action"] != de.FIGHT      # high-EV one bumped to fallback


def test_capacity_ample_budget_keeps_all_fights(c):
    rows = [{"p_win": 0.95, "disputed_amount_inr": 50000, "deadline_dt": 10**9}
            for _ in range(4)]
    out = de.decide(_df(rows), current_ratio=0.5, c=c, escalation_rate=0.0,
                    hours_budget=100)
    assert (out["action"] == de.FIGHT).sum() == 4


# --------------------------------------------------------------- config wiring
def test_load_config_matches_yaml():
    c, rate = de.load_config()
    assert isinstance(c, de.Costs)
    assert 0.0 <= rate <= 1.0
    assert c.representment_fee_inr > 0
