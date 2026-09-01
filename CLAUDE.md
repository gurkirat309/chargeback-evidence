# CLAUDE.md — Project RokdaDaav

Read this file fully before writing any code. It defines scope, schema, and
constraints. When a request conflicts with this file, say so instead of
silently deviating.

---

## 1. What we are building

A chargeback dispute triage and response system for online merchants.

When a customer disputes a card transaction, the merchant can either contest it
(a "representment") or accept the loss. Contesting costs money and staff time,
so contesting a case you will lose is negative expected value. This system:

1. Gathers evidence for the disputed transaction from multiple sources
2. Predicts P(win) with a calibrated ML model
3. Decides FIGHT / ACCEPT / REFUND / ESCALATE using explicit rupee arithmetic
4. Generates a rebuttal letter where every factual claim cites a real artifact
5. Verifies that letter with a second LLM pass that strips unsupported claims

Submission context: a hackathon track called "AI Risk Manager". The judging bar
is **honest metrics including false-positive cost, on a held-out test set**, and
the project is **strictly defense-only**. Deliverable is a 5-minute video.

---

## 2. The two ideas that make this project distinct

Preserve both. If a change would weaken either, flag it.

**A. Sometimes refund a case you would win.**
Card networks track a merchant's dispute ratio (disputes / transactions).
Crossing a program threshold triggers fines and monitoring. Winning a
representment recovers the money but the dispute **still counts** toward the
ratio. So a merchant near the threshold should sometimes refund immediately —
even on winnable cases — because the ratio relief exceeds the recovery. This is
modelled in the decision engine via a `ratio_benefit` term.

**B. Agent-initiated disputes are a distinct class.**
AI shopping agents now purchase autonomously. The buyer does not see the
purchase until after payment, so disputing is the fastest way to undo it. These
cases have an unusual evidence signature — device fingerprint does not match the
customer's history, IP is a datacenter, no human browsing session — but the
payment authenticated correctly and the mandate was valid. A naive model reads
this as fraud and predicts a loss. We model it as its own reason-code family
with its own evidence strategy.

---

## 3. Scope

### Building
- Synthetic dispute layer over the IEEE-CIS transaction dataset
- Winnability model (gradient boosting, calibrated)
- Decision engine (pure arithmetic, no ML, no LLM)
- Evidence retriever ("evidence agent") — fixed 8-step sequence
- Letter generator (LLM) + claim verifier (second LLM)
- Evaluation harness with rupee-denominated cost curves
- Dashboard UI for the demo

### Not building — do not add these
- Real payment gateway or courier integrations
- Authentication, user accounts, multi-tenancy
- Real-time transaction fraud scoring at checkout
- Agent frameworks (LangChain, AutoGPT, CrewAI, tool-calling loops)
- Anything that generates plausible consumer dispute claims — the track
  disqualifies offense-capable work
- Docker, CI, cloud deploy, tests beyond the decision engine

### Deliberately not agentic
The evidence retriever is a fixed sequence of lookups, not a model choosing its
own control flow. This is intentional: decisions here move money and touch
network compliance, so they must be reproducible, auditable, and
regression-testable. Do not introduce a reasoning loop over source selection.

---

## 4. Data

Source: IEEE-CIS Fraud Detection (Kaggle). Local files:

```
data/raw/train_transaction.csv     # ~590k rows, 394 cols, has isFraud label
data/raw/train_identity.csv        # device/identity, join on TransactionID
```

Only ~24% of transactions have an identity record. Plan for nulls.

`TransactionDT` is **seconds from an undisclosed reference point**, not a
timestamp. It can be used for ordering and gaps. It cannot yield calendar dates,
day-of-week, or holidays. Do not attempt to derive them.

Columns we actually use (ignore V*, C*, D*, M* except where noted):

```
TransactionID, TransactionDT, TransactionAmt, ProductCD,
card1, card2, card3, card4, card5, card6,
addr1, addr2, P_emaildomain, isFraud
DeviceType, DeviceInfo, id_30, id_31, id_33   (from identity)

# R_emaildomain dropped: 76.75% null, not worth imputing (see inspection).
```

We generate the dispute layer on top. There is no public dataset of chargeback
outcomes; this limitation is stated openly in the README and in the video.

---

## 5. Schema

### disputes.csv
```
dispute_id            str
TransactionID         int    FK to IEEE-CIS
filed_dt              int    TransactionDT + lag, in seconds
days_txn_to_dispute   int
reason_code           str    fraud | inr | nad | subscription | agent_initiated
disputed_amount_inr   float
deadline_dt           int
agent_initiated       bool
```

### evidence.csv  (one row per dispute, boolean/typed artifact presence)
```
dispute_id                  str
avs_match                   bool
cvv_match                   bool
three_ds_status             str    passed | attempted | none
device_match_status         str    matched | mismatched | unknown
prior_txn_count             int
prior_undisputed_count      int
account_age_days            int
delivery_proof_type         str    none | tracking | signature | otp | photo
delivery_otp_verified       bool
ip_to_ship_km               float
ip_is_datacenter            bool
customer_contacted_support  bool
support_ticket_count        int
product_photos_on_file      bool
refund_policy_shown         bool
consent_record_exists       bool   (subscription cases)
agent_mandate_on_file       bool   (agent cases)
```

`device_match_status` is three-state, not boolean: only ~24% of transactions
have an identity record, so `unknown` (no identity record) is the default for
three quarters of the data and a boolean would encode missingness, not signal.
`matched` / `mismatched` require an identity record.

### outcomes.csv
```
dispute_id   str
p_win_true   float   ground-truth probability used by the generator
won          int     0/1, sampled from p_win_true
```

`p_win_true` is for diagnostics only. **Never expose it as a model feature.**

---

## 6. Outcome generation — critical

The generator must NOT contain a deterministic rule that maps evidence to
outcome. If it does, the model will trivially recover it, score 0.99 AUC, and
the whole ML story is worthless.

Required shape:

```python
logit  = BASE_LOGIT[reason_code]
logit += WEIGHTS[feature] * value   # for each evidence feature
logit += issuer_leniency[card_issuer_group]   # NOT exposed as a feature
logit += np.random.normal(0, 0.7)             # irreducible noise
p_win  = sigmoid(logit)
won    = np.random.binomial(1, p_win)
```

Two deliberate sources of honest uncertainty:
- `issuer_leniency` is a real effect the model cannot observe
- the noise term caps achievable performance

**Acceptance criterion: test AUC must land in 0.72–0.80.**
Above 0.85 means the generator is leaking — loosen weights, raise noise, and
regenerate. Report this check explicitly.

Evidence presence must also correlate with the underlying `isFraud` label so
evidence carries genuine signal rather than noise. Genuinely fraudulent
transactions should less often have `device_match_status == 'matched'`, long
account age, and prior undisputed transactions.

---

## 7. Reason code mix

| code | share | base win rate | primary evidence |
|---|---|---|---|
| `fraud` | 35% | ~25% | device match, 3DS, prior txns |
| `inr` (item not received) | 25% | ~60% | delivery proof, OTP |
| `nad` (not as described) | 20% | ~35% | photos, policy, support log |
| `subscription` | 12% | ~45% | consent record |
| `agent_initiated` | 8% | ~40% | agent mandate, auth scope |

Agent-initiated cases must be generated with the distinctive signature. Sample
them **only from transactions that have an identity record**, so the anomaly is
an *observed* device that isn't the customer's, not missing device data:
`device_match_status='mismatched'`, `ip_is_datacenter=True`,
`customer_contacted_support=False`, but `three_ds_status='passed'` and
`agent_mandate_on_file=True`. The distinction between `mismatched` and `unknown`
is the whole point of the class — a naive model must not be able to separate
agent cases from the missing-identity majority on missingness alone.

Overall dispute rate: 1.0% of transactions. Target ~2,000 disputes.

---

## 8. Cost model

All money constants live in `config/costs.yaml`. Nothing is hardcoded
elsewhere. Every rupee figure shown in the UI must trace back to this file.

```yaml
usd_to_inr: 88.0
representment_fee_inr: 2500
analyst_minutes_per_case: 35
analyst_cost_per_hour_inr: 400
analyst_hours_per_day: 6

vamp_threshold_pct: 0.90
vamp_warning_band_pct: 0.15
vamp_fine_per_dispute_inr: 800

sources:
  dispute_rate_cnp: "0.6-1.0% — Chargebacks911 (vendor, directional)"
  chargeback_fee: "$20-100 — Chargebacks911 (vendor, directional)"
  avg_dispute_amount: "$76 — Chargebacks911 (vendor, directional)"
  analyst_time: "team assumption, not sourced"
  win_rates: "team assumption, not sourced"
```

Vendor-sourced figures are directional. Say so in the README.

---

## 9. Decision engine

Pure functions. No ML, no LLM, no randomness. Unit tested.

```
cost_to_fight = representment_fee_inr
              + (analyst_minutes_per_case / 60) * analyst_cost_per_hour_inr

EV_fight  = p_win * disputed_amount_inr - cost_to_fight
EV_accept = 0
EV_refund = -disputed_amount_inr + ratio_benefit(current_ratio)
```

`ratio_benefit` — piecewise, monotonic in `current_ratio`:

```
if ratio < threshold - warning_band:          0
elif ratio < threshold:                        fine_per_dispute * scale
                                               where scale ramps 0 -> 1
                                               across the warning band
else:                                          fine_per_dispute * 3
```

Decision: pick the highest EV. Then route the **top N% of cases ranked by
`uncertainty * disputed_amount_inr`** to `ESCALATE` instead — uncertain and
expensive cases go to a human with evidence pre-assembled. Here
`uncertainty = 1 - 2*|p_win - 0.5|` (peaks at p_win=0.5, zero at the extremes)
and `N` comes from config as `escalation_rate_target: 0.10`.

This replaces the earlier absolute `disputed_amount_inr > 10000` cutoff:
disputes over-sample high-value transactions, so a fixed rupee line fires on
most of the distribution (q75 ≈ ₹11,000) rather than a rare tail. Ranking by
capacity mirrors how a real risk team works — they escalate what they can
staff, not what crosses an arbitrary line.

Then apply the daily analyst capacity constraint: sort FIGHT cases by
EV-per-analyst-hour, take them greedily until the hour budget is exhausted,
with deadline pre-emption for cases expiring within 48h. Cases that do not fit
fall back to their next-best EV option.

---

## 10. Evidence agent

Fixed 8-step sequence, executed in order, each step logged with a timestamp so
the UI can animate it:

```
1. Order record            5. Device & IP match
2. Payment / 3DS check     6. Past disputes on instrument
3. Customer history        7. Support conversations
4. Delivery record         8. Bundle assembly
```

Output: a structured numbered bundle. Every item carries an `artifact_id` that
resolves to a source record, so the UI can make each claim clickable.

---

## 11. LLM layer

Two calls, both to `claude-sonnet-4-6` via the Anthropic API. **Cache every
response to `data/llm_cache/` keyed by a hash of the input.** The demo must
never depend on a live API call.

**Generator** — receives only the assembled evidence bundle and the reason code.
Returns structured JSON: a list of `{claim, artifact_id}` plus a short framing
paragraph. It does not receive `p_win`, does not decide anything, and cannot see
artifacts that are absent.

**Verifier** — a separate call. For each claim, receives the claim and its cited
artifact, returns `{supported: bool, reason: str}`. Unsupported claims are
stripped before the packet is assembled.

Metrics to report: **citation coverage %** (claims with a resolving artifact)
and **hallucination rate under stress** (delete a random artifact, regenerate,
measure how often the letter asserts the deleted fact anyway).

---

## 12. Evaluation harness

First-class deliverable, not an afterthought. If time runs short, ship a worse
model with rigorous metrics rather than a better model with vague ones.

- **Temporal split — on `filed_dt`, not `TransactionDT`.** Compute
  `filed_dt = TransactionDT + lag_seconds`. Sort disputes by `filed_dt`, train on
  the first 70%, test on the last 30%. Never a random split. In production we
  decide about a dispute *when it is filed*, so a model trained on the past must
  be trained on past filings; splitting on transaction date would drop a day-100
  transaction filed on day 170 into training alongside earlier-filed test
  disputes — that is leakage.
- **Drop right-censored disputes.** Any dispute whose `filed_dt` lands beyond
  the 182-day data window is dropped, not clipped — clipping would pile mass at
  the boundary and distort the tail of the lag distribution. The verifier prints
  both the `TransactionDT` and `filed_dt` split boundaries and the count dropped.
- **182 days is short for drift analysis.** The window is ~6 months, so the
  README must not overclaim temporal robustness — the split guards against
  look-ahead leakage, it does not demonstrate resistance to long-run drift.
- Precision / recall / F1 on the FIGHT decision
- Calibration: reliability diagram, Brier score, expected calibration error
- Segmented metrics by reason code and by amount band
- Rupee-denominated confusion matrix:
  - FP (fought, lost) = representment fee + analyst cost
  - FN (accepted, would have won) = full recoverable amount
- Threshold sweep with net-recovery curve, optimum marked
- **Baseline comparison** — this is the headline table:

| policy | net recovery ₹ |
|---|---|
| fight everything | |
| fight nothing | |
| fight if amount > ₹2,000 | |
| fight if p_win > 0.5 | |
| **RokdaDaav (EV + ratio + capacity)** | |

If RokdaDaav does not beat "fight if amount > ₹2,000", there is no product. Report
it honestly either way.

**Censored labels.** We only observe outcomes for cases that were fought. State
this limitation explicitly in the README and in the video: in production the
training data is biased, and correcting it requires a randomized holdout where
some low-scoring cases are fought anyway.

---

## 13. Tech stack

```
Python 3.13
pandas 3.x, numpy, scikit-learn, lightgbm
FastAPI + uvicorn
SQLite (or parquet on disk — no server)
anthropic
matplotlib for evaluation plots
Frontend: single-page React via CDN, or plain HTML + Alpine. No build step.
```

Do not add a framework not listed here without asking.

---

## 14. Repo layout

```
config/costs.yaml
config/generator.yaml
data/raw/                 # IEEE-CIS, gitignored
data/processed/
data/llm_cache/
src/generate_data.py
src/verify_data.py
src/features.py
src/train_model.py
src/decision_engine.py
src/evidence_agent.py
src/llm_generator.py
src/llm_verifier.py
src/evaluate.py
tests/test_decision_engine.py
app/main.py
app/static/
reports/                  # generated plots and tables
README.md
```

---

## 15. Build phases

Do not start a phase before the previous one is verified.

1. **Data** — generator + `verify_data.py`. Stop and inspect output.
2. **Model** — train, calibrate, temporal holdout. Check AUC is 0.72–0.80.
3. **Decision engine** — pure functions + unit tests.
4. **Evidence agent + LLM layer** — with caching.
5. **Evaluation harness** — baselines, curves, segment tables.
6. **UI** — last, built to serve the video.

---

## 16. Working style

- Prefer boring, readable code. This is read by judges, not scaled.
- No silent fallbacks. If evidence is missing, the letter gets shorter — never
  invented. Same principle applies in code: fail loudly.
- Every magic number goes in a config file.
- Print diagnostics liberally. We need to see distributions, not trust them.
- When something in this file appears wrong or infeasible, say so directly
  rather than working around it quietly.
