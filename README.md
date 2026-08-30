# Contra — Chargeback Evidence & Dispute Triage

A chargeback dispute triage and representment system for online merchants. When a
customer disputes a card transaction, contesting it costs money and staff time, so
contesting a case you will lose is negative expected value. Contra gathers evidence,
predicts P(win) with a calibrated model, and decides FIGHT / ACCEPT / REFUND /
ESCALATE using explicit rupee arithmetic.

> Hackathon track: **AI Risk Manager**. Strictly **defense-only** — nothing here
> generates consumer dispute claims. See [`CLAUDE.md`](CLAUDE.md) for full scope.

## Status

**Phase 1 — synthetic dispute generator: complete and verified.** Later phases
(winnability model, decision engine, evidence agent, LLM letter + verifier,
evaluation harness, dashboard) are not built yet; see `CLAUDE.md` §15.

## Two ideas that make this distinct

1. **Sometimes refund a case you would win.** A dispute counts toward the card
   network's monitoring ratio even when you win the representment. Near a program
   threshold, immediate refund can beat recovery. Modelled as a `ratio_benefit` term.
2. **Agent-initiated disputes are their own class.** AI shopping agents buy
   autonomously; the buyer disputes to undo it. These authenticate correctly but look
   anomalous on device/IP — a naive model misreads them as fraud.

## Data

Uses the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
dataset (not redistributed here — gitignored). Place the two training files at:

```
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

590k transactions over a 182-day window; only ~24% carry an identity record. We
generate a synthetic dispute layer on top — there is no public dataset of chargeback
outcomes, and that limitation is stated openly.

## Setup

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# or:  source .venv/bin/activate && pip install -r requirements.txt
```

## Run Phase 1

```bash
./.venv/Scripts/python.exe src/generate_data.py    # writes data/processed/*.parquet
./.venv/Scripts/python.exe src/verify_data.py      # prints the 8 verification checks
```

The generator is driven entirely by `config/generator.yaml` and `config/costs.yaml`
— no magic numbers in the Python. Reproducible from a single seed. Runs in <10s.

## Key design decisions (Phase 1)

- **Subset spans the full window.** We take a 200k evenly-strided subset across all
  182 days (`sampling.mode: span`), not the first 200k, so the temporal split covers
  the real 6 months and right-censoring is a small honest tail rather than ~40%.
- **No deterministic evidence→outcome rule.** Outcome is an additive logit over
  observable evidence + an *unobserved* issuer-leniency term + Gaussian noise, then a
  Bernoulli draw. The unobserved term and noise cap achievable performance on purpose.
- **Per-reason intercept is Jensen-corrected.** Win rate is `sigmoid(logit)`; a wide
  logit spread pulls the mean toward 0.5, so each reason's intercept is solved so the
  *realised* mean win rate matches the §7 target.
- **Split on `filed_dt`, drop right-censored.** In production we decide when a dispute
  is *filed*, so the temporal split is on filing date, not transaction date. Disputes
  filed beyond the 182-day window are dropped, never clipped.
- **`device_match_status` is three-state** (`matched` / `mismatched` / `unknown`).
  `unknown` = no identity record (three quarters of data); a boolean would encode
  missingness rather than signal.

## Leakage acceptance criterion

A plain logistic regression on the evidence features, evaluated on the temporal test
split, must land **AUC in 0.72–0.80**. Above 0.85 means the generator is leaking and
we regenerate with looser weights. Current: **0.777** (PASS). The single tuning knob is
`outcome.weight_scale`.

## Limitations (honest)

- **182 days is short for drift analysis.** The temporal split guards against
  look-ahead leakage; it does **not** demonstrate resistance to long-run drift.
- **Censored labels.** In production we only observe outcomes for cases that were
  fought, so real training data is biased; correcting it needs a randomized holdout.
  A naive `fought` column is included to make this structure discussable.
- **Cost figures are directional.** Vendor-sourced constants in `config/costs.yaml`
  (Chargebacks911) are directional, not audited.

## Layout

```
config/            costs.yaml, generator.yaml   (all constants live here)
src/               generate_data.py, verify_data.py
data/raw/          IEEE-CIS input (gitignored)
data/processed/    generated parquet (gitignored)
CLAUDE.md          project contract — scope, schema, constraints
```
