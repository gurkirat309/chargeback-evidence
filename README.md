# Contra — Chargeback Evidence & Dispute Triage

A chargeback dispute triage and representment system for online merchants. When a
customer disputes a card transaction, contesting it costs money and staff time, so
contesting a case you will lose is negative expected value. Contra gathers evidence,
predicts P(win) with a calibrated model, and decides FIGHT / ACCEPT / REFUND /
ESCALATE using explicit rupee arithmetic.

> Hackathon track: **AI Risk Manager**. Strictly **defense-only** — nothing here
> generates consumer dispute claims. See [`CLAUDE.md`](CLAUDE.md) for full scope.

## Status

Phases 1–4 complete: **synthetic dispute generator**, **winnability model**
(calibrated logistic regression), **decision engine** (pure EV arithmetic, unit
tested), and the **evidence agent + LLM letter/verifier** (Groq). Remaining:
evaluation harness (§12) and dashboard UI (§15 steps 5–6).

> **LLM provider note.** CLAUDE.md §11 specifies `claude-sonnet-4-6` via the
> Anthropic API. Per the maintainer's direction this build uses **Groq**
> (`openai/gpt-oss-120b` generator, `openai/gpt-oss-20b` verifier; key in `.env`
> as `GROQ_API_KEY`, never committed). Everything else in §11 is preserved: two
> calls, an independent verifier, and every response cached to `data/llm_cache/`
> keyed by an input hash — so once warmed, the demo needs no live call.

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
- **Reason-conditional filing lag.** Time-to-file is a real dispute signal, so the lag
  is drawn per reason code: fraud is noticed on the statement (~18d), item-not-received
  waits out the delivery window (~28d), subscription is caught only after a billing
  cycle (~32d, long tail), agent-initiated is a fast undo (~8d).
- **Split on `filed_dt`, drop right-censored.** In production we decide when a dispute
  is *filed*, so the temporal split is on filing date, not transaction date. A dispute
  is observed only if filed before `observation_end = last-transaction day +
  observation_buffer_days`; later filings are dropped, never clipped. The buffer
  (14 days) models the real gap between the last transaction and the data-extract date
  — without it, censoring maxes out (~13%) by assuming extract at the instant of the
  final transaction. At 14 days censoring is ~5%, and because slow-filing reasons
  censor more, the *observed* reason mix drifts slightly toward fast-filing reasons
  (fraud/agent) relative to the generated mix — a realistic censoring bias, reported by
  the verifier.
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
- **Agent-initiated share is set for evaluation power, not realism.** It is 15% of
  disputes (~300 cases, ~90 in the test split) so the differentiating class has enough
  test mass for segment metrics with usable confidence intervals; a realistic share
  would be low single digits. Because agent cases require an *observed* (mismatched)
  device, they need an identity record, which caps how strongly no-identity
  transactions can be over-represented elsewhere.
- **`consent_record_exists` is general evidence, not subscription-only.** It is present
  at ~58–79% across all reason codes (verifier check 4b), so it is not a reason-code
  proxy; its correlation with `won` reflects its outcome weight. It carries the most
  weight for subscription cases.
- **Censored labels.** In production we only observe outcomes for cases that were
  fought, so real training data is biased; correcting it needs a randomized holdout.
  A naive `fought` column is included to make this structure discussable.
- **Cost figures are directional.** Vendor-sourced constants in `config/costs.yaml`
  (Chargebacks911) are directional, not audited.

## LLM layer (Phase 4)

Three stages, all deterministic-by-cache:

- **Evidence agent** (`src/evidence_agent.py`) — a fixed 8-step lookup sequence
  (deliberately *not* agentic) that assembles a numbered bundle; every artifact
  carries an `artifact_id` resolving to a source record. Only artifacts that
  actually exist are emitted, so the generator can never cite an absent one.
- **Generator** (`src/llm_generator.py`) — sees only the bundle + reason code,
  returns `{framing, claims:[{claim, artifact_id}]}`. No `p_win`, no decision.
- **Verifier** (`src/llm_verifier.py`) — a separate, smaller model; for each
  claim it sees the claim and its cited artifact and returns
  `{supported, reason}`. Unsupported claims are stripped before assembly — the
  letter gets shorter, never invented.

Run the demo/metrics: `python src/llm_demo.py --n 12`. Measured on a 12-dispute
test sample: **citation coverage 100%**, verifier **strips ~8%** of generated
claims as unsupported, **hallucination-under-stress 0%** (delete a cited artifact,
regenerate — the deleted fact is not re-asserted), and an adversarial probe
(fabricated claims that contradict their artifact) is **caught 5/5**.

## Layout

```
config/            costs.yaml, generator.yaml   (all constants live here)
src/               generate_data.py, verify_data.py
data/raw/          IEEE-CIS input (gitignored)
data/processed/    generated parquet (gitignored)
CLAUDE.md          project contract — scope, schema, constraints
```
