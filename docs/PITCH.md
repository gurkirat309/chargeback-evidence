# RokdaDaav — 5-minute video pitch

**Track 02 — AI Risk Manager. Tagline: "Fight Only When It Pays."**

Before recording: run `python app/main.py --warm` once, then
`python app/main.py`, and have **http://127.0.0.1:8000** open. Keep a terminal
ready for the agent demo (section 4).

---

### 0:00–0:35 · The problem (hook)
- Chargebacks and returns quietly eat merchant margin. When a customer disputes a
  card charge, the merchant can **contest (representment)** or **eat the loss**.
- Contesting costs money and analyst time. **Fighting a case you'll lose is
  negative expected value.** Most tools fight everything or fight nothing.
- "RokdaDaav fights only when it pays — and proves it in rupees on held-out data."

### 0:35–1:20 · Two non-obvious insights (what makes it different)
1. **Sometimes refund a case you'd win.** A dispute counts toward the card
   network's monitoring ratio even if you win the representment. Near the
   threshold, an immediate refund (ratio relief) beats the recovery.
2. **Agent-initiated disputes are their own class.** AI shopping agents buy
   autonomously; the buyer disputes to undo it. They authenticate fine but look
   anomalous on device/IP — a naive model misreads them as fraud. We model them
   as their own reason code.

### 1:20–2:20 · Live: one dispute, end to end (dashboard)
Click a FIGHT case (e.g. an `inr` dispute, p_win ~0.89):
- **Calibrated p(win)** — "89% isn't a vibe; test ECE is 0.056, so it's safe to
  multiply by rupees."
- **Decision — EV arithmetic**: `EV_fight = p_win·amount − cost_to_fight`, the
  winning row highlighted. Every number traces to `config/costs.yaml`.
- **Evidence agent** — the fixed 8-step trace animates; each artifact has an id.
- **Rebuttal letter** — click a claim; it jumps to the exact artifact it cites.
  "100% citation coverage; a second model strips any unsupported claim."

### 2:20–3:00 · Live: insight #1, the ratio slider
- Drag **Merchant dispute-ratio** from 0.50 toward 1.0. As it crosses the VAMP
  threshold, watch cases flip to **REFUND** and net recovery tick up.
- "That's the counter-intuitive move — refunding a winnable case to avoid fines —
  made automatically, with the rupee tradeoff shown."

### 3:00–3:50 · The honest metrics (the judging bar)
Show `python src/evaluate.py` output (or the header chips):

| policy | net recovery ₹ |
|---|---:|
| fight everything | 1,972,116 |
| fight if amount > ₹2,000 | 2,028,095 |
| fight if p_win > 0.5 | 1,815,433 |
| **RokdaDaav** | **2,444,442** |

- "**+20.5%** over the strongest simple baseline, on a **temporal held-out test
  set** — trained on the past, tested on the future."
- Precision/recall/F1 on the FIGHT decision, and the **rupee confusion matrix**:
  false-positives (fought & lost) cost fee + analyst time; we price them.
- Show `reports/net_recovery_curve.png`: no single p_win threshold matches us,
  because we rank on **p_win × amount** under an analyst-hour budget.

### 3:50–4:35 · RokdaDaav as an MCP tool (the agent angle)
Run the agent demo in the terminal:
- A merchant asks in plain English: *"Customer disputed DSP001561 — should I
  fight it?"* A live model **chooses** to call `decide_dispute` → answers
  "FIGHT, +₹27,730." Ask about a weak case → it calls `score_winnability` →
  "13%, don't fight."
- "The AI Risk Manager isn't guessing — it stands on a deterministic, tested
  engine exposed over MCP. Any assistant can consult it."

### 4:35–5:00 · Honesty + close
- **Limitations, stated openly**: synthetic dispute layer (no public chargeback-
  outcome data); 182-day window is short for drift; **censored labels** (in
  production you only see outcomes for fought cases); cost constants are
  vendor-directional.
- **Strictly defense-only**: nothing generates dispute claims; the letter can
  only cite real, verified evidence.
- Close on the tagline: **"RokdaDaav — Fight Only When It Pays."**

---

## What to have on screen when
| beat | screen |
|---|---|
| 0:00–1:20 | title slide / tagline |
| 1:20–3:00 | dashboard at :8000 |
| 3:00–3:50 | evaluate.py output + net_recovery_curve.png + calibration.png |
| 3:50–4:35 | terminal agent demo |
| 4:35–5:00 | limitations slide |
