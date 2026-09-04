# RokdaDaav — 5-minute video script (click-by-click)

**Format below:** 🎙️ = say this · 🖱️ = do this on screen. Times are cumulative.

---

## Before you hit record (2-min setup)
1. Terminal: `cd C:\Fraud` → `python app/main.py` (server running).
2. Browser tab **A**: http://127.0.0.1:8000 → **Ctrl+Shift+R** (hard refresh).
   - Set the **dispute-ratio slider back to 0.50** (left = safe zone).
   - Scroll to the very top.
3. Browser tab **B**: your Razorpay **test** dashboard → **Transactions → Orders** (for the "real order" proof).
4. Terminal tab **2** (optional): ready to run `python src/evaluate.py`.
5. Close other tabs. Nothing credential-shaped on screen (the key is already masked).

---

## 0:00 – 0:30 · The problem + the name
🖱️ *Dashboard top (tab A), showing the RokdaDaav header + the scoreboard chips.*

🎙️ "When a customer disputes a card payment, a merchant can fight it or eat the
loss. Fighting costs a fee plus staff time — so fighting a case you'll lose
actually loses *more* money. Most tools fight everything, or nothing.
This is **RokdaDaav** — *fight only when it pays.* It decides, in rupees, on
held-out data, and it plugs straight into Razorpay."

🖱️ *Point the cursor at the green chip.* 

🎙️ "That green number is our policy beating every simple baseline — I'll prove
it at the end."

## 0:30 – 1:00 · The two ideas that make it different
🎙️ "Two ideas most people miss. **One** — sometimes you should *refund* a case
you'd *win*, because a dispute counts against your Razorpay/network ratio even if
you win, and crossing the fine threshold costs more than the recovery.
**Two** — AI-shopping-agent disputes are their own class: they authenticate fine
but look anomalous, and a naive model wrongly calls them fraud. We handle both."

## 1:00 – 1:55 · LIVE: it runs itself, wired to Razorpay  ← the wow
🖱️ *Top-right panel: "⚡ Live triage — Razorpay dispute webhook." Point at the green
"Connected to your Razorpay account (test mode)" badge.*

🎙️ "This is connected to my real Razorpay test account. When a dispute hits the
Razorpay webhook, RokdaDaav triages it automatically — no human."

🖱️ *Click **"▸ Simulate incoming Razorpay disputes."** Let the rows stream in.*

🎙️ "Each one is scored, priced in rupees, and routed on its own — fight, accept,
refund, or escalate to a human. Fight cases come with a drafted letter and land
in an approve queue."

🖱️ *In the "Approve & send queue," click **"Approve & submit"** on a FIGHT row.
Wait for "✓ recorded in Razorpay: order_…".*

🎙️ "Approve, and it writes a real record to Razorpay."

🖱️ *Switch to tab B (Razorpay → Orders), refresh, point at the new order tagged
`source: RokdaDaav`.*

🎙️ "There it is in my actual Razorpay dashboard. The connection and the write
are real; the dispute events are simulated because test mode doesn't emit
disputes on demand — and we say that openly."

## 1:55 – 2:25 · Idea #1 live: the refund move
🖱️ *Back to tab A. Drag the **dispute-ratio slider** from 0.50 up past 0.90 (turns red:
"over VAMP threshold"). Re-run "Simulate incoming Razorpay disputes."*

🎙️ "Now I push the merchant's dispute ratio into the danger zone. Watch —
winnable cases start flipping to **REFUND**, automatically, to dodge the network
fine. That's the counter-intuitive risk move, made for you, with the rupee
trade-off shown."

🖱️ *Drag the slider back to 0.50.*

## 2:25 – 3:15 · One dispute, end to end
🖱️ *Left queue: click a FIGHT case (e.g. an `inr` one at ~89%). Scroll the right
panel slowly.*

🎙️ "Open any dispute. The win probability is **calibrated** — on our test set the
error is about five percent — so it's safe to multiply by rupees."

🖱️ *Point at the Decision panel.*

🎙️ "Here's the actual arithmetic: expected value of fighting equals win
probability times the amount, minus the cost to fight. The winning option is
highlighted."

🖱️ *Scroll to the evidence agent; let a step or two show.*

🎙️ "Then a fixed eight-step evidence agent assembles the case…"

🖱️ *Scroll to the cream **letter** document.*

🎙️ "…and this is the rebuttal letter it would submit — a real document, every
point citing a specific piece of evidence."

🖱️ *Scroll to the red ✗ stripped claim in the audit trail.*

🎙️ "And this red line matters most: the AI wrote a claim it couldn't back up, and
a second AI **deleted it**. RokdaDaav can't fabricate — the letter only keeps what
the evidence supports."

## 3:15 – 3:45 · Ask RokdaDaav (the agent)
🖱️ *Scroll to the "Ask RokdaDaav" panel. Type: `Should I fight DSP001561?` → Ask.*

🎙️ "You can also just ask it. A live AI agent decides which of RokdaDaav's tools
to call…"

🖱️ *Point at the `🔧 called decide_dispute(...)` line, then the answer.*

🎙️ "…it calls the decision tool itself and comes back: fight, plus twenty-seven
thousand rupees expected value. Same tools are exposed over MCP, so any assistant
— or Razorpay's own systems — can consult it."

## 3:45 – 4:35 · The honest metrics (the judging bar)
🖱️ *Point at the scoreboard chips at the top. (Optional: switch to terminal tab 2,
run `python src/evaluate.py`, show the table.)*

🎙️ "Here's the part the track actually grades — honest metrics on a **temporal
held-out** test set: trained on the past, tested on the future. RokdaDaav
recovers about **twenty-four-point-four lakh rupees**, beating the best simple
rule — 'fight if amount over two thousand' — by **twenty-and-a-half percent**.
And it beats the best single probability threshold too, because it ranks on
value — probability *times* amount — under a real staffing budget."

🎙️ "Crucially, we price being wrong: a fought-and-lost case costs the fee plus
analyst time, and that false-positive cost is in the numbers — not hidden."

## 4:35 – 5:00 · Honesty + defense-only + close
🖱️ *Back to the dashboard, calm shot of the full screen.*

🎙️ "We're honest about the limits: there's no public chargeback-outcome data, so
the dispute layer is synthetic and we say so; the window is six months, so we
don't overclaim drift. And it's strictly defense-only — nothing here generates
disputes, and the letter can only cite real evidence.
**RokdaDaav — fight only when it pays.** Thank you."

---

## Timing cheat-sheet
| segment | ends at | screen |
|---|---|---|
| problem + name | 0:30 | dashboard top |
| two ideas | 1:00 | dashboard |
| live triage + real Razorpay | 1:55 | Live-triage panel + Razorpay Orders tab |
| refund via slider | 2:25 | ratio slider |
| one dispute end-to-end | 3:15 | dispute detail → letter |
| Ask agent | 3:45 | Ask panel |
| honest metrics | 4:35 | scoreboard / evaluate.py |
| honesty + close | 5:00 | full dashboard |

## If you're running long, cut in this order
1. Drop the `evaluate.py` terminal shot (just point at the chips) — saves ~15s.
2. Shorten "two ideas" to one line — saves ~15s.
3. Skip switching to the Razorpay Orders tab (the "✓ recorded" line is enough).
