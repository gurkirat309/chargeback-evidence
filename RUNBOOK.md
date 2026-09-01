# RokdaDaav — Runbook

Everything you need to run the project end to end. Windows paths shown; on
macOS/Linux swap `.venv\Scripts\python.exe` for `.venv/bin/python`.

## 0. Prerequisites (one time)

1. **Python 3.13** installed.
2. The two Kaggle IEEE-CIS files placed at:
   - `data/raw/train_transaction.csv`
   - `data/raw/train_identity.csv`
3. A Groq API key in `.env` at the repo root (never committed):
   ```
   GROQ_API_KEY=gsk_xxxxxxxx
   ```
4. Create the environment and install deps:
   ```bash
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

From here on, `python` means `.venv\Scripts\python.exe`.

## 1. Build the pipeline (run in order)

| # | command | what it does | output |
|---|---|---|---|
| 1 | `python src/generate_data.py` | synthesize the dispute layer over IEEE-CIS | `data/processed/*.parquet`, `meta.json` |
| 2 | `python src/verify_data.py` | 8 sanity checks (leakage AUC, reason mix, agent signature, censoring) | console |
| 3 | `python src/train_model.py` | train + calibrate the winnability model; persist `p_win` | `predictions.parquet`, `reports/calibration.png` |
| 4 | `python src/evaluate.py` | the headline: baseline table, rupee confusion, net-recovery curve | console, `reports/net_recovery_curve.png` |
| 5 | `python -m pytest tests/ -q` | decision-engine unit tests (17) | console |

Steps 1→4 are the "honest metrics" story. Runtime is a few seconds each
(train_model runs a weight_scale sweep, so ~30–60s).

## 2. Run the dashboard (the demo UI)

```bash
python app/main.py --warm     # one-time: pre-warm the LLM letter cache
python app/main.py            # serve
```
Open **http://127.0.0.1:8000**. Pick a dispute from the queue; watch the 8-step
evidence agent animate, read the rupee EV math, click a letter claim to jump to
its cited artifact, and drag the **dispute-ratio slider** over 0.90 to watch
winnable cases flip to REFUND.

## 3. Run the MCP server (RokdaDaav as a tool)

```bash
python src/mcp_server.py      # stdio; a host process spawns this
```
Or wire it into any MCP client via the committed `.mcp.json`. In Claude Code it
is auto-discovered in this repo; in Claude Desktop/Cursor, add:
```json
{ "mcpServers": { "rokdadaav": {
  "command": "C:/Fraud/.venv/Scripts/python.exe",
  "args": ["C:/Fraud/src/mcp_server.py"] } } }
```
Tools: `list_disputes`, `score_winnability`, `assemble_evidence`,
`decide_dispute`, `draft_rebuttal`, `evaluate_policy`. Resources:
`rokdadaav://methodology`, `rokdadaav://metrics`.

## 4. (Optional) Live agent demo

A small script where a Groq model, given the tool list, decides on its own which
RokdaDaav tools to call to answer a merchant's plain-English question. See
`docs/PITCH.md` for how it fits the video.

## Reset / regenerate
Everything under `data/processed/`, `data/llm_cache/`, and `reports/` is
regenerable — delete and re-run section 1 (and `--warm`) to rebuild from the seed.
