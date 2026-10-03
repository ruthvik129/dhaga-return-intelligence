# Dhaga & Co. — Return Reason Intelligence

44% of Dhaga & Co.'s returns come back tagged `Other`. The real reason is sitting in a free-text box, in a mix of English, Hindi, and Hinglish, and nobody reads 440 comments a week. This turns that text into structured reasons a merchandiser can act on, routes the rows the models should not decide alone to a human, and reports what it cost and how often it was right.

## Run it

Python 3.12 (pandas 3.x and LangChain 1.x are both pinned against it) and Node 20+ for the front end.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env     # then add your OpenRouter key

cd web && npm install && cd ..
```

**Develop** (two terminals; Vite proxies `/api` to the API):

```bash
.venv/bin/uvicorn api.server:app --reload --port 8000
cd web && npm run dev          # http://localhost:5173
```

**Serve one process** (FastAPI serves the built front end):

```bash
cd web && npm run build && cd ..
.venv/bin/uvicorn api.server:app --port 8000   # http://localhost:8000
```

The dashboard is **upload-first**: it opens empty and every screen is built only from a file you upload (or the bundled sample, if you choose it) and the rows you choose to classify. Each browser session gets its own run (`output/runs/<session>.json`), so one person's upload never replaces another person's dashboard. Evaluation scores the run only if the uploaded file carries the `expected_*` answer-key columns; otherwise it says so.

The UI never names the models and never says "Model A" / "Model B". The cascade is described as a **first pass** and a **second check**, and `api/presentation.py` enforces that on the server, so model identity does not reach the browser at all (`tests/test_api.py` scans every endpoint for it).

To classify from the command line instead:

```bash
D=data/dhaga_returns_intelligence_synthetic_1000.xlsx
.venv/bin/python run_pipeline.py --dataset $D --limit 40       # a cheap slice
.venv/bin/python run_pipeline.py --dataset $D                  # all 440 rows, ~6 min, ~$0.011
.venv/bin/python run_pipeline.py --dataset $D --inject-faults  # exercise the repair path
.venv/bin/python run_pipeline.py --refresh-prices              # re-fetch OpenRouter pricing
.venv/bin/python -m pytest tests/ -q                           # 137 tests, no network
```

To choose models, compare candidates on the same rows (terminal only: the table names models, which the web app never does):

```bash
.venv/bin/python compare_models.py --dataset $D --rows 200 \
    --first-pass mistralai/mistral-nemo openai/gpt-oss-20b \
    --second-check openai/gpt-oss-120b
```

It prints auto-accept accuracy, overall and sub-reason accuracy, how many rows went to a second check or to review, and cost per 1,000 comments for every pairing, recommends the cheapest first pass that clears `--target` (default 95% auto-accept accuracy), and saves the table to `output/comparisons/`.

Responses are cached in SQLite (`output/state.sqlite3`), so re-running the same rows with the same prompts is free.

---

## How it works

```
440 'Other' rows
   │
   ├─ no usable text ─────────────────────────────► human review (no model call)
   │
   └─ Model A, step A: primary reason + confidence + needs_review
         │
         ├─ confidence ≥ 0.85 and not flagged ──► Model A, step B: sub-reason,
         │                                         body area, evidence quote
         │                                              └─► CLASSIFIED_MODEL_A
         │
         └─ flagged, unclear, low confidence, or schema failure
               └─ Model B, step A
                     ├─ agrees with A, confidence ≥ 0.80, not flagged
                     │     └─► Model B, step B ──► CLASSIFIED_MODEL_B
                     └─ disagrees, flags, or is unsure ──► human review
```

Design choices that are deliberate, not incidental:

- **Two calls per row, not one.** Step B is handed the already-decided primary reason and a schema narrowed to exactly that primary's legal sub-reasons, so an illegal pair like `fit` + `late_delivery` is structurally impossible rather than caught after the fact. It also makes a wrong sub-reason attributable to the step that produced it.
- **Two models, cheap-first.** Model A sees every row; Model B sees 5.5% of them. The cascade only pays for itself while that share stays small, which is why the Run Cost screen reports it.
- **Disagreement, not confidence, decides escalated rows.** Self-reported confidence is systematically overconfident, so on escalated rows the router asks whether the two models agree instead of trusting either one's number.
- **The router never calls a model.** Every accept/escalate/human decision is plain Python with configurable thresholds, which is what makes the behaviour testable without a network.
- **Evidence must be verbatim.** The model returns a quote; `code` checks it is a real span of the customer's comment. 440 of 440 passed. An empty quote is allowed — that is the model correctly declining to claim support.
- **Body area is never inferred.** `none` is a legal enum value precisely so the schema does not force a guess. Zero rows claimed a body area the ground truth says was unstated.

### Portability

Model config goes through `init_chat_model("provider:name")`, so switching providers is an `.env` edit:

```
MODEL_A_PROVIDER=openrouter
MODEL_A_NAME=openai/gpt-oss-120b
MODEL_A_MAX_TOKENS=4096
MODEL_B_PROVIDER=openrouter
MODEL_B_NAME=openai/gpt-4.1
```

Measured on a 60-row slice, this pairing scored 91.7% on primary reason against 93.3% for the previous one, at 20x the cost ($0.49 vs $0.025 per 1,000) and roughly 6x the wall clock. 60 rows is too small to call the accuracy difference real; the full-run comparison is the open question. Two facts that matter when choosing: `gpt-oss-120b` is a reasoning model, so it needs `MODEL_A_MAX_TOKENS` around 4096 or it truncates mid-JSON, and Model B is 72% of the bill while seeing 13% of the rows.

OpenRouter is the default because one key fronts 460+ models, which turns the planned eval-and-cost sweep into a config change instead of a credential hunt (`config.openrouter_candidates()` already returns the cheapest-first list of models that support the tool calling and structured output this chain needs). `langchain-anthropic` and `langchain-google-genai` are installed so pointing at a first-party provider needs no dependency change.

**Temperature is unset for both models, on purpose.** The spec asks for explicit temperatures, but several current models reject the parameter outright with a 400. So "unset" means "do not send the field", not "send 0". Set `MODEL_A_TEMPERATURE=0` in `.env` for any model that accepts it; the UI states which setting was actually in force for the run on screen.

---

## Screens

| Screen | What it answers |
| --- | --- |
| **Overview** | How many returns, how many `Other`, how many auto-classified vs queued. Input validation report. |
| **Upload & Process** | The starting point. Upload a file (or choose the sample), check its validation report, pick how many `Other` rows to classify, and watch live progress. Every other screen stays empty until this has run. |
| **Insights** | What `Other` actually was. Sub-reason and body-area breakdowns, share-within-group with lift and CIs, and every aggregate drills down to the verbatim comments behind it. CSV export. |
| **Review Queue** | The flagged rows, each with its full escalation trace. Confirm, correct, or mark unclear; the charts update immediately. |
| **Evaluation** | Accuracy with denominators and Wilson intervals, the confusion matrix, the specific confusions the spec asks about, the rules baseline beside the cascade, and the list of misses to read. |
| **Run Cost** | Measured tokens and spend per stage, cost per 1,000 comments, what escalation cost, and a projection including reviewer hours. |

Numbers are framed as **share within a group against the overall share, with lift** — never as a "return rate". The dataset has no units-sold denominator, so a rate would be an invented statistic. This also stops a high-volume vendor from looking like a quality problem purely because it ships more.

---

## Layout

```
api/server.py              JSON API for the front end — no model calls
api/presentation.py        strips model identity before anything reaches the browser
web/                       React + Vite + TypeScript front end, six screens
design-system/             ui-ux-pro-max design tokens the front end follows
run_pipeline.py            CLI: load → classify → route → persist
src/schemas.py             taxonomy + pydantic/JSON-Schema contracts
src/prompts.py             prompt text, generated from the taxonomy
src/models.py              THE ONLY MODULE THAT CALLS AN LLM
src/router.py              accept / escalate / human — pure code
src/baseline_rules.py      keyword control arm, no network
src/data_io.py             .xlsx loading, validation, SQLite state
src/metrics.py             aggregation, evaluation, cost
src/config.py              models, thresholds, OpenRouter price snapshot — all env-overridable
tests/                     79 tests, none of which touch the network
output/run_latest.json     the committed reference run (the numbers above; API tests use it)
docs/                      requirements, design record, engineering review
```
