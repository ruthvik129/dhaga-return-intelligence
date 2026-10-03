# TODOS

## Return Intelligence

### Dropdown-versus-text agreement check on the 217 text-bearing structured rows

**What:** Run the existing classifier over the 217 non-`Other` rows that also carry free text, and compare the model's reading of the text against the dropdown reason the customer selected.

**Why:** It tests whether the 560 "structured" rows are actually reliable. If a customer picked `Changed Mind` but wrote `chest tight hai`, the structured data is wrong too, which would mean the real no-useful-reason problem is larger than the stated 44% rather than confined to the `Other` bucket. It is the most interesting unexplored finding in the dataset and needs no new infrastructure once the pipeline exists.

**Context:** Found by parsing `dhaga_returns_intelligence_synthetic_1000.xlsx` directly during /office-hours on 2026-10-02: of the 560 rows with a structured dropdown reason, 217 also have non-empty `return_reason_text`. Deliberately deferred under premise P1 to protect MVP scope and to keep the trust-spine demo from diluting into a second storyline. Starting point is `src/data_io.py`, which already loads every row, so the filter is one line: `dropdown != "Other" AND return_reason_text.strip()`. The comparison needs a dropdown-to-taxonomy mapping (e.g. `Size/Fit` to `fit`, `Quality/Fabric` to `quality_issue`), which does not exist yet and is the only real new work. No ground truth is needed, because a disagreement is self-evident.

Important caveat to carry forward: a disagreement rate is **not** proof the dropdown is wrong. A customer can pick the dropdown accurately and then write about a second, different issue. Report it as "rows where the free text describes a reason the dropdown does not capture", never as a dropdown error rate.

**Effort:** M
**Priority:** P3
**Depends on:** The MVP classification pipeline existing and its accuracy being trusted. Strictly after the MVP ships. Decided at /plan-eng-review D9 on 2026-10-02.

### Eval and cost matrix across OpenRouter candidates

**What:** Sweep Model A / Model B pairings over the cheapest-first OpenRouter candidates and pick the final pairing from measurements rather than from a guess.

**Why:** The current pairing (`mistralai/mistral-nemo` + `openai/gpt-oss-120b`) was chosen to make the pipeline measurable at near-zero cost, not because it is the right pairing. Model A is overconfident — it auto-accepted 94.5% of rows — so the number that decides this is auto-accept precision per candidate, not headline accuracy.

**Context:** The infrastructure exists. `config.openrouter_candidates()` returns cheapest-first models that support the tool calling and structured output this chain needs; `--refresh-prices` snapshots live per-token rates; `metrics.evaluate()` already reports auto-accept precision with a Wilson interval; the SQLite response cache keeps repeat runs free. The sweep itself is a loop over `ModelSpec` pairs calling `run_pipeline.run(limit=...)` and tabulating accuracy against cost. Use a fixed row subset so candidates are compared on identical input.

**Effort:** M
**Priority:** P2
**Depends on:** Nothing. The user asked for this explicitly as the step after the MVP works.

### Deploy to a public URL

**What:** Host the web app (FastAPI serving the built React front end) somewhere reachable (spec section 25).

**Why:** It is an open acceptance criterion.

**Context:** The dashboard is upload-first, so a deploy needs an OpenRouter key before anyone sees a chart. Three caveats to handle: on an ephemeral filesystem the per-session runs (`output/runs/`), uploads, SQLite corrections store and response cache vanish on restart; a deployed URL with a live key is a spend surface, so gate or rate-limit Classify; and uploaded files are never cleaned up yet.

**Effort:** S
**Priority:** P2
**Depends on:** Nothing.

## Completed

_(none yet)_
