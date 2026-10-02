# LLM maintenance copilot

A **ground-side** assistant that explains what the deterministic AeroMind pipeline already decided.
It runs on the ground station (laptop), never on the aircraft or the edge device.

> **Honesty note.** The provider, validation, safety and fallback layers are tested with mocked providers
> and a fake Ollama HTTP server (`pytest tests/llm`). The quality of answers from a *live* Groq or Ollama
> model has **not** been evaluated in this repository. Every answer is labelled as AI-generated or as a
> rule-based template.

## What it may and may not do

The deterministic system decides: anomaly, fault, confidence, RUL, sensor health, physics evidence,
failure risk, the maintenance policy outcome and every cost figure. The LLM only writes explanatory text
about those results. It must never control aircraft systems, change or override a result, invent values,
faults, history, costs or parts availability, or claim certification, regulatory approval or OEM
authorisation. The rules are in `llm/safety.py` (`SAFETY_RULES`) and are repeated in the system prompt.

## Providers and fallback

```
Groq (cloud, primary)  ->  Ollama (local)  ->  rule-based template (always available)
```

| `AEROMIND_LLM_PROVIDER` | Behaviour |
|---|---|
| `auto` (default) | Groq if `GROQ_API_KEY` is set, then Ollama if its server is running with the model pulled |
| `groq`, `ollama` | only that provider |
| `ollama,groq` | custom priority list |
| `off` | rule-based templates only |

The ground-station header and the copilot panel show **LLM STATUS**:

| Status | Meaning |
|---|---|
| `ONLINE` | Groq will answer |
| `LOCAL` | Ollama will answer |
| `FALLBACK` | no LLM is reachable; answers are rule-based templates |
| `UNAVAILABLE` | the copilot itself failed on that request |

A provider that fails is skipped for 30 s. Each response states whether it was AI-generated
(`ai_generated`, `provider`, `model`); a template answer is never presented as LLM output.
The application is fully functional with no API key and no internet.

## Configuration

Copy `.env.example` to `.env` (git-ignored) or set environment variables. Real environment variables win over `.env`.

| Variable | Purpose |
|---|---|
| `AEROMIND_LLM_PROVIDER` | provider priority (above) |
| `GROQ_API_KEY`, `GROQ_MODEL` | Groq key and model (default `llama-3.3-70b-versatile`; Groq retires models, so check their list) |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Ollama server (default `http://127.0.0.1:11434`) and model (default `llama3.2`) |
| `AEROMIND_KNOWLEDGE_DIR` | optional override of `docs/knowledge` |

The key is only placed in the HTTPS `Authorization` header. It is never printed, logged, put in an
exception message, or returned by the API. `aeromind llm doctor` reports it only as `set` / `NOT set`.

## What the model receives

Not raw sensor data. `llm/context.py` builds a controlled summary from the deterministic outputs:
aircraft, flight phase, anomaly score and trend, fault and confidence, RUL p10/p50/p90, sensor health,
physics evidence, failure-before-check probability, decision, window, required part (placeholder),
recent advisories, cost assumptions (labelled), a simulation statement, and an explicit list of what is
**not** available (maintenance history, parts stock, real flight data). The simulator's hidden ground
truth is excluded. Two short excerpts from `docs/knowledge/` are added as reference notes.

## Tasks

| Task (UI button) | Notes |
|---|---|
| `explain_alert` (Explain Alert) | what happened, evidence, confidence, RUL, limitations |
| `summarize_aircraft` | concise engineering summary |
| `why_fault` (Why This Fault?) | classifier vs sensor fault vs other, from stated evidence only |
| `explain_rul` | p10/p50/p90 and uncertainty; never a guarantee |
| `maintenance_assist` (What Should Maintenance Inspect?) | draft, technician review required |
| `what_if` (What If We Defer?) | numbers come from the **deterministic decision engine** (same code as the ground station); the model only explains them. Delay is in flight hours. |
| `work_order` (Draft Work Order) | status `DRAFT`. Only the inspection checklist and notes may come from the LLM; aircraft, fault, evidence, RUL, risk, decision, window and part are copied from the deterministic data by code |
| `fleet_summary` | counts, attention list, sensor problems, workload and patterns are computed by code; the LLM writes only the narrative |

A free-text question is routed to one of these tasks by keyword (`infer_task`).

## Validation and safety

1. The model must return one JSON object matching a typed schema (`AdvisoryExplanation`, or the two work-order fields).
2. The text is checked (`llm/safety.py`): numbers must appear in the context (and a number stated next to "RUL" must be one of the p10/p50/p90 values); no unknown aircraft, no fault type absent from the context, no decision other than the deterministic one; no certification, approval or OEM-authorisation claims; no flight-control or system commands.
3. Malformed or rejected output gets **one** repair retry that lists what was wrong.
4. If it is still rejected, or no provider is available, the rule-based template is returned and the response says so.

These are conservative text filters, not a proof. A wrong but plausible sentence that uses only context numbers can pass; the label "AI-generated assistance, technician review required" is always attached by code, not by the model.

## Interfaces

- Ground-station UI: **AI Maintenance Copilot** panel (aircraft selector, quick actions, free-text box). Deterministic output and AI text are separate, labelled blocks.
- API: `GET /api/llm/status`, `POST /api/copilot` with `{"tail", "task", "question", "hours_to_next_check"}`. Unknown aircraft returns 404, unknown task 400.
- CLI: `aeromind copilot --aircraft VT-AMA03 --inject bearing_wear` (interactive), `--ask "..."`, `--task work_order`; `aeromind llm doctor [--ping]`.

## Groq setup

1. Create a key in the Groq console (https://console.groq.com).
2. Put it in `.env` as `GROQ_API_KEY=...` (or set it in the environment). Never commit `.env`.
3. `aeromind llm doctor --ping` sends one tiny request to check the key and model.

## Ollama setup

1. Install Ollama (https://ollama.com), then `ollama pull llama3.2` and keep `ollama serve` running.
2. `aeromind llm doctor` should show the server reachable and the model pulled.
3. Force it with `AEROMIND_LLM_PROVIDER=ollama`.

## Tests

`pytest tests/llm tests/integration/test_server_copilot.py` run in CI with no API key and no network:
provider contracts (Groq and Ollama with mocked HTTP), real HTTP against a fake Ollama server, router order,
schema validation, malformed output and repair, safety rules (including a prompt-injection attempt), context
construction, work-order and fleet generation, and the API.
