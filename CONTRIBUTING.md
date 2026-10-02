# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # add ,lstm for the PyTorch LSTM tests
pytest -q
```

Python 3.10 to 3.12. The test suite needs no API key and no network: LLM providers are mocked.

## Ground rules

- **Evidence first.** A number in the README, the docs or the UI must come from a command in this repository
  (`python -m aeromind report`, `evaluate`, `bench`, ...) or be labelled an assumption. Do not type results by hand.
- **Say what is simulated.** Label results as simulated, real public dataset, or unvalidated hardware. Do not claim
  certification, partnerships, flight data, ACARS/SWIM/MRO connectivity, or Jetson/Raspberry Pi results that were not measured.
- **The LLM explains; the deterministic pipeline decides.** The copilot must not change, invent or override a
  deterministic result. Changes under `src/aeromind/llm/` need tests in `tests/llm/` (including a safety case).
- **One inference implementation.** The edge agent, the simulated fleet and the CLI all use `EdgePipeline`; do not add a parallel path.
- **No secrets.** Never commit `.env`, keys or tokens. `.env.example` lists the variables.

## Layout

See [docs/architecture.md](docs/architecture.md). Tests: `tests/unit`, `tests/integration`, `tests/llm`, `tests/e2e`
(browser tests skip when Playwright is not installed).

## Pull requests

Keep changes focused, run `pytest`, update the docs that mention what you changed, and keep
[docs/limitations.md](docs/limitations.md) truthful.
