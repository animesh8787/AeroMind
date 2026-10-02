# Security

AeroMind is a research prototype on simulated data. It is not certified, not for use on aircraft, and not for
safety-critical decisions.

## Reporting a vulnerability

Please report security issues privately to the repository owner (use the contact on their GitHub profile) rather than in a public issue.
Include what you found, how to reproduce it, and the affected version or commit.

## Security-relevant design

- **Model updates:** packages are signed with Ed25519 over a SHA-256 manifest; a modified file is rejected and the last valid model stays active (`security/signing.py`, tested).
- **Secrets:** `GROQ_API_KEY` and the ingest token are read from the environment or a git-ignored `.env`. They are never printed, logged or returned by the API; `aeromind llm doctor` shows only whether they are set.
- **LLM boundary:** the copilot is ground-side, receives a structured summary (not raw sensor streams), cannot change deterministic results, and its output is validated before display.
- **Prompt injection:** user text is treated as untrusted in the prompt, and model output is filtered for invented values, approval claims and control commands. This reduces risk; it is not a guarantee.

## Known gaps

- The ground-station web UI and API have **no authentication**. Run on localhost or a trusted network; set `AEROMIND_INGEST_TOKEN` when listening on a LAN. Do not expose to the internet.
- The ground-station signing key lives in `artifacts/ground/keys` on the same machine; there is no key management or hardware root of trust.
- Sending a question to Groq sends the structured context to a third-party cloud service. Use Ollama or `AEROMIND_LLM_PROVIDER=off` to keep everything local.
- Dependencies are not pinned to exact versions.
