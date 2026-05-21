# Session 6 Cognitive Agent (agent6)

This repository contains a small cognitive agent example (Session 6) that runs a decision/perception loop backed by an LLM gateway and a set of sandboxed tools. The main entrypoint is `agent6.py` and the tool implementations are in `mcp_server.py`.

## Key components
- `agent6.py` — main agent loop, memory, artifact store, decision & perception layers, and `CognitiveAgent` class.
- `mcp_server.py` — tool implementations exposed to the agent runtime (web search, fetch_url, filesystem sandbox, calendar event creation, etc.).
- `llm_gatewayV3/` — local LLM gateway client used by the agent to call structured LLM responses.
- `sandbox/` — writable sandbox used by file tools and calendar event creation.


## Prerequisites
- Python 3.10+ (project uses features from recent Python versions).
- Install dependencies in `requirements.txt` (recommend using a virtualenv):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Environment and secrets
- The repo uses a `.env` file for API keys (e.g. `TAVILY_API_KEY`). **Do not commit** this file — it is already ignored by `.gitignore`.
- Usage logging is written to `usage.json`; this file is also ignored by `.gitignore`.

## Running the agent

Basic one-shot usage:

```bash
python agent6.py --state-dir state_run1 --prompt "Your query here"
```

- `--state-dir` controls persistent memory and artifacts for that run (use separate state dirs for independent traces).
- Example prompts (used by tests and examples):
  - Fetch a webpage and extract facts:
    `python agent6.py --state-dir state_run1 --prompt "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory."`
  - Find activities and check weather:
    `python agent6.py --state-dir state_run2 --prompt "Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather and recommend the most appropriate."`
  - Remember facts and create reminders:
    `python agent6.py --state-dir state_run3 --prompt "My mom's birthday is 22 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day."`
  - Search and synthesize pages:
    `python agent6.py --state-dir state_run4 --prompt "Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on."`

## How the agent works (high level)
- The `CognitiveAgent.run()` loop performs up to `--max-iters` iterations.
- The Perception layer (`_perceive`) turns a query into 1–4 goals using the LLM.
- The Decision layer (`_decide`) chooses a single next action: either answer or call a tool.
- Tools are implemented in `mcp_server.py` and invoked via `ToolRuntime.call()`; fetches are recorded as artifacts and persisted in the `state_dir` under `artifacts/`.
- The `MemoryStore` persists facts and tool results to `memory.json` in the `state_dir`.

## Tools provided
- `web_search(query, max_results)` — Tavily primary, DuckDuckGo fallback (hard cap 5 results).
- `fetch_url(url)` — headless crawl via `crawl4ai` that returns clean markdown and is stored as an artifact.
- `get_time(timezone)` — returns current time in a named IANA timezone.
- `currency_convert(amount, from_currency, to_currency)` — converts via frankfurter.dev.
- Filesystem sandbox: `read_file`, `list_dir`, `create_file`, `update_file`, `edit_file` (sandboxed under `sandbox/`).
- `create_calendar_event(title, date, reminders)` — creates a `.ics` file in `sandbox/calendar/`.

## Tests
- Run unit tests with:

```bash
python -m unittest discover -v
```

The test suite uses `tests/test_agent6.py` which includes `FakeToolModule` and `FakeBrain` to perform deterministic assertions without network calls.

## Security & privacy notes
- Keep `.env` out of version control (it is listed in `.gitignore`).
- `usage.json` logs search provider usage and is ignored by git; remove or rotate keys if you share state directories.

## Suggestions / common edits
- `.gitignore` already includes `.env` and `usage.json`. Consider also ignoring common virtualenv directories (e.g. `.venv/`) if you use them.

## Contact / next steps
- If you want, I can:
  - Add a short `Makefile` or `scripts/` helpers for common commands.
  - Expand README with developer notes on adding new tools.

---
Generated on: 2026-05-21
