# Headroom Middleware Plugin for Hermes Agent

A Hermes Agent plugin that compresses LLM request context using [Headroom](https://github.com/chopratejas/headroom) before provider calls. Not a context engine replacement — additive middleware that sits between Hermes and your API provider.

## What it does

- **Compresses large tool outputs** before they hit the LLM (81%+ token reduction on real data)
- **Preserves system prompts, file reads, browser snapshots, terminal output** — never compresses reference data
- **Tracks cumulative savings** per session via `headroom_status` tool
- **Shows savings in the runtime footer** — `🗜️ 562` tokens saved displayed alongside model/context%
- **Appends savings note to LLM responses** via `transform_llm_output` hook — agent sees compression stats after each turn with activity
- **Registers `headroom_retrieve`** for CCR marker recovery
- **Fail-open by default** — if Headroom throws, requests go through uncompressed

## Architecture

```
Hermes conversation loop
  → build provider kwargs
  → llm_request middleware (compresses messages)
  → Hermes normal provider execution (OpenRouter, Anthropic, etc.)
```

Hermes stays in control of routing, retries, streaming, tools, memory, and context compression. The plugin only rewrites the request payload.

## Install

```bash
# 1. Install headroom-ai in the Hermes venv
~/.hermes/hermes-agent/venv/bin/pip install "headroom-ai[proxy,mcp]"

# 2. Copy plugin to Hermes user plugins
cp -r headroom ~/.hermes/plugins/

# 3. Enable in config
hermes config set headroom.enabled true
hermes plugins enable headroom

# 4. Enable footer display (optional — shows 🗜️ savings in context bar)
hermes config set display.runtime_footer.enabled true
hermes config set display.runtime_footer.fields '[model,context_pct,cwd,headroom_saved]'

# 5. Restart Hermes
```

## Config

```yaml
headroom:
  enabled: true
  fail_open: true
  min_chars: 8000
  compress_user_messages: false
  compress_system_messages: false
  protect_recent: 4
  exclude_roles: system
  exclude_tools: browser_snapshot,read_file,read_terminal,terminal,execute_code
  api_modes: chat_completions,codex_responses,anthropic_messages

display:
  runtime_footer:
    enabled: true
    fields: [model, context_pct, cwd, headroom_saved]
```

## Tools

- `headroom_retrieve` — Retrieve original uncompressed content behind CCR markers
- `headroom_status` — Show cumulative session savings (tokens saved, compression count, ratio)

## Tested

- 21,441 → 4,025 tokens (81.2% reduction) on realistic JSON tool output
- Zero compression on system messages, file reads, browser snapshots, terminal output
- Fail-open verified: Headroom errors pass through uncompressed
- Edge cases verified: empty messages, missing fields, unknown api_mode, None content — all handled without crashing
- Footer integration: `owl-alpha · 39% · ~ · 🗜️ 562` displays correctly
- `transform_llm_output` hook: savings note appended to response text when compressions occurred; returns None (unchanged) when no activity
- 10/10 automated tests pass
