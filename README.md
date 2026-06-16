# Headroom Middleware Plugin for Hermes Agent

A Hermes Agent plugin that compresses LLM request context using [Headroom](https://github.com/chopratejas/headroom) before provider calls. Not a context engine replacement — additive middleware that sits between Hermes and your API provider.

## What it does

- **Compresses large tool outputs** before they hit the LLM (79%+ token reduction on real data)
- **Preserves system prompts, file reads, browser snapshots, terminal output** — never compresses reference data
- **Tracks cumulative savings** per session via `headroom_status` tool
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

# 4. Restart Hermes
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
```

## Tools

- `headroom_retrieve` — Retrieve original uncompressed content behind CCR markers
- `headroom_status` — Show cumulative session savings

## Tested

- 21,702 → 4,543 tokens (79.1% reduction) on realistic JSON tool output
- Zero compression on system messages, file reads, browser snapshots
- Fail-open verified: Headroom errors pass through uncompressed
