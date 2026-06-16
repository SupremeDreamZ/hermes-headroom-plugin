"""Headroom middleware plugin for Hermes Agent.

This plugin keeps Hermes in control of provider routing, retries, streaming,
fallbacks, tools, memory, and context compression. It only rewrites the LLM
request payload immediately before Hermes sends it to the active provider.

Architecture:
    Hermes conversation loop -> provider kwargs -> this llm_request middleware
    -> headroom.compress(messages) -> Hermes normal provider execution.

It also registers a native ``headroom_retrieve`` tool so CCR markers produced
by either inline Headroom compression or the older proxy route are recoverable.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from typing import Any

from hermes_cli.config import cfg_get, load_config
from hermes_cli.middleware import LLM_REQUEST_MIDDLEWARE
from tools.registry import tool_error, tool_result

log = logging.getLogger(__name__)

_HEADROOM_TOOL_NAME = "headroom_retrieve"
_HASH_RE = re.compile(r"[a-fA-F0-9]{24}")

HEADROOM_RETRIEVE_SCHEMA = {
    "name": _HEADROOM_TOOL_NAME,
    "description": (
        "Retrieve the original uncompressed content behind a Headroom CCR "
        "compression marker. Markers look like '[N items compressed ... "
        "hash=abc123...]', '<<ccr:abc123...>>', or "
        "'<<ccr:abc123...,base64,4.5KB>>'. They are not file paths. "
        "When you need exact details hidden behind one of these markers, call "
        "this tool with the hash. Use the optional query to search/filter very "
        "large originals."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "hash": {
                "type": "string",
                "description": "24-character hash from the Headroom compression marker, or the full marker text.",
            },
            "query": {
                "type": "string",
                "description": "Optional search query to filter large originals to relevant chunks/items.",
            },
        },
        "required": ["hash"],
    },
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _headroom_cfg(name: str, default: Any = None) -> Any:
    try:
        cfg = load_config()
        return cfg_get(cfg, "headroom", name, default=default)
    except Exception as exc:  # pragma: no cover - fail-open config read guard
        log.debug("Headroom plugin could not read config key %s: %s", name, exc)
        return default


def _headroom_enabled() -> bool:
    return _as_bool(_headroom_cfg("enabled", False), False)


def _allowed_api_modes() -> set[str]:
    raw = _headroom_cfg(
        "api_modes",
        ["chat_completions", "codex_responses", "anthropic_messages"],
    )
    if isinstance(raw, str):
        # Handle both JSON array strings and comma-separated strings
        raw = raw.strip()
        if raw.startswith("["):
            try:
                import json
                raw = json.loads(raw)
            except Exception:
                raw = [part.strip().strip('"').strip("'") for part in raw.split(",")]
        else:
            raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, (list, tuple, set)):
        raw = ["chat_completions", "codex_responses", "anthropic_messages"]
    return {str(item).strip() for item in raw if str(item).strip()}


def _select_message_field(request: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]] | None]:
    for field in ("messages", "input"):
        value = request.get(field)
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return field, value
    return None, None


def _estimate_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        try:
            total += len(json.dumps(message, ensure_ascii=False, default=str))
        except Exception:
            total += len(str(message))
    return total


def _compress_messages(messages: list[dict[str, Any]], *, model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from headroom import CompressConfig, compress

    config = CompressConfig(
        compress_user_messages=_as_bool(_headroom_cfg("compress_user_messages", False), False),
        compress_system_messages=_as_bool(_headroom_cfg("compress_system_messages", False), False),
        protect_recent=_as_int(_headroom_cfg("protect_recent", 4), 4),
        protect_analysis_context=_as_bool(_headroom_cfg("protect_analysis_context", True), True),
        min_tokens_to_compress=_as_int(_headroom_cfg("min_tokens_to_compress", 250), 250),
        target_ratio=_as_optional_float(_headroom_cfg("target_ratio", None)),
        kompress_model=_as_optional_str(_headroom_cfg("kompress_model", None)),
        savings_profile=_as_optional_str(_headroom_cfg("savings_profile", None)),
    )

    kwargs: dict[str, Any] = {"model": model or "gpt-4o", "config": config}
    model_limit = _headroom_cfg("model_limit", None)
    if model_limit not in (None, ""):
        kwargs["model_limit"] = _as_int(model_limit, 200000)

    result = compress(messages, **kwargs)
    stats = {
        "tokens_before": int(getattr(result, "tokens_before", 0) or 0),
        "tokens_after": int(getattr(result, "tokens_after", 0) or 0),
        "tokens_saved": int(getattr(result, "tokens_saved", 0) or 0),
        "compression_ratio": float(getattr(result, "compression_ratio", 0.0) or 0.0),
        "transforms_applied": list(getattr(result, "transforms_applied", []) or []),
    }
    return list(getattr(result, "messages", messages) or messages), stats


def _excluded_roles() -> set[str]:
    """Message roles that should never be compressed."""
    raw = _headroom_cfg("exclude_roles", ["system"])
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, (list, tuple, set)):
        raw = ["system"]
    return {str(item).strip().lower() for item in raw if str(item).strip()}


def _excluded_tool_names() -> set[str]:
    """Tool names whose output should never be compressed.

    These are tools that return reference data the agent needs verbatim
    (file contents, browser snapshots, terminal output, etc.).
    """
    raw = _headroom_cfg(
        "exclude_tools",
        ["browser_snapshot", "read_file", "read_terminal", "terminal", "execute_code"],
    )
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, (list, tuple, set)):
        raw = ["browser_snapshot", "read_file", "read_terminal", "terminal", "execute_code"]
    return {str(item).strip().lower() for item in raw if str(item).strip()}


def _should_compress_message(message: dict[str, Any], *, exclude_roles: set[str], exclude_tools: set[str]) -> bool:
    """Return True if a message is eligible for compression."""
    role = str(message.get("role", "")).strip().lower()
    if role in exclude_roles:
        return False
    # Never compress tool messages from excluded tools
    if role == "tool":
        tool_name = str(message.get("name", "") or message.get("tool_name", "")).strip().lower()
        if tool_name in exclude_tools:
            return False
        # Only compress large tool outputs (>2000 chars)
        content = str(message.get("content", "") or "")
        if len(content) < 2000:
            return False
    return True


def _filter_messages_for_compression(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Return (compressible_messages, their original indices)."""
    exclude_roles = _excluded_roles()
    exclude_tools = _excluded_tool_names()
    compressible = []
    indices = []
    for idx, msg in enumerate(messages):
        if _should_compress_message(msg, exclude_roles=exclude_roles, exclude_tools=exclude_tools):
            compressible.append(msg)
            indices.append(idx)
    return compressible, indices


def headroom_llm_request_middleware(request: dict[str, Any], **context: Any) -> dict[str, Any] | None:
    """Compress provider request messages before Hermes sends them.

    Return ``None`` to leave the request untouched. Any exception fails open by
    default so Headroom can never break normal Hermes provider execution unless
    ``headroom.fail_open: false`` is explicitly configured.
    """
    if not _headroom_enabled():
        return None

    api_mode = str(context.get("api_mode") or "").strip()
    allowed_modes = _allowed_api_modes()
    if api_mode and "*" not in allowed_modes and api_mode not in allowed_modes:
        return None

    field, messages = _select_message_field(request)
    if not field or not messages:
        return None

    # Filter to only compressible messages
    compressible, compress_indices = _filter_messages_for_compression(messages)
    if not compressible:
        return None

    min_chars = _as_int(_headroom_cfg("min_chars", 8000), 8000)
    char_count = _estimate_chars(compressible)
    if char_count < min_chars:
        return None

    fail_open = _as_bool(_headroom_cfg("fail_open", True), True)
    model = str(context.get("model") or request.get("model") or "")

    try:
        compressed_subset, stats = _compress_messages(compressible, model=model)
    except Exception as exc:
        log.warning("Headroom compression failed; continuing uncompressed: %s", exc)
        if fail_open:
            return None
        raise

    if stats["tokens_saved"] <= 0:
        return None

    # Merge compressed messages back into the full message list
    new_messages = list(messages)
    for orig_idx, compressed_msg in zip(compress_indices, compressed_subset):
        new_messages[orig_idx] = compressed_msg

    # Accumulate session stats
    _session_stats["tokens_saved_total"] += stats["tokens_saved"]
    _session_stats["tokens_before_total"] += stats["tokens_before"]
    _session_stats["tokens_after_total"] += stats["tokens_after"]
    _session_stats["compressions"] += 1

    new_request = copy.deepcopy(request)
    new_request[field] = new_messages
    return {
        "request": new_request,
        "source": "headroom",
        "name": "headroom",
        "reason": (
            f"compressed {stats['tokens_before']} -> {stats['tokens_after']} tokens; "
            f"saved {stats['tokens_saved']} ({stats['compression_ratio']:.1%}); "
            f"transforms={','.join(stats['transforms_applied']) or 'none'}"
        ),
    }


def _normalize_hash(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    match = _HASH_RE.search(text)
    if match:
        return match.group(0).lower()
    # Tolerate old/short dev markers enough to return a clear miss instead of
    # telling the model to read a fake file path.
    text = text.strip("<>").removeprefix("ccr:").removeprefix("hash=")
    return text.split(",", 1)[0].strip().lower()


def _retrieve_from_local_store(hash_key: str, query: str) -> dict[str, Any] | None:
    try:
        from headroom.cache.compression_store import get_compression_store
    except Exception as exc:
        log.debug("Headroom local compression store unavailable: %s", exc)
        return None

    try:
        store = get_compression_store()
        if query:
            results = store.search(hash_key, query)
            entry = store.retrieve(hash_key, query=query)
            if entry is None:
                return None
            return {
                "source": "local-store",
                "hash": hash_key,
                "query": query,
                "search_results": results,
                "result_count": len(results),
                "original_tokens": entry.original_tokens,
                "compressed_tokens": entry.compressed_tokens,
                "tool_name": entry.tool_name,
                "note": "Search results are filtered. Omit query to retrieve the full original content.",
            }

        entry = store.retrieve(hash_key)
        if entry is None:
            return None
        return {
            "source": "local-store",
            "hash": hash_key,
            "original_content": entry.original_content,
            "original_tokens": entry.original_tokens,
            "compressed_tokens": entry.compressed_tokens,
            "tool_name": entry.tool_name,
            "original_item_count": entry.original_item_count,
            "compressed_item_count": entry.compressed_item_count,
        }
    except Exception as exc:
        log.debug("Headroom local retrieve failed for %s: %s", hash_key, exc)
        return None


def _proxy_urls() -> list[str]:
    raw = _headroom_cfg("retrieve_proxy_urls", ["http://127.0.0.1:8788", "http://127.0.0.1:8787"])
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                import json
                raw = json.loads(raw)
            except Exception:
                raw = [part.strip().strip('"').strip("'") for part in raw.split(",")]
        else:
            raw = [part.strip() for part in raw.split(",")]
    urls = []
    if isinstance(raw, (list, tuple, set)):
        for value in raw:
            text = str(value or "").strip().rstrip("/")
            if text:
                urls.append(text)
    return urls or ["http://127.0.0.1:8788", "http://127.0.0.1:8787"]


def _retrieve_from_proxy(hash_key: str, query: str) -> dict[str, Any] | None:
    try:
        import httpx
    except Exception as exc:
        log.debug("httpx unavailable for Headroom proxy retrieve: %s", exc)
        return None

    payload: dict[str, Any] = {"hash": hash_key}
    if query:
        payload["query"] = query

    last_error = None
    for base_url in _proxy_urls():
        try:
            resp = httpx.post(f"{base_url}/v1/retrieve", json=payload, timeout=15)
        except Exception as exc:
            last_error = f"{base_url}: {type(exc).__name__}: {exc}"
            continue
        if resp.status_code == 404:
            last_error = f"{base_url}: 404 not found/expired"
            continue
        if resp.status_code != 200:
            last_error = f"{base_url}: HTTP {resp.status_code}: {resp.text[:200]}"
            continue
        data = resp.json()
        return {
            "source": f"proxy:{base_url}",
            "hash": hash_key,
            "original_content": data.get("original_content", ""),
            "original_tokens": data.get("original_tokens"),
            "tool_name": data.get("tool_name"),
        }

    if last_error:
        return {"_miss": last_error}
    return None


def _handle_headroom_retrieve(args: dict[str, Any], **kw: Any) -> str:
    hash_key = _normalize_hash(args.get("hash"))
    if not hash_key:
        return tool_error("hash is required from a Headroom CCR marker like 'hash=<24 hex>' or '<<ccr:<24 hex>>'")

    query = str(args.get("query") or "").strip()

    result = _retrieve_from_local_store(hash_key, query)
    if result is not None:
        return tool_result(result)

    proxy_result = _retrieve_from_proxy(hash_key, query)
    if proxy_result and "_miss" not in proxy_result:
        return tool_result(proxy_result)

    miss = proxy_result.get("_miss") if isinstance(proxy_result, dict) else None
    detail = f" Last proxy result: {miss}." if miss else ""
    return tool_error(
        "Content not found in the inline Headroom store or configured proxy stores. "
        "The CCR TTL may have expired, the proxy may have restarted, or the marker "
        "came from a different Hermes process. Re-run the original command/request "
        "to regenerate the data." + detail
    )


def _headroom_available() -> bool:
    try:
        import headroom  # noqa: F401
        return True
    except Exception:
        return False


# -- Session cumulative stats ------------------------------------------------

_session_stats: dict[str, int] = {
    "tokens_saved_total": 0,
    "tokens_before_total": 0,
    "tokens_after_total": 0,
    "compressions": 0,
}


def _get_session_stats() -> dict[str, int]:
    return dict(_session_stats)


def _reset_session_stats() -> None:
    _session_stats["tokens_saved_total"] = 0
    _session_stats["tokens_before_total"] = 0
    _session_stats["tokens_after_total"] = 0
    _session_stats["compressions"] = 0


def _handle_headroom_status(args: dict[str, Any], **kw: Any) -> str:
    stats = _get_session_stats()
    saved = stats["tokens_saved_total"]
    comps = stats["compressions"]
    before = stats["tokens_before_total"]
    after = stats["tokens_after_total"]

    if comps == 0:
        return tool_result({
            "status": "no_compressions_yet",
            "message": "No compressions have been performed this session. Send a message with large context to trigger Headroom.",
            "enabled": _headroom_enabled(),
        })

    ratio = (saved / before * 100) if before > 0 else 0.0
    return tool_result({
        "status": "ok",
        "compressions": comps,
        "tokens_saved": saved,
        "tokens_before_total": before,
        "tokens_after_total": after,
        "savings_pct": round(ratio, 1),
        "message": f"Saved {saved:,} tokens across {comps} compression{'s' if comps != 1 else ''} ({ratio:.1f}% reduction)",
    })


HEADROOM_STATUS_SCHEMA = {
    "name": "headroom_status",
    "description": "Show cumulative Headroom compression stats for this session.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def register(ctx: Any) -> None:
    ctx.register_middleware(LLM_REQUEST_MIDDLEWARE, headroom_llm_request_middleware)
    ctx.register_tool(
        name=_HEADROOM_TOOL_NAME,
        toolset="headroom",
        schema=HEADROOM_RETRIEVE_SCHEMA,
        handler=_handle_headroom_retrieve,
        check_fn=_headroom_available,
        emoji="🗜️",
        override=True,
    )
    ctx.register_tool(
        name="headroom_status",
        toolset="headroom",
        schema=HEADROOM_STATUS_SCHEMA,
        handler=_handle_headroom_status,
        check_fn=_headroom_available,
        emoji="📊",
        override=True,
    )
