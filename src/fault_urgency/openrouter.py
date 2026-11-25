"""Send one non-streaming, zero-priced request to OpenRouter."""

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from pathlib import Path
from time import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dotenv import dotenv_values

_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
_FREE_MODEL = "openrouter/free"
_MAX_COMPLETION_TOKENS = 2048
_TIMEOUT_SECONDS = 30
_RESPONSE_LIMIT = 1_048_576
_STATUS_MESSAGES = {
    400: "OpenRouter rejected the structured-output request.",
    401: "OpenRouter rejected the API key. Check the value in the local .env file.",
    402: "OpenRouter rejected the free request due to an account, credit, or key limit.",
    403: "OpenRouter denied access. Check account permissions and regional availability.",
    404: "The requested route or a compatible free model is unavailable.",
    408: "The OpenRouter request timed out.",
    429: "The OpenRouter or free model request limit was reached.",
    451: "The OpenRouter request is blocked by a regional or legal restriction.",
    500: "OpenRouter encountered a server error.",
    502: "The free model provider failed to serve the request.",
    503: "No compatible free model is available for the request.",
    504: "The free model provider timed out.",
}
_REGIONAL_ERROR = re.compile(r"\b(?:region(?:al)?|geograph\w*|country|location|jurisdiction)\b")
_MODEL_SLUG = re.compile(r"[a-zA-Z0-9_.-]{1,80}/[a-zA-Z0-9_.-]{1,120}(?::free)?")
_GENERATION_ID = re.compile(r"gen-[a-zA-Z0-9_-]{1,120}")
_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "error"})
_TOKEN_LIMIT_REASONS = frozenset({"length", "max_tokens", "max_output_tokens", "max_completion_tokens"})
_NATIVE_REASONS = _FINISH_REASONS | _TOKEN_LIMIT_REASONS | {
    "end_turn", "eos", "eos_token", "refusal", "safety", "stop_sequence", "tool_use",
}


@dataclass(frozen=True, slots=True)
class OpenRouterDiagnostics:
    """Allowlisted metadata reported by OpenRouter."""

    model: str | None
    generation_id: str | None
    finish_reason: str | None
    native_finish_reason: str | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    content_characters: int | None
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    http_status: int | None = None
    max_completion_tokens: int = _MAX_COMPLETION_TOKENS
    response_format: str = "json_schema"
    cost: int | str | None = None


class OpenRouterError(RuntimeError):
    """An OpenRouter failure with a message that is safe to display."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_delay: float | None = None,
        issue: str = "request_error",
        diagnostics: OpenRouterDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_delay = retry_delay
        self.issue = issue
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class _Completion:
    diagnostics: OpenRouterDiagnostics
    content: str = field(default="", repr=False)
    issue: str | None = None


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl) -> None:
        return None


_OPENER = build_opener(_NoRedirects())


def _read_api_key(env_file: str | Path) -> str:
    try:
        values = dotenv_values(env_file, encoding="utf-8-sig", interpolate=False)
    except (OSError, UnicodeError):
        raise OpenRouterError("The local .env file could not be read.") from None
    key = (values.get("OPENROUTER_API_KEY") or "").strip()
    if not key or not key.isascii() or not key.isprintable() or any(ch.isspace() for ch in key):
        raise OpenRouterError("Set OPENROUTER_API_KEY in the local .env file.")
    return key


def _decode_response(raw: bytes) -> dict[str, Any] | None:
    if len(raw) > _RESPONSE_LIMIT:
        return None
    try:
        data = json.loads(raw, parse_float=Decimal)
    except (ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _retry_delay(value: str | None) -> float | None:
    if not value:
        return None
    try:
        if re.fullmatch(r"[0-9]{1,12}", value.strip()):
            return float(value)
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is not None:
            return max(0, retry_at.timestamp() - time())
    except (ValueError, OverflowError, OSError):
        pass
    return None


def _api_error(
    status: int,
    error: object = None,
    *,
    retry_after: str | None = None,
    diagnostics: OpenRouterDiagnostics | None = None,
) -> OpenRouterError:
    issue = "request_error"
    if isinstance(error, dict):
        remote_code = error.get("code")
        if status == 200 and type(remote_code) is int and 400 <= remote_code <= 599:
            status = remote_code
        message = error.get("message")
        if isinstance(message, str) and status in {400, 403, 404, 451, 503}:
            normalized = message.casefold()
            if _REGIONAL_ERROR.search(normalized):
                return OpenRouterError(
                    f"OpenRouter reported a regional restriction (status {status}).",
                    status_code=status, issue="regional_restriction", diagnostics=diagnostics,
                )
            if "no endpoints found" in normalized or "no compatible" in normalized:
                return OpenRouterError(
                    f"No compatible free model is available for the request (status {status}).",
                    status_code=status, issue="free_model_unavailable", diagnostics=diagnostics,
                    retry_delay=_retry_delay(retry_after),
                )
    if 300 <= status < 400:
        return OpenRouterError(
            f"OpenRouter returned a redirect (status {status}).",
            status_code=status, diagnostics=diagnostics,
        )
    if status in {404, 503}:
        issue = "free_model_unavailable"
    elif status == 429:
        issue = "rate_limited"
    reason = _STATUS_MESSAGES.get(status, "OpenRouter failed to return a decision.")
    return OpenRouterError(
        f"{reason.rstrip('.')} (status {status}).",
        status_code=status,
        retry_delay=_retry_delay(retry_after) if status in {429, 503} else None,
        issue=issue,
        diagnostics=diagnostics,
    )


def _require_free_response(
    data: dict[str, Any], diagnostics: OpenRouterDiagnostics, *, allow_missing: bool = False
) -> None:
    model = data.get("model")
    free_model = isinstance(model, str) and (model == _FREE_MODEL or model.endswith(":free"))
    if not free_model and (not allow_missing or "model" in data):
        raise OpenRouterError(
            "OpenRouter did not identify a free model in the response.",
            issue="cost_guard", diagnostics=diagnostics,
        )

    usage = data.get("usage")
    if not isinstance(usage, dict):
        if allow_missing and usage is None:
            return
        raise OpenRouterError(
            "OpenRouter did not return usage data to verify zero cost.",
            issue="cost_guard", diagnostics=diagnostics,
        )
    cost = usage.get("cost")
    if (not allow_missing or "cost" in usage) and (type(cost) not in (int, Decimal) or cost != 0):
        raise OpenRouterError(
            "OpenRouter did not confirm a zero usage cost for the free request.",
            issue="cost_guard", diagnostics=diagnostics,
        )
    if usage.get("is_byok") is True:
        raise OpenRouterError(
            "OpenRouter reported a bring-your-own-key route for the request.",
            issue="cost_guard", diagnostics=diagnostics,
        )
    details = usage.get("cost_details")
    if isinstance(details, dict) and details.get("upstream_inference_cost") not in (None, 0):
        raise OpenRouterError(
            "OpenRouter reported nonzero upstream inference cost.",
            issue="cost_guard", diagnostics=diagnostics,
        )


def _finish_label(value: object, allowed: frozenset[str]) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in allowed:
        return value.lower()
    return "other"


def _token_count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _safe_identifier(value: object, pattern: re.Pattern[str], api_key: str) -> str | None:
    if isinstance(value, str) and pattern.fullmatch(value) and api_key not in value:
        return value
    return None


def _first_choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    return first_choice if isinstance(first_choice, dict) else {}


def _reported_cost(value: object) -> int | str | None:
    if type(value) in (int, Decimal) and Decimal(value).is_finite() and value >= 0:
        return 0 if value == 0 else str(value)
    return None


def _response_diagnostics(
    data: dict[str, Any], api_key: str, http_status: int | None
) -> OpenRouterDiagnostics:
    choice = _first_choice(data)
    raw_usage = data.get("usage")
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    details = usage.get("completion_tokens_details")
    raw_message = choice.get("message")
    raw_content = raw_message.get("content") if isinstance(raw_message, dict) else None
    return OpenRouterDiagnostics(
        model=_safe_identifier(data.get("model"), _MODEL_SLUG, api_key),
        generation_id=_safe_identifier(data.get("id"), _GENERATION_ID, api_key),
        finish_reason=_finish_label(choice.get("finish_reason"), _FINISH_REASONS),
        native_finish_reason=_finish_label(choice.get("native_finish_reason"), _NATIVE_REASONS),
        completion_tokens=_token_count(usage.get("completion_tokens")),
        reasoning_tokens=_token_count(details.get("reasoning_tokens")) if isinstance(details, dict) else None,
        content_characters=len(raw_content) if isinstance(raw_content, str) else None,
        prompt_tokens=_token_count(usage.get("prompt_tokens")),
        total_tokens=_token_count(usage.get("total_tokens")),
        http_status=http_status,
        cost=_reported_cost(usage.get("cost")),
    )


def _read_completion(
    data: dict[str, Any], api_key: str, http_status: int, retry_after: str | None
) -> _Completion:
    diagnostics = _response_diagnostics(data, api_key, http_status)
    choice = _first_choice(data)
    if data.get("error") is not None or choice.get("error") or diagnostics.finish_reason == "error":
        _require_free_response(data, diagnostics, allow_missing=True)
        raise _api_error(
            http_status, data.get("error") or choice.get("error"),
            retry_after=retry_after, diagnostics=diagnostics,
        )
    _require_free_response(data, diagnostics)
    raw_message = choice.get("message")
    message = raw_message if isinstance(raw_message, dict) else {}
    raw_content = message.get("content")
    content = raw_content if isinstance(raw_content, str) else ""

    if diagnostics.finish_reason == "length" or diagnostics.native_finish_reason in _TOKEN_LIMIT_REASONS:
        return _Completion(diagnostics, content=content, issue="truncated")
    if diagnostics.finish_reason == "content_filter" or message.get("refusal"):
        return _Completion(diagnostics, content=content, issue="refused_or_filtered")
    if not choice:
        return _Completion(diagnostics, issue="missing_choice")
    if diagnostics.finish_reason != "stop":
        return _Completion(diagnostics, content=content, issue="unexpected_finish_reason")
    if not isinstance(raw_message, dict):
        return _Completion(diagnostics, issue="missing_message")
    if raw_content is not None and not isinstance(raw_content, str):
        return _Completion(diagnostics, issue="non_text_content")
    if not content.strip():
        if diagnostics.reasoning_tokens or message.get("reasoning") or message.get("reasoning_details"):
            return _Completion(diagnostics, issue="reasoning_only")
        issue = "empty_content" if isinstance(raw_content, str) else "missing_content"
        return _Completion(diagnostics, issue=issue)
    return _Completion(diagnostics, content=content)


def _free_request_body(instructions: str, sanitized_input: str, schema: dict[str, Any]) -> bytes:
    payload = {
        "model": _FREE_MODEL,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": sanitized_input},
        ],
        "provider": {
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": {"prompt": 0, "completion": 0, "request": 0},
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "fault_assessment", "strict": True, "schema": schema},
        },
        "max_completion_tokens": _MAX_COMPLETION_TOKENS,
        "stream": False,
    }
    return json.dumps(payload).encode("utf-8")


def _request_free_completion(body: bytes, *, api_key: str) -> _Completion:
    request = Request(
        _CHAT_URL,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    request.add_unredirected_header("Authorization", f"Bearer {api_key}")
    status = None
    try:
        with _OPENER.open(request, timeout=_TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", 200)
            headers = getattr(response, "headers", None)
            retry_after = headers.get("Retry-After") if headers is not None else None
            data = _decode_response(response.read(_RESPONSE_LIMIT + 1))
    except HTTPError as error:
        retry_after = error.headers.get("Retry-After") if error.headers is not None else None
        with error:
            try:
                details = _decode_response(error.read(_RESPONSE_LIMIT + 1))
            except (OSError, HTTPException):
                details = None
        data = details or {}
        diagnostics = _response_diagnostics(data, api_key, error.code)
        _require_free_response(data, diagnostics, allow_missing=True)
        raise _api_error(
            error.code, data.get("error"), retry_after=retry_after,
            diagnostics=diagnostics,
        ) from None
    except (OSError, HTTPException):
        raise OpenRouterError(
            "The OpenRouter HTTPS request could not be completed.", issue="transport_error",
            diagnostics=_response_diagnostics({}, api_key, status),
        ) from None

    if data is None:
        raise OpenRouterError(
            "OpenRouter returned an invalid JSON response.", issue="invalid_response_json",
            diagnostics=_response_diagnostics({}, api_key, status),
        )
    return _read_completion(data, api_key, status, retry_after)
