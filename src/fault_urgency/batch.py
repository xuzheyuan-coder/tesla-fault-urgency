"""Send one free OpenRouter request per CSV record and checkpoint each attempt."""

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
from io import StringIO
from math import ceil
from pathlib import Path
from time import sleep, time
from typing import Literal

from pydantic import ValidationError

from .ai_classifier import (
    _PreparedBatchAssessment,
    _prepare_batch_assessment,
    _resolve_batch_assessment,
)
from .batch_reports import write_batch_reports
from .models import FaultMessage, UrgencyDecision, UrgencyLevel
from .openrouter import OpenRouterDiagnostics, OpenRouterError, _read_api_key, _request_free_completion

DEFAULT_CHECKPOINT = Path("dist/fault-batch-per-message.sqlite3")
MIN_REQUEST_INTERVAL = 6
MAX_ATTEMPTS_PER_MESSAGE = 2
_DEFAULT_COOLDOWN = 60
_READY = frozenset({"pending", "retry_pending"})
_FINISHED = frozenset({"classified", "validation_failed", "incomplete_output", "request_failed"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS batch_items (
    batch_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    local_safety_urgent INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    decision_json TEXT,
    ai_decision_json TEXT,
    issues_json TEXT NOT NULL DEFAULT '[]',
    human_review_required INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (batch_id, message_id)
);
CREATE TABLE IF NOT EXISTS batch_attempts (
    batch_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL,
    cooldown_until REAL,
    diagnostics_json TEXT,
    issues_json TEXT NOT NULL DEFAULT '[]',
    error_message TEXT,
    zero_cost_verified INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (batch_id, message_id, attempt_number)
);
"""


class BatchInputError(ValueError):
    """A batch argument or CSV validation error that is safe to display."""


@dataclass(frozen=True, slots=True)
class BatchSummary:
    mode: Literal["dry_run", "execute"]
    batch_id: str
    records: int
    planned_initial_requests: int
    initial_requests_remaining: int
    request_budget: int
    requests_sent: int
    requests_remaining: int
    attempts_recorded: int
    retry_pending: int
    unconfirmed_requests: int
    completed: int
    pending: int
    classified: int
    validation_failed: int
    incomplete_output: int
    request_failed: int
    human_review_required: int
    zero_cost_verified: int
    retry_after_seconds: int | None
    pause_reason: str | None
    output_directory: str | None


@dataclass(frozen=True, slots=True)
class _Row:
    message_id: str
    assessment: _PreparedBatchAssessment


@dataclass(frozen=True, slots=True)
class _AttemptOutcome:
    status: str
    issues: tuple[str, ...]
    diagnostics: OpenRouterDiagnostics | None = None
    ai_decision: UrgencyDecision | None = None
    decision: UrgencyDecision | None = None
    error_message: str | None = None
    pause_reason: str | None = None
    cooldown_until: float | None = None


def _load_batch(path: Path) -> tuple[str, list[_Row]]:
    try:
        raw = path.read_bytes()
        reader = csv.DictReader(StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields):
            raise BatchInputError("CSV headers must be unique.")
        if not {"message_id", "message_text"}.issubset(fields):
            raise BatchInputError("CSV requires message_id and message_text columns.")
        rows = []
        seen = set()
        for record in reader:
            line = reader.line_num
            if None in record or None in record.values():
                raise BatchInputError(f"CSV row ending on line {line} does not match the headers.")
            try:
                message = FaultMessage(
                    message_id=record["message_id"],
                    message_text=record["message_text"],
                    alert_code=record.get("vehicle_alert_code", record.get("alert_code")),
                    customer_reported_urgency=record.get("customer_reported_urgency"),
                )
            except ValidationError:
                raise BatchInputError(f"Invalid fault message on CSV line {line}.") from None
            if message.message_id != record["message_id"]:
                raise BatchInputError(f"Message ID has surrounding whitespace on CSV line {line}.")
            if message.message_id in seen:
                raise BatchInputError(f"Duplicate message ID on CSV line {line}.")
            seen.add(message.message_id)
            rows.append(_Row(message.message_id, _prepare_batch_assessment(message)))
    except (OSError, UnicodeError, csv.Error):
        raise BatchInputError("Unable to read the CSV as a UTF-8 fault-message file.") from None
    if not rows:
        raise BatchInputError("CSV does not contain any fault messages.")
    return hashlib.sha256(raw).hexdigest(), rows


def _seed_items(db: sqlite3.Connection, batch_id: str, rows: list[_Row]) -> None:
    with db:
        db.executemany(
            """INSERT INTO batch_items (batch_id, position, message_id, local_safety_urgent)
               VALUES (?, ?, ?, ?) ON CONFLICT(batch_id, message_id) DO NOTHING""",
            (
                (batch_id, position, row.message_id,
                 row.assessment.local.urgency == UrgencyLevel.SAFETY_URGENT)
                for position, row in enumerate(rows)
            ),
        )


def _item(db: sqlite3.Connection, batch_id: str, message_id: str) -> sqlite3.Row:
    return db.execute(
        "SELECT status, attempt_count FROM batch_items WHERE batch_id = ? AND message_id = ?",
        (batch_id, message_id),
    ).fetchone()


def _cooldown_until(db: sqlite3.Connection) -> float:
    return db.execute("SELECT MAX(cooldown_until) FROM batch_attempts").fetchone()[0] or 0


def _claim_attempt(
    db: sqlite3.Connection, batch_id: str, message_id: str
) -> int | Literal["cooldown"] | None:
    while True:
        with db:
            # Commit the reservation before performing any network I/O.
            db.execute("BEGIN IMMEDIATE")
            item = _item(db, batch_id, message_id)
            if item["status"] not in _READY:
                return None
            now = time()
            if _cooldown_until(db) > now:
                return "cooldown"
            last_start = db.execute("SELECT MAX(started_at) FROM batch_attempts").fetchone()[0]
            wait = MIN_REQUEST_INTERVAL + last_start - now if last_start is not None else 0
            if wait <= 0:
                number = item["attempt_count"] + 1
                db.execute(
                    """UPDATE batch_items SET status = 'in_flight', attempt_count = ?,
                       human_review_required = 1, issues_json = '["unconfirmed_request"]'
                       WHERE batch_id = ? AND message_id = ?""",
                    (number, batch_id, message_id),
                )
                db.execute(
                    """INSERT INTO batch_attempts
                       (batch_id, message_id, attempt_number, started_at, status)
                       VALUES (?, ?, ?, ?, 'in_flight')""",
                    (batch_id, message_id, number, now),
                )
                return number
        sleep(wait)


def _request_failure(error: OpenRouterError, attempt_number: int) -> _AttemptOutcome:
    must_wait = error.issue in {"rate_limited", "free_model_unavailable"}
    transient = error.issue in {"transport_error", "invalid_response_json"} or (
        error.issue == "request_error" and error.status_code in {408, 500, 502, 504}
    )
    retry = attempt_number < MAX_ATTEMPTS_PER_MESSAGE and (must_wait or transient)
    cooldown = None
    if must_wait:
        delay = error.retry_delay if error.retry_delay is not None else _DEFAULT_COOLDOWN
        cooldown = time() + max(delay, MIN_REQUEST_INTERVAL)
    message = str(error)
    return _AttemptOutcome(
        "retry_pending" if retry else "request_failed",
        (error.issue,),
        diagnostics=error.diagnostics,
        error_message=message,
        pause_reason=message if must_wait or not transient else None,
        cooldown_until=cooldown,
    )


def _send(row: _Row, attempt_number: int, api_key: str) -> _AttemptOutcome:
    try:
        completion = _request_free_completion(row.assessment.body, api_key=api_key)
    except OpenRouterError as error:
        return _request_failure(error, attempt_number)
    assessment = _resolve_batch_assessment(row.assessment, completion, api_key=api_key)
    return _AttemptOutcome(
        assessment.status, assessment.issues, completion.diagnostics,
        assessment.ai_decision, assessment.decision,
    )


def _save_outcome(
    db: sqlite3.Connection, batch_id: str, row: _Row, number: int, result: _AttemptOutcome
) -> None:
    diagnostics = json.dumps(asdict(result.diagnostics)) if result.diagnostics else None
    issues = json.dumps(result.issues)
    review = result.status != "retry_pending" and (
        result.decision is None or result.decision.human_review_required
    )
    with db:
        db.execute(
            """UPDATE batch_attempts SET finished_at = ?, status = ?, cooldown_until = ?,
               diagnostics_json = ?, issues_json = ?, error_message = ?, zero_cost_verified = ?
               WHERE batch_id = ? AND message_id = ? AND attempt_number = ?""",
            (
                time(), "request_error" if result.error_message else result.status, result.cooldown_until,
                diagnostics, issues, result.error_message, result.error_message is None,
                batch_id, row.message_id, number,
            ),
        )
        db.execute(
            """UPDATE batch_items SET status = ?, decision_json = ?, ai_decision_json = ?,
               issues_json = ?, human_review_required = ? WHERE batch_id = ? AND message_id = ?""",
            (
                result.status, result.decision.model_dump_json() if result.decision else None,
                result.ai_decision.model_dump_json() if result.ai_decision else None,
                issues, review, batch_id, row.message_id,
            ),
        )


def _execute(
    db: sqlite3.Connection, batch_id: str, rows: list[_Row], budget: int, env_file: str | Path
) -> tuple[int, str | None]:
    sent = 0
    api_key = None
    cooldown = "OpenRouter cooldown is still active."
    for row in rows:
        while sent < budget and _item(db, batch_id, row.message_id)["status"] in _READY:
            if _cooldown_until(db) > time():
                return sent, cooldown
            if api_key is None:
                try:
                    api_key = _read_api_key(env_file)
                except OpenRouterError as error:
                    return sent, str(error)
            number = _claim_attempt(db, batch_id, row.message_id)
            if number == "cooldown":
                return sent, cooldown
            if number is None:
                break
            sent += 1
            result = _send(row, number, api_key)
            _save_outcome(db, batch_id, row, number, result)
            if result.pause_reason:
                return sent, result.pause_reason
        if sent >= budget:
            break
    return sent, None


def _summarize(
    db: sqlite3.Connection | None,
    batch_id: str,
    count: int,
    *,
    budget: int = 0,
    sent: int = 0,
    pause: str | None = None,
    output_directory: Path | None = None,
) -> BatchSummary:
    items = []
    verified = cooldown = 0
    if db is not None:
        items = db.execute("SELECT * FROM batch_items WHERE batch_id = ?", (batch_id,)).fetchall()
        verified = db.execute(
            "SELECT COALESCE(SUM(zero_cost_verified), 0) FROM batch_attempts WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()[0]
        cooldown = max(0, ceil(_cooldown_until(db) - time()))
    status = Counter(item["status"] for item in items)
    first_remaining = count - sum(item["attempt_count"] > 0 for item in items)
    completed = sum(status[key] for key in _FINISHED)
    return BatchSummary(
        mode="execute" if budget else "dry_run",
        batch_id=batch_id,
        records=count,
        planned_initial_requests=count,
        initial_requests_remaining=first_remaining,
        request_budget=budget,
        requests_sent=sent,
        requests_remaining=first_remaining + status["retry_pending"],
        attempts_recorded=sum(item["attempt_count"] for item in items),
        retry_pending=status["retry_pending"],
        unconfirmed_requests=status["in_flight"],
        completed=completed,
        pending=count - completed,
        classified=status["classified"],
        validation_failed=status["validation_failed"],
        incomplete_output=status["incomplete_output"],
        request_failed=status["request_failed"],
        human_review_required=sum(item["human_review_required"] for item in items),
        zero_cost_verified=verified,
        retry_after_seconds=cooldown or None,
        pause_reason=pause,
        output_directory=str(output_directory) if output_directory else None,
    )


def run_csv_batch(
    source: str | Path,
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    max_requests: int | None = None,
    env_file: str | Path = ".env",
) -> BatchSummary:
    """Read and plan a batch. A positive request budget explicitly enables execution."""
    if max_requests is not None and (type(max_requests) is not int or max_requests < 1):
        raise BatchInputError("The request budget must be a positive integer.")
    source, checkpoint = Path(source), Path(checkpoint).resolve()
    if source.resolve() == checkpoint:
        raise BatchInputError("The source CSV and checkpoint must use different files.")
    batch_id, rows = _load_batch(source)
    if max_requests is None:
        if not checkpoint.exists():
            return _summarize(None, batch_id, len(rows))
        with closing(sqlite3.connect(checkpoint.as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            has_items = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'batch_items'"
            ).fetchone()
            return _summarize(db if has_items else None, batch_id, len(rows))

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    report_dir = checkpoint.with_suffix(".reports") / batch_id[:16]
    with closing(sqlite3.connect(checkpoint)) as db:
        db.row_factory = sqlite3.Row
        db.executescript(_SCHEMA)
        _seed_items(db, batch_id, rows)
        sent, pause = _execute(db, batch_id, rows, max_requests, env_file)
        with db:
            db.execute("BEGIN")
            summary = _summarize(
                db, batch_id, len(rows), budget=max_requests, sent=sent, pause=pause,
                output_directory=report_dir,
            )
            write_batch_reports(db, batch_id, report_dir, asdict(summary))
            return summary
