"""Export the current batch snapshot and its attempt audit without source text."""

import csv
import json
import sqlite3
from collections import Counter
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile

_OUTPUT_FIELDS = ("urgency", "fix_path", "reason", "matched_alert_code", "matched_symptoms")
_USAGE_FIELDS = (
    "model", "generation_id", "http_status", "finish_reason", "native_finish_reason",
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "cost",
)
_ATTEMPT_FIELDS = (
    "message_id", "attempt_number", "status", "started_at", "finished_at", "issues",
    "error_message", "zero_cost_verified", *_USAGE_FIELDS,
)
_RESULT_FIELDS = (
    "message_id", "status", "attempt_count", "issues", "local_safety_urgent",
    "human_review_required", *_OUTPUT_FIELDS, *(f"ai_{field}" for field in _OUTPUT_FIELDS),
    "ai_human_review_required", *_USAGE_FIELDS,
)
_VALIDATION_FAILURES = {"validation_failed", "incomplete_output"}


def _write_report(path: Path, write) -> None:
    # Preserve the prior report when an export is interrupted before it is complete.
    with NamedTemporaryFile(
        mode="w", encoding="utf-8-sig" if path.suffix == ".csv" else "utf-8",
        newline="", dir=path.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            write(stream)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, names: tuple[str, ...], rows: list[dict]) -> None:
    def emit(stream):
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)

    _write_report(path, emit)


def _safe_cell(value: str) -> str:
    if value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _decision_fields(serialized: str | None, prefix: str = "") -> dict:
    if serialized is None:
        return {}
    decision = json.loads(serialized)
    values = {f"{prefix}{field}": decision[field] for field in _OUTPUT_FIELDS}
    values[f"{prefix}reason"] = _safe_cell(decision["reason"])
    values[f"{prefix}matched_symptoms"] = json.dumps(decision["matched_symptoms"], ensure_ascii=False)
    if prefix:
        values[f"{prefix}human_review_required"] = decision["human_review_required"]
    return values


def write_batch_reports(
    db: sqlite3.Connection, batch_id: str, directory: Path, summary: dict
) -> None:
    """Write one result per input row plus separate review, validation, and attempt reports."""
    attempts = []
    latest_usage = {}
    for row in db.execute(
        """SELECT * FROM batch_attempts WHERE batch_id = ? ORDER BY started_at, attempt_number""",
        (batch_id,),
    ):
        diagnostics = json.loads(row["diagnostics_json"]) if row["diagnostics_json"] else {}
        usage = {name: diagnostics.get(name) for name in _USAGE_FIELDS}
        latest_usage[row["message_id"]] = usage
        attempt = {name: row[name] for name in _ATTEMPT_FIELDS if name not in usage and name != "issues"}
        attempt["issues"] = " | ".join(json.loads(row["issues_json"]))
        attempt.update(usage)
        attempts.append(attempt)

    results = []
    issues = Counter()
    for row in db.execute(
        "SELECT * FROM batch_items WHERE batch_id = ? ORDER BY position", (batch_id,)
    ):
        row_issues = json.loads(row["issues_json"])
        issues.update(row_issues)
        results.append({
            "message_id": row["message_id"],
            "status": row["status"],
            "attempt_count": row["attempt_count"],
            "issues": " | ".join(row_issues),
            "local_safety_urgent": bool(row["local_safety_urgent"]),
            "human_review_required": bool(row["human_review_required"]),
            **_decision_fields(row["decision_json"]),
            **_decision_fields(row["ai_decision_json"], "ai_"),
            **latest_usage.get(row["message_id"], {}),
        })

    validation_failures = [row for row in results if row["status"] in _VALIDATION_FAILURES]
    review = [row for row in results if row["human_review_required"]]
    reported_costs = [Decimal(row["cost"]) for row in attempts if row["cost"] is not None]
    statistics = {
        **summary,
        "by_status": dict(Counter(row["status"] for row in results)),
        "by_urgency": dict(Counter(row["urgency"] for row in results if row.get("urgency"))),
        "issues_by_record": dict(issues),
        "validation_failure_records": len(validation_failures),
        "reported_total_cost": str(sum(reported_costs)) if reported_costs else None,
        "attempts_with_reported_cost": len(reported_costs),
        "models": dict(Counter(row["model"] for row in attempts if row["model"])),
        "reported_token_totals": {
            field: sum(values) if (values := [
                row[field] for row in attempts if row[field] is not None
            ]) else None
            for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    _write_csv(directory / "results.csv", _RESULT_FIELDS, results)
    _write_csv(directory / "validation_failures.csv", _RESULT_FIELDS, validation_failures)
    _write_csv(directory / "human_review.csv", _RESULT_FIELDS, review)
    _write_csv(directory / "attempts.csv", _ATTEMPT_FIELDS, attempts)
    _write_report(
        directory / "summary.json",
        lambda stream: json.dump(statistics, stream, indent=2, ensure_ascii=False),
    )
