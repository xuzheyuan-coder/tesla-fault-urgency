import csv
import json
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Event
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

from fault_urgency import batch, openrouter
from fault_urgency.__main__ import main
from fault_urgency.ai_classifier import _BatchDecision
from fault_urgency.batch import BatchInputError, run_csv_batch

_KEY = "mock-batch-key"
_TOUCHSCREEN = ("UI_a004", "The touchscreen freezes while opening navigation.")
_HEATER = ("HVAC_w003", "The heater blows cold air.")
_LATCH = ("CP_a004", "The charge port latch won't close.")
_SAFETY = ("NONE", "The brake pedal slowly sinks to the floor.")
_SPARSE = ("", "Something feels different.")
_ZERO_CSV = (
    Path(__file__).resolve().parents[1] / "Tesla刹车问题"
    / "Overnight Fault Message Batch (16 November 2026).csv"
)
_VALID_DECISION = {
    "urgency": "routine",
    "fix_path": "routine_service_appointment",
    "reason": "The available evidence supports routine service.",
    "matched_alert_code": None,
    "matched_symptoms": [],
    "human_review_required": False,
}


@dataclass
class _Clock:
    now: float = 100

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _response(content: str | None = None, *, finish="stop", **overrides) -> BytesIO:
    envelope = {
        "id": "gen-mock-batch",
        "model": "test-provider/model:free",
        "usage": {
            "cost": 0, "prompt_tokens": 240, "completion_tokens": 30, "total_tokens": 270,
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
        "choices": [{
            "finish_reason": finish,
            "message": {"content": json.dumps(_VALID_DECISION) if content is None else content},
        }],
    }
    envelope.update(overrides)
    return BytesIO(json.dumps(envelope).encode())


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock, _Clock]:
    clock = _Clock()
    dotenv = Mock(return_value={"OPENROUTER_API_KEY": _KEY})
    post = Mock(side_effect=lambda *_args, **_kwargs: _response())
    monkeypatch.setattr(openrouter, "dotenv_values", dotenv)
    monkeypatch.setattr(openrouter._OPENER, "open", post)
    monkeypatch.setattr(batch, "time", lambda: clock.now)
    monkeypatch.setattr(batch, "sleep", clock.sleep)
    return post, dotenv, clock


def _csv(path: Path, *rows: tuple[str, tuple[str, str]]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["message_id", "vehicle_alert_code", "message_text", "customer_name", "vin_last6"])
        writer.writerows((key, code, text, "Private Name", "987654") for key, (code, text) in rows)
    return path


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _output(summary, name: str) -> list[dict]:
    return _read_csv(Path(summary.output_directory) / name)


def _posted_ids(post: Mock) -> list[str]:
    return [
        json.loads(json.loads(call.args[0].data)["messages"][1]["content"])["message_id"]
        for call in post.call_args_list
    ]


def _stored_status(path: Path) -> dict[str, str]:
    with closing(sqlite3.connect(path)) as db:
        return dict(db.execute("SELECT message_id, status FROM batch_items"))


def test_csv_cli_defaults_to_offline_preview(tmp_path, capsys, offline):
    post, dotenv, _ = offline
    source = _csv(
        tmp_path / "faults.csv", ("case-1", _SAFETY), ("case-2", _SPARSE),
        ("case-3", _TOUCHSCREEN), ("case-4", _TOUCHSCREEN),
    )
    checkpoint = tmp_path / "new" / "batch.sqlite3"

    assert main(["--csv", str(source), "--checkpoint", str(checkpoint)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["mode"] == "dry_run"
    assert summary["records"] == summary["planned_initial_requests"] == summary["requests_remaining"] == 4
    assert summary["requests_sent"] == 0
    assert not checkpoint.parent.exists()
    post.assert_not_called()
    dotenv.assert_not_called()


@pytest.mark.parametrize("flags", [("--execute",), ("--max-requests", "5")])
def test_cli_requires_explicit_execution_and_a_budget(tmp_path, flags, offline):
    with pytest.raises(SystemExit) as error:
        main(["--csv", str(tmp_path / "unused.csv"), *flags])
    assert error.value.code == 2
    offline[0].assert_not_called()
    offline[1].assert_not_called()


@pytest.mark.parametrize("budget", [0, -1, True])
def test_budget_must_be_a_positive_integer(tmp_path, budget):
    with pytest.raises(BatchInputError, match="positive integer"):
        run_csv_batch(tmp_path / "unused.csv", max_requests=budget)


@pytest.mark.parametrize(("contents", "error"), [
    ("message_text\nhi\n", "requires message_id"),
    ("message_id,message_text\na,One\na,Two\n", "Duplicate message ID"),
    ("message_id,message_text\na,One\nb\n", "does not match the headers"),
    ("message_id,message_text\na,One\nb,\n", "Invalid fault message"),
    ('message_id,message_text\n" a ",One\n', "surrounding whitespace"),
])
def test_batch_checks_every_row_before_sending_or_creating_state(tmp_path, offline, contents, error):
    source = tmp_path / "faults.csv"
    source.write_text(contents, encoding="utf-8")
    checkpoint = tmp_path / "batch.sqlite3"

    with pytest.raises(BatchInputError, match=error):
        run_csv_batch(source, checkpoint=checkpoint, max_requests=5)

    assert not checkpoint.exists()
    offline[0].assert_not_called()
    offline[1].assert_not_called()


def test_budget_resumes_distinct_items_and_sends_local_safety_and_sparse_reports(
    tmp_path, offline, capsys
):
    post, dotenv, clock = offline
    private_text = f"Contact alice@example.test. {_TOUCHSCREEN[1]}"
    source = _csv(
        tmp_path / "faults.csv",
        ("case-1", _SAFETY), ("case-2", _SPARSE), ("case-3", _TOUCHSCREEN),
        ("case-4", (_TOUCHSCREEN[0], private_text)),
    )
    checkpoint = tmp_path / "state.sqlite3"
    starts = []

    def respond(request, **_kwargs):
        starts.append(clock.now)
        sanitized = json.loads(json.loads(request.data)["messages"][1]["content"])
        assert _stored_status(checkpoint)[sanitized["message_id"]] == "in_flight"
        return _response()

    post.side_effect = respond
    assert main([
        "--csv", str(source), "--checkpoint", str(checkpoint), "--execute", "--max-requests", "2",
    ]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["completed"] == 2 and first["requests_remaining"] == 2
    assert first["requests_sent"] == first["zero_cost_verified"] == 2

    dotenv.reset_mock()
    before = checkpoint.read_bytes()
    preview = run_csv_batch(source, checkpoint=checkpoint)
    assert preview.completed == preview.initial_requests_remaining == 2
    assert checkpoint.read_bytes() == before
    dotenv.assert_not_called()

    resumed = run_csv_batch(source, checkpoint=checkpoint, max_requests=2)
    records = _output(resumed, "results.csv")
    assert resumed.classified == resumed.completed == 4
    assert resumed.requests_sent == 2 and resumed.zero_cost_verified == 4
    assert _posted_ids(post) == ["case-1", "case-2", "case-3", "case-4"]
    assert starts == [100, 106, 112, 118]
    assert records[0]["local_safety_urgent"] == "True"
    assert records[0]["ai_urgency"] == "routine"
    assert records[0]["urgency"] == "safety_urgent"
    assert records[0]["fix_path"] == "stop_driving_and_tow"
    assert records[1]["urgency"] == "soon"
    assert records[2]["fix_path"] == records[3]["fix_path"] == "remote_diagnosis_or_ota"
    assert [row["message_id"] for row in _output(resumed, "human_review.csv")] == ["case-1", "case-2"]

    for call in post.call_args_list:
        body = call.args[0].data
        payload = json.loads(body)
        assert payload["model"] == "openrouter/free"
        assert payload["provider"] == {
            "allow_fallbacks": False, "require_parameters": True,
            "max_price": {"prompt": 0, "completion": 0, "request": 0},
        }
        schema = payload["response_format"]["json_schema"]["schema"]
        assert set(schema["required"]) == set(_VALID_DECISION)
        assert b"alice@example.test" not in body
        assert b"Private Name" not in body
        assert b"987654" not in body
    for value in (_KEY, private_text, "alice@example.test", "Private Name"):
        assert value.encode() not in checkpoint.read_bytes()
    dotenv.reset_mock()

    replay = run_csv_batch(source, checkpoint=checkpoint, max_requests=5)
    assert replay.completed == 4 and replay.requests_sent == 0
    assert post.call_count == 4
    dotenv.assert_not_called()


def test_interruption_preserves_success_and_does_not_replay_unconfirmed_attempts(tmp_path, offline):
    post, _, _ = offline
    source = _csv(
        tmp_path / "faults.csv", ("case-1", _TOUCHSCREEN), ("case-2", _HEATER), ("case-3", _LATCH),
    )
    checkpoint = tmp_path / "batch.sqlite3"
    post.side_effect = [_response(), KeyboardInterrupt(), _response()]

    with pytest.raises(KeyboardInterrupt):
        run_csv_batch(source, checkpoint=checkpoint, max_requests=5)

    assert _stored_status(checkpoint) == {
        "case-1": "classified", "case-2": "in_flight", "case-3": "pending",
    }
    resumed = run_csv_batch(source, checkpoint=checkpoint, max_requests=5)
    assert resumed.completed == 2 and resumed.pending == 1
    assert resumed.requests_sent == resumed.unconfirmed_requests == 1
    assert resumed.requests_remaining == 0
    assert _posted_ids(post) == ["case-1", "case-2", "case-3"]
    assert _output(resumed, "human_review.csv")[0]["issues"] == "unconfirmed_request"


def test_shared_database_cannot_send_an_in_flight_message_twice(tmp_path, offline):
    post, _, _ = offline
    source = _csv(tmp_path / "faults.csv", ("case-1", _TOUCHSCREEN))
    checkpoint = tmp_path / "batch.sqlite3"
    sent = Event()
    release = Event()

    def respond(*_args, **_kwargs):
        sent.set()
        assert release.wait(timeout=5)
        return _response()

    post.side_effect = respond
    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = executor.submit(run_csv_batch, source, checkpoint=checkpoint, max_requests=1)
        try:
            assert sent.wait(timeout=5)
            other = run_csv_batch(source, checkpoint=checkpoint, max_requests=1)
            assert other.requests_sent == 0 and other.unconfirmed_requests == 1
        finally:
            release.set()
        assert worker.result(timeout=5).completed == 1
    post.assert_called_once()


def test_missing_key_does_not_consume_attempt_or_budget(tmp_path, offline):
    post, dotenv, _ = offline
    source = _csv(tmp_path / "faults.csv", ("case-1", _SAFETY))
    checkpoint = tmp_path / "batch.sqlite3"
    dotenv.return_value = {}

    paused = run_csv_batch(source, checkpoint=checkpoint, max_requests=1)

    assert paused.requests_sent == paused.attempts_recorded == 0
    assert paused.initial_requests_remaining == paused.pending == 1
    assert "Set OPENROUTER_API_KEY" in paused.pause_reason
    assert _output(paused, "attempts.csv") == []
    post.assert_not_called()
    dotenv.return_value = {"OPENROUTER_API_KEY": _KEY}
    resumed = run_csv_batch(source, checkpoint=checkpoint, max_requests=1)
    assert resumed.requests_sent == resumed.completed == 1


@pytest.mark.parametrize(("content", "issue"), [
    ("{", "malformed_json"),
    (json.dumps({k: v for k, v in _VALID_DECISION.items() if k != "reason"}), "missing_fields"),
    (json.dumps({k: v for k, v in _VALID_DECISION.items() if k != "matched_symptoms"}), "missing_fields"),
    (json.dumps({**_VALID_DECISION, "urgency": "low"}), "invalid_enum"),
    (json.dumps({**_VALID_DECISION, "fix_path": "manual"}), "invalid_enum"),
    (json.dumps({**_VALID_DECISION, "urgency": "safety_urgent"}), "schema_validation_failed"),
    (json.dumps({**_VALID_DECISION, "reason": "   "}), "schema_validation_failed"),
    (json.dumps({**_VALID_DECISION, "human_review_required": "false"}), "schema_validation_failed"),
    (json.dumps({**_VALID_DECISION, "matched_alert_code": "unrecognized"}), "unexpected_alert_code"),
    (json.dumps({**_VALID_DECISION, "reason": _KEY}), "unsafe_output"),
])
def test_violations_record_exact_issue_with_no_fabricated_decision(tmp_path, offline, content, issue):
    post, _, _ = offline
    source = _csv(tmp_path / "faults.csv", ("case-1", _SAFETY))
    checkpoint = tmp_path / "batch.sqlite3"
    post.side_effect = [_response(content)]

    result = run_csv_batch(source, checkpoint=checkpoint, max_requests=5)

    assert result.validation_failed == result.human_review_required == 1
    assert result.classified == 0 and result.zero_cost_verified == 1
    failed = _output(result, "validation_failures.csv")
    assert len(failed) == 1
    assert failed == _output(result, "results.csv") == _output(result, "human_review.csv")
    row = failed[0]
    assert row["issues"] == issue
    assert row["urgency"] == row["fix_path"] == row["reason"] == row["ai_reason"] == ""
    assert row["local_safety_urgent"] == "True"
    assert _KEY.encode() not in checkpoint.read_bytes()
    post.assert_called_once()


def test_truncated_json_is_parsed_and_reported_without_a_decision(tmp_path, offline):
    post, _, _ = offline
    post.side_effect = [_response("{", finish="length")]
    source = _csv(tmp_path / "faults.csv", ("case-1", _TOUCHSCREEN))

    result = run_csv_batch(source, checkpoint=tmp_path / "batch.sqlite3", max_requests=5)

    assert result.incomplete_output == 1 and result.classified == 0
    failed = _output(result, "validation_failures.csv")[0]
    assert failed["issues"] == "truncated | malformed_json"
    assert failed["ai_reason"] == "" and failed["reason"] == ""
    post.assert_called_once()


def test_retry_uses_remaining_budget_and_is_limited_durably(tmp_path, offline):
    post, _, _ = offline
    source = _csv(tmp_path / "faults.csv", ("case-1", _LATCH), ("case-2", _HEATER))
    checkpoint = tmp_path / "batch.sqlite3"
    post.side_effect = [URLError("temporary"), URLError("temporary"), _response()]

    first = run_csv_batch(source, checkpoint=checkpoint, max_requests=1)
    assert first.retry_pending == first.requests_sent == 1
    assert first.initial_requests_remaining == 1 and first.requests_remaining == 2
    assert first.completed == 0

    resumed = run_csv_batch(source, checkpoint=checkpoint, max_requests=5)
    assert resumed.requests_sent == 2 and resumed.attempts_recorded == 3
    assert resumed.completed == 2 and resumed.classified == resumed.request_failed == 1
    history = _output(resumed, "attempts.csv")
    assert [(row["message_id"], row["attempt_number"]) for row in history] == [
        ("case-1", "1"), ("case-1", "2"), ("case-2", "1"),
    ]
    assert history[0]["status"] == "request_error" and history[0]["cost"] == ""
    assert history[-1]["status"] == "classified" and history[-1]["cost"] == "0"
    assert _posted_ids(post) == ["case-1", "case-1", "case-2"]
    replay = run_csv_batch(source, checkpoint=checkpoint, max_requests=5)
    assert replay.requests_sent == 0 and post.call_count == 3


@pytest.mark.parametrize(("status", "reason"), [(429, "request limit"), (503, "free model")])
def test_rate_and_free_model_errors_pause_and_honor_retry_after(tmp_path, offline, status, reason):
    post, dotenv, clock = offline
    source = _csv(tmp_path / "faults.csv", ("case-1", _TOUCHSCREEN), ("case-2", _HEATER))
    checkpoint = tmp_path / "batch.sqlite3"
    error = HTTPError("https://example.test", status, "limited", {"Retry-After": "25"}, BytesIO(b"{}"))
    post.side_effect = [error, _response(), _response()]

    first = run_csv_batch(source, checkpoint=checkpoint, max_requests=5)
    assert first.requests_sent == first.retry_pending == 1
    assert first.retry_after_seconds == 25 and reason in first.pause_reason
    assert _output(first, "attempts.csv")[0]["http_status"] == str(status)
    dotenv.reset_mock()

    waiting = run_csv_batch(source, checkpoint=checkpoint, max_requests=5)
    assert waiting.requests_sent == 0 and waiting.retry_after_seconds == 25
    dotenv.assert_not_called()

    clock.now += 25
    resumed = run_csv_batch(source, checkpoint=checkpoint, max_requests=2)
    assert resumed.completed == resumed.requests_sent == 2
    assert resumed.retry_after_seconds is None and resumed.pause_reason is None
    assert _posted_ids(post) == ["case-1", "case-1", "case-2"]


@pytest.mark.parametrize(("envelope", "reason", "cost"), [
    ({"usage": {"cost": 0.0003}}, "zero usage cost", "0.0003"),
    ({"model": "test-provider/paid"}, "identify a free model", "0"),
    ({"usage": {"prompt_tokens": 8}}, "zero usage cost", ""),
    ({"usage": {"cost": 0.0003}, "error": {"code": 429}}, "zero usage cost", "0.0003"),
    ({"model": "test-provider/paid", "error": {"code": 429}}, "identify a free model", "0"),
    ({"usage": {"cost": 0, "is_byok": True}, "error": {"code": 429}}, "bring-your-own-key", "0"),
])
def test_failed_cost_check_stops_before_another_message(tmp_path, offline, envelope, reason, cost):
    post, _, _ = offline
    source = _csv(tmp_path / "faults.csv", ("case-1", _TOUCHSCREEN), ("case-2", _HEATER))
    post.side_effect = [_response(**envelope)]

    result = run_csv_batch(source, checkpoint=tmp_path / "batch.sqlite3", max_requests=5)

    assert result.requests_sent == result.request_failed == 1
    assert result.retry_pending == result.zero_cost_verified == 0
    assert result.initial_requests_remaining == 1 and reason in result.pause_reason
    history = _output(result, "attempts.csv")
    assert history[0]["issues"] == "cost_guard" and history[0]["cost"] == cost
    assert history[0]["http_status"] == "200"
    assert _output(result, "human_review.csv")[0]["urgency"] == ""
    post.assert_called_once()


def test_mock_300_independent_requests_reach_pydantic_and_export_in_input_order(tmp_path, monkeypatch, offline):
    post, _, _ = offline
    source = _csv(
        tmp_path / "faults.csv",
        *((f"case-{i:03d}", [_SAFETY, _SPARSE, _TOUCHSCREEN][i % 3]) for i in range(300)),
    )
    checkpoint = tmp_path / "batch.sqlite3"
    validator = Mock(wraps=_BatchDecision.model_validate_json)
    monkeypatch.setattr(_BatchDecision, "model_validate_json", validator)
    expected = [row["message_id"] for row in _read_csv(source)]

    first = run_csv_batch(source, checkpoint=checkpoint, max_requests=101)
    assert first.classified == first.requests_sent == validator.call_count == 101
    resumed = run_csv_batch(source, checkpoint=checkpoint, max_requests=199)
    assert resumed.requests_sent == 199 and resumed.classified == validator.call_count == 300
    assert resumed.pending == resumed.requests_remaining == 0
    assert _posted_ids(post) == expected
    assert [row["message_id"] for row in _output(resumed, "results.csv")] == expected
    assert [row["message_id"] for row in _output(resumed, "attempts.csv")] == expected
    assert len(set(call.args[0].data for call in post.call_args_list)) == 300
    assert [row["local_safety_urgent"] for row in _output(resumed, "results.csv")].count("True") == 100
    stats = json.loads((Path(resumed.output_directory) / "summary.json").read_text(encoding="utf-8"))
    assert stats["by_status"] == {"classified": 300}
    assert stats["reported_total_cost"] == "0"
    assert stats["reported_token_totals"] == {
        "prompt_tokens": 72000, "completion_tokens": 9000, "reasoning_tokens": 3000, "total_tokens": 81000,
    }
    assert stats["models"] == {"test-provider/model:free": 300}
    assert _output(resumed, "validation_failures.csv") == []


@pytest.mark.skipif(not _ZERO_CSV.is_file(), reason="The private Zero CSV is available only locally.")
def test_mock_original_zero_csv_sends_all_300_ids_unchanged(tmp_path, monkeypatch, offline):
    post, dotenv, _ = offline
    checkpoint = tmp_path / "batch.sqlite3"
    validator = Mock(wraps=_BatchDecision.model_validate_json)
    monkeypatch.setattr(_BatchDecision, "model_validate_json", validator)
    preview = run_csv_batch(_ZERO_CSV, checkpoint=checkpoint)
    assert preview.records == preview.requests_remaining == 300
    post.assert_not_called()
    dotenv.assert_not_called()

    result = run_csv_batch(_ZERO_CSV, checkpoint=checkpoint, max_requests=300)
    expected = [row["message_id"] for row in _read_csv(_ZERO_CSV)]
    decisions = _output(result, "results.csv")
    assert result.requests_sent == result.zero_cost_verified == validator.call_count == 300
    assert result.classified == result.completed == 300 and result.requests_remaining == 0
    assert _posted_ids(post) == [row["message_id"] for row in decisions] == expected
    assert Counter(row["local_safety_urgent"] for row in decisions) == {"True": 105, "False": 195}
    replay = run_csv_batch(_ZERO_CSV, checkpoint=checkpoint, max_requests=300)
    assert replay.requests_sent == 0 and post.call_count == 300
