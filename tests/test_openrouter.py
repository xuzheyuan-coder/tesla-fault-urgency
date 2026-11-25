import json
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

from fault_urgency import (
    FaultMessage,
    FixPath,
    OpenRouterError,
    UrgencyLevel,
    assess_with_openrouter,
    classify_fault_message,
    classify_with_openrouter,
)
from fault_urgency import openrouter
from fault_urgency.__main__ import main

_FAKE_KEY = "unit-test-key-only"


@pytest.fixture(autouse=True)
def post(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock = Mock(side_effect=AssertionError("This test did not set up an OpenRouter response."))
    monkeypatch.setattr(openrouter._OPENER, "open", mock)
    return mock


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text(f'OPENROUTER_API_KEY="{_FAKE_KEY}"\n', encoding="utf-8")
    return path


def _fault(
    text: str = "The touchscreen freezes when opening navigation.",
    code: str | None = "UI_a004",
) -> FaultMessage:
    return FaultMessage(message_id="test-openrouter", alert_code=code, message_text=text)


def _respond(
    mock: Mock,
    *,
    urgency: str = "routine",
    review: bool = False,
    finish_reason: str = "stop",
    content: str | None = None,
    **envelope: object,
) -> None:
    if content is None:
        content = json.dumps({"urgency": urgency, "human_review_required": review})
    mock.side_effect = None
    mock.return_value = BytesIO(
        json.dumps(
            {
                "model": "test-provider/model:free",
                "usage": {"cost": 0, "is_byok": False},
                "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
                **envelope,
            }
        ).encode()
    )


def test_uses_one_structured_free_request_with_a_zero_price_cap(
    post: Mock, env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "ignore-this-environment-value")
    monkeypatch.setenv("OPENROUTER_MODEL", "paid-provider/model")
    _respond(post)

    decision = classify_with_openrouter(_fault(), env_file=env_file)

    post.assert_called_once()
    request = post.call_args.args[0]
    payload = json.loads(request.data)
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == f"Bearer {_FAKE_KEY}"
    assert "Authorization" not in request.headers
    assert _FAKE_KEY not in request.data.decode()
    assert payload["model"] == "openrouter/free"
    assert "models" not in payload
    assert payload["max_completion_tokens"] == 2048
    assert "max_tokens" not in payload
    assert payload["provider"] == {
        "allow_fallbacks": False,
        "require_parameters": True,
        "max_price": {"prompt": 0, "completion": 0, "request": 0},
    }
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert decision.urgency == UrgencyLevel.ROUTINE
    assert decision.fix_path == FixPath.REMOTE_DIAGNOSIS_OR_OTA


def test_sends_fault_observations_instead_of_original_text_or_identifiers(
    post: Mock, env_file: Path
) -> None:
    first = FaultMessage(
        message_id="case-98765",
        alert_code="ZZZTEST00AA000000",
        message_text=(
            "Alice Example, alice@example.test, VIN ZZZTEST00AA000000, plate XTR-276. "
            "I have to press the brake pedal much harder."
        ),
        customer_reported_urgency="Alice says this can wait.",
    )
    second = FaultMessage(
        message_id="case-11111",
        alert_code="NONE",
        message_text="Bob Sample, bob@example.test. I have to press the brake pedal much harder.",
    )

    for message in (first, second):
        _respond(post, urgency="safety_urgent")
        decision = classify_with_openrouter(message, env_file=env_file)
        assert decision.urgency == UrgencyLevel.SAFETY_URGENT
        assert decision.fix_path == FixPath.STOP_DRIVING_AND_TOW
    first_payload, second_payload = (call.args[0].data for call in post.call_args_list)
    assert first_payload == second_payload
    payload = json.loads(first_payload)
    assert json.loads(payload["messages"][1]["content"]) == {
        "alert_code": None,
        "observations": ["extra pedal effort or travel", "braking, brake pedals, or ABS"],
        "local_review_required": False,
    }


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("", "BRK_a001"),
        ("The brake pedal keeps sinking. No warning light and no rush.", "NONE"),
        ("The steering is suddenly heavy while turning.", None),
        ("The wipers stopped and I cannot see the road in heavy rain.", "WIP_w001"),
    ],
)
def test_confirmed_local_safety_decisions_bypass_the_api(
    text: str, code: str | None, post: Mock, tmp_path: Path
) -> None:
    message = _fault(text, code)
    decision = classify_with_openrouter(message, env_file=tmp_path / ".env")
    assert decision == classify_fault_message(message)
    assert decision.urgency == UrgencyLevel.SAFETY_URGENT
    post.assert_not_called()


@pytest.mark.parametrize(
    "text", ["Please call Alice Example.", "The touchscreen.", ""]
)
def test_missing_symptom_details_go_to_review_without_a_request(
    text: str, post: Mock, tmp_path: Path
) -> None:
    decision = classify_with_openrouter(_fault(text), env_file=tmp_path / ".env")
    assert decision.urgency == UrgencyLevel.SOON
    assert decision.fix_path == FixPath.EARLIER_SERVICE_APPOINTMENT
    assert decision.human_review_required is True
    post.assert_not_called()


def test_ai_cannot_remove_a_local_review_requirement(post: Mock, env_file: Path) -> None:
    _respond(post)
    message = _fault("The brakes feel weird and the car shakes.", None)
    decision = classify_with_openrouter(message, env_file=env_file)
    assert decision == classify_fault_message(message)
    assert decision.urgency == UrgencyLevel.SOON
    assert decision.human_review_required is True
    post.assert_called_once()


@pytest.mark.parametrize(("urgency", "review"), [("routine", True), ("soon", False)])
def test_ai_uncertainty_or_soon_requires_human_review(
    urgency: str, review: bool, post: Mock, env_file: Path
) -> None:
    _respond(post, urgency=urgency, review=review)
    decision = classify_with_openrouter(_fault(), env_file=env_file)
    assert decision.urgency == UrgencyLevel.SOON
    assert decision.fix_path == FixPath.EARLIER_SERVICE_APPOINTMENT
    assert decision.human_review_required is True


@pytest.mark.parametrize(
    ("code", "text", "fix_path"),
    [
        ("ui_A004", "The touchscreen freezes.", FixPath.REMOTE_DIAGNOSIS_OR_OTA),
        ("HVAC_w003", "The heater blows cold air.", FixPath.ROUTINE_SERVICE_APPOINTMENT),
        ("CP_a004", "The charge port latch will not close.", FixPath.ROUTINE_SERVICE_APPOINTMENT),
        ("WIP_w001", "The wipers leave light streaks.", FixPath.ROUTINE_SERVICE_APPOINTMENT),
    ],
)
def test_routine_ai_results_keep_the_local_fix_path(
    code: str, text: str, fix_path: FixPath, post: Mock, env_file: Path
) -> None:
    _respond(post)
    decision = classify_with_openrouter(_fault(text, code), env_file=env_file)
    assert decision.fix_path == fix_path
    payload = json.loads(post.call_args.args[0].data)
    assert json.loads(payload["messages"][1]["content"])["alert_code"] == decision.matched_alert_code


@pytest.mark.parametrize(
    ("content", "finish_reason", "status", "issue"),
    [
        ("not JSON", "stop", "invalid_output", "malformed_json"),
        (
            '{"urgency":"unknown","human_review_required":false}',
            "stop", "invalid_output", "schema_validation_failed",
        ),
        (
            '{"urgency":"routine","human_review_required":"false"}',
            "stop", "invalid_output", "schema_validation_failed",
        ),
        ('{"urgency":"routine"}', "stop", "invalid_output", "missing_fields"),
        (
            '{"urgency":"soon","human_review_required":true,"notes":"unexpected"}',
            "stop", "invalid_output", "schema_validation_failed",
        ),
        (
            '{"urgency":"routine","human_review_required":false}',
            "length", "incomplete_output", "truncated",
        ),
    ],
)
def test_incomplete_or_invalid_model_output_is_referred_for_review(
    content: str, finish_reason: str, status: str, issue: str, post: Mock, env_file: Path
) -> None:
    _respond(post, content=content, finish_reason=finish_reason)
    result = assess_with_openrouter(_fault(), env_file=env_file)
    assert result.status == status
    assert result.issue == issue
    decision = result.decision
    assert decision.urgency == UrgencyLevel.SOON
    assert decision.fix_path == FixPath.EARLIER_SERVICE_APPOINTMENT
    assert decision.human_review_required is True
    post.assert_called_once()


@pytest.mark.parametrize(
    ("choices", "issue"),
    [
        ([], "missing_choice"),
        ([{"message": {"content": "{}"}}], "unexpected_finish_reason"),
        ([{"finish_reason": "tool_calls", "message": {"content": "{}"}}], "unexpected_finish_reason"),
        ([{"finish_reason": "stop"}], "missing_message"),
        ([{"finish_reason": "stop", "message": {}}], "missing_content"),
        ([{"finish_reason": "stop", "message": {"content": " "}}], "empty_content"),
        ([{"finish_reason": "stop", "message": {"content": []}}], "non_text_content"),
        (
            [{"finish_reason": "stop", "message": {"content": None, "reasoning": "Thinking"}}],
            "reasoning_only",
        ),
        ([{"finish_reason": "content_filter", "message": {"refusal": "Filtered"}}], "refused_or_filtered"),
        (
            [{"finish_reason": "stop", "native_finish_reason": "MAX_TOKENS", "message": {"content": "{}"}}],
            "truncated",
        ),
    ],
)
def test_incomplete_response_shapes_have_distinct_diagnostics(
    choices: list[dict], issue: str, post: Mock, env_file: Path
) -> None:
    _respond(post, choices=choices)
    result = assess_with_openrouter(_fault(), env_file=env_file)
    assert result.status == "incomplete_output"
    assert result.issue == issue
    assert result.decision.human_review_required
    post.assert_called_once()


@pytest.mark.parametrize(
    ("status", "detail", "message"),
    [
        (301, "redirect", "redirect"),
        (401, "key rejected", "API key"),
        (402, "account limit", "account, credit, or key limit"),
        (403, "unsupported in your country", "regional restriction"),
        (429, "quota reached", "request limit"),
        (503, "no endpoints found", "No compatible free model"),
    ],
)
def test_service_errors_are_safe_and_never_retried(
    status: int, detail: str, message: str, post: Mock, env_file: Path
) -> None:
    body = json.dumps({"error": {"code": status, "message": f"{detail} {_FAKE_KEY}"}})
    post.side_effect = HTTPError(openrouter._CHAT_URL, status, _FAKE_KEY, {}, BytesIO(body.encode()))
    with pytest.raises(OpenRouterError, match=message) as caught:
        classify_with_openrouter(_fault(), env_file=env_file)
    assert _FAKE_KEY not in str(caught.value)
    post.assert_called_once()


def test_provider_error_inside_a_success_status_is_reported(post: Mock, env_file: Path) -> None:
    _respond(post, error={"code": 429, "message": _FAKE_KEY})
    with pytest.raises(OpenRouterError, match="request limit"):
        classify_with_openrouter(_fault(), env_file=env_file)
    post.assert_called_once()


def test_network_failure_reports_a_safe_error_without_retry(post: Mock, env_file: Path) -> None:
    post.side_effect = URLError(f"Network failure {_FAKE_KEY}")
    with pytest.raises(OpenRouterError, match="HTTPS request") as caught:
        classify_with_openrouter(_fault(), env_file=env_file)
    assert _FAKE_KEY not in str(caught.value)
    post.assert_called_once()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": "paid-provider/model"}, "free model"),
        ({"usage": {"cost": 0.0003}}, "zero usage cost"),
        ({"usage": {"cost": "0"}}, "zero usage cost"),
        ({"usage": None}, "usage data"),
        ({"usage": {"cost": 0, "is_byok": True}}, "bring-your-own-key"),
        ({"usage": {"cost": 0, "cost_details": {"upstream_inference_cost": 0.01}}}, "upstream"),
    ],
)
def test_response_must_verify_free_routing_and_zero_cost(
    overrides: dict, message: str, post: Mock, env_file: Path
) -> None:
    _respond(post, **overrides)
    with pytest.raises(OpenRouterError, match=message):
        classify_with_openrouter(_fault(), env_file=env_file)
    post.assert_called_once()


def test_missing_key_stops_before_network(post: Mock, tmp_path: Path) -> None:
    with pytest.raises(OpenRouterError, match="Set OPENROUTER_API_KEY"):
        classify_with_openrouter(_fault(), env_file=tmp_path / ".env")
    post.assert_not_called()


@pytest.mark.parametrize(
    ("urgency", "review", "path"),
    [
        ("routine", False, "remote_diagnosis_or_ota"),
        ("soon", True, "earlier_service_appointment"),
    ],
)
def test_smoke_cli_distinguishes_a_valid_classification_from_a_fallback(
    urgency: str,
    review: bool,
    path: str,
    post: Mock,
    env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.chdir(env_file.parent)
    _respond(post, urgency=urgency, review=review)
    assert main(["--smoke-test"]) == 0
    captured = capsys.readouterr()
    assert "zero cost" in captured.err
    assert json.loads(captured.out)["fix_path"] == path
    diagnostics = json.loads(captured.err.splitlines()[0])
    assert diagnostics["assessment_status"] == "classified"
    assert diagnostics["issue"] is None
    assert _FAKE_KEY not in captured.out + captured.err
    post.assert_called_once()


def test_smoke_cli_reports_truncation_model_and_token_counts(
    post: Mock, env_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.chdir(env_file.parent)
    _respond(
        post,
        id="gen-test-completion",
        choices=[
            {
                "finish_reason": "length",
                "native_finish_reason": "MAX_TOKENS",
                "message": {"content": "", "reasoning": f"Mock reasoning with {_FAKE_KEY}"},
            },
        ],
        usage={
            "cost": 0, "prompt_tokens": 400, "completion_tokens": 2048, "total_tokens": 2448,
            "completion_tokens_details": {"reasoning_tokens": 2048},
        },
    )

    assert main(["--smoke-test"]) == 2
    captured = capsys.readouterr()
    diagnostics = json.loads(captured.err.splitlines()[0])
    assert diagnostics == {
        "assessment_status": "incomplete_output",
        "issue": "truncated",
        "model": "test-provider/model:free",
        "generation_id": "gen-test-completion",
        "finish_reason": "length",
        "native_finish_reason": "max_tokens",
        "completion_tokens": 2048,
        "reasoning_tokens": 2048,
        "content_characters": 0,
        "prompt_tokens": 400,
        "total_tokens": 2448,
        "http_status": 200,
        "max_completion_tokens": 2048,
        "response_format": "json_schema",
        "cost": 0,
    }
    assert json.loads(captured.out)["human_review_required"] is True
    assert _FAKE_KEY not in captured.out + captured.err
    assert "Mock reasoning" not in captured.out + captured.err
    post.assert_called_once()


def test_smoke_cli_reports_request_failure_as_separate_status(
    post: Mock, env_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.chdir(env_file.parent)
    post.side_effect = URLError(_FAKE_KEY)
    assert main(["--smoke-test"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["assessment_status"] == "request_error"
    assert _FAKE_KEY not in captured.err
    post.assert_called_once()


def test_diagnostics_withhold_untrusted_metadata(
    post: Mock, env_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.chdir(env_file.parent)
    _respond(
        post,
        model=f"test-provider/{_FAKE_KEY}:free",
        id=f"gen-{_FAKE_KEY}",
        choices=[
            {
                "finish_reason": "stop",
                "native_finish_reason": _FAKE_KEY,
                "message": {"content": '{"urgency":"routine","human_review_required":false}'},
            },
        ],
    )

    assert main(["--smoke-test"]) == 0
    captured = capsys.readouterr()
    diagnostics = json.loads(captured.err.splitlines()[0])
    assert diagnostics["model"] is None
    assert diagnostics["generation_id"] is None
    assert diagnostics["native_finish_reason"] == "other"
    assert _FAKE_KEY not in captured.out + captured.err


def test_cli_classifies_a_single_json_message(
    post: Mock,
    env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.chdir(env_file.parent)
    path = env_file.parent / "message.json"
    message = _fault("The heater blows cold air.", "HVAC_w003")
    path.write_text(message.model_dump_json(), encoding="utf-8")
    _respond(post)

    assert main(["--input", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["fix_path"] == "routine_service_appointment"
    post.assert_called_once()
