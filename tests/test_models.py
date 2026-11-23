import pytest
from pydantic import BaseModel, ValidationError

from fault_urgency import FaultMessage, FixPath, UrgencyDecision, UrgencyLevel


_VALID_MESSAGE = {"message_id": "test-001", "message_text": "The wiper blade squeaks."}
_VALID_DECISION = {
    "urgency": "soon",
    "fix_path": "earlier_service_appointment",
    "reason": "The steering description needs review.",
}


def test_enums_have_the_required_serialized_values() -> None:
    assert {level.value for level in UrgencyLevel} == {"safety_urgent", "soon", "routine"}
    assert {path.value for path in FixPath} == {
        "stop_driving_and_tow",
        "earlier_service_appointment",
        "remote_diagnosis_or_ota",
        "routine_service_appointment",
    }


@pytest.mark.parametrize("missing_code", [None, "NONE", " none ", "", "  "])
def test_missing_alert_values_are_normalized(missing_code: str | None) -> None:
    message = FaultMessage(message_id="test-001", message_text="Wiper squeak", alert_code=missing_code)

    assert message.alert_code is None


def test_fault_message_requires_a_code_or_text() -> None:
    with pytest.raises(ValidationError, match="Provide message text or a vehicle alert code"):
        FaultMessage(message_id="test-001", message_text=" ", alert_code="NONE")


@pytest.mark.parametrize(
    ("model", "record", "required_field"),
    [
        (FaultMessage, _VALID_MESSAGE, "message_id"),
        (FaultMessage, {**_VALID_MESSAGE, "alert_code": "BRK_w003"}, "message_text"),
        (UrgencyDecision, _VALID_DECISION, "urgency"),
        (UrgencyDecision, _VALID_DECISION, "fix_path"),
        (UrgencyDecision, _VALID_DECISION, "reason"),
    ],
)
def test_missing_required_field_reports_its_name(
    model: type[BaseModel], record: dict[str, object], required_field: str
) -> None:
    incomplete = record.copy()
    del incomplete[required_field]

    with pytest.raises(ValidationError) as caught:
        model.model_validate(incomplete)

    (error,) = caught.value.errors()
    assert error["loc"] == (required_field,)
    assert error["type"] == "missing"


@pytest.mark.parametrize(
    ("model", "record", "location", "value", "error_type"),
    [
        (FaultMessage, _VALID_MESSAGE, ("message_id",), 123, "string_type"),
        (FaultMessage, _VALID_MESSAGE, ("message_id",), None, "string_type"),
        (FaultMessage, _VALID_MESSAGE, ("message_text",), None, "string_type"),
        (FaultMessage, _VALID_MESSAGE, ("message_text",), b"wiper squeak", "string_type"),
        (FaultMessage, _VALID_MESSAGE, ("alert_code",), b"BRK_w003", "string_type"),
        (FaultMessage, _VALID_MESSAGE, ("customer_reported_urgency",), False, "string_type"),
        (UrgencyDecision, _VALID_DECISION, ("urgency",), None, "value_error"),
        (UrgencyDecision, _VALID_DECISION, ("urgency",), b"soon", "value_error"),
        (UrgencyDecision, _VALID_DECISION, ("fix_path",), None, "value_error"),
        (
            UrgencyDecision,
            _VALID_DECISION,
            ("fix_path",),
            b"earlier_service_appointment",
            "value_error",
        ),
        (UrgencyDecision, _VALID_DECISION, ("reason",), None, "string_type"),
        (UrgencyDecision, _VALID_DECISION, ("reason",), b"review needed", "string_type"),
        (UrgencyDecision, _VALID_DECISION, ("matched_alert_code",), b"BRK_w003", "string_type"),
        (UrgencyDecision, _VALID_DECISION, ("matched_symptoms",), ("wobble",), "list_type"),
        (UrgencyDecision, _VALID_DECISION, ("matched_symptoms",), None, "list_type"),
        (
            UrgencyDecision,
            _VALID_DECISION,
            ("matched_symptoms", 1),
            ["wobble", b"grinding"],
            "string_type",
        ),
        (UrgencyDecision, _VALID_DECISION, ("human_review_required",), "false", "bool_type"),
        (UrgencyDecision, _VALID_DECISION, ("human_review_required",), 1, "bool_type"),
        (UrgencyDecision, _VALID_DECISION, ("human_review_required",), None, "bool_type"),
    ],
)
def test_incorrect_field_type_reports_its_location(
    model: type[BaseModel],
    record: dict[str, object],
    location: tuple[str | int, ...],
    value: object,
    error_type: str,
) -> None:
    with pytest.raises(ValidationError) as caught:
        model.model_validate({**record, location[0]: value})

    (error,) = caught.value.errors()
    assert error["loc"] == location
    assert error["type"] == error_type


@pytest.mark.parametrize(
    ("field", "value"),
    [("urgency", "critical"), ("fix_path", "unsupported_path")],
)
def test_unknown_enum_strings_are_rejected(field: str, value: str) -> None:
    with pytest.raises(ValidationError) as caught:
        UrgencyDecision.model_validate({**_VALID_DECISION, field: value})

    (error,) = caught.value.errors()
    assert error["loc"] == (field,)
    assert error["type"] == "enum"


def test_omitted_optional_fields_use_their_defaults() -> None:
    message = FaultMessage.model_validate(_VALID_MESSAGE)
    decision = UrgencyDecision.model_validate(_VALID_DECISION)

    assert message.alert_code is None
    assert message.customer_reported_urgency is None
    assert decision.matched_alert_code is None
    assert decision.matched_symptoms == []
    assert decision.human_review_required is False


def test_nullable_fields_accept_explicit_none() -> None:
    message = FaultMessage.model_validate(
        {**_VALID_MESSAGE, "alert_code": None, "customer_reported_urgency": None}
    )
    decision = UrgencyDecision.model_validate({**_VALID_DECISION, "matched_alert_code": None})

    assert message.alert_code is None
    assert message.customer_reported_urgency is None
    assert decision.matched_alert_code is None


def test_optional_fields_accept_valid_values() -> None:
    message = FaultMessage.model_validate(
        {**_VALID_MESSAGE, "alert_code": " WIP_w001 ", "customer_reported_urgency": " not urgent "}
    )
    decision = UrgencyDecision.model_validate(
        {
            **_VALID_DECISION,
            "matched_alert_code": "WIP_w001",
            "matched_symptoms": ["wiper visibility concern"],
            "human_review_required": True,
        }
    )

    assert message.alert_code == "WIP_w001"
    assert message.customer_reported_urgency == "not urgent"
    assert decision.matched_alert_code == "WIP_w001"
    assert decision.matched_symptoms == ["wiper visibility concern"]
    assert decision.human_review_required is True


@pytest.mark.parametrize(
    ("urgency", "fix_path"),
    [
        (UrgencyLevel.SAFETY_URGENT, FixPath.ROUTINE_SERVICE_APPOINTMENT),
        (UrgencyLevel.SOON, FixPath.REMOTE_DIAGNOSIS_OR_OTA),
        (UrgencyLevel.ROUTINE, FixPath.STOP_DRIVING_AND_TOW),
    ],
)
def test_incompatible_urgency_and_fix_path_are_rejected(
    urgency: UrgencyLevel, fix_path: FixPath
) -> None:
    with pytest.raises(ValidationError, match="incompatible"):
        UrgencyDecision(urgency=urgency, fix_path=fix_path, reason="Test decision")


def test_decision_serializes_to_the_expected_shape() -> None:
    decision = UrgencyDecision(
        urgency=UrgencyLevel.SOON,
        fix_path=FixPath.EARLIER_SERVICE_APPOINTMENT,
        reason="The brake description needs review.",
        human_review_required=True,
    )

    assert decision.model_dump(mode="json") == {
        "urgency": "soon",
        "fix_path": "earlier_service_appointment",
        "reason": "The brake description needs review.",
        "matched_alert_code": None,
        "matched_symptoms": [],
        "human_review_required": True,
    }
    assert UrgencyDecision.model_validate_json(decision.model_dump_json()) == decision
