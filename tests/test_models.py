import pytest
from pydantic import ValidationError

from fault_urgency import FaultMessage, FixPath, UrgencyDecision, UrgencyLevel


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


def test_fault_message_validates_field_types() -> None:
    with pytest.raises(ValidationError, match="message_text"):
        FaultMessage.model_validate({"message_id": "test-001", "message_text": 42})


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
