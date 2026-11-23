import pytest

from fault_urgency import (
    FaultMessage,
    FixPath,
    UrgencyDecision,
    UrgencyLevel,
    classify_fault_message,
)


def classify(
    message_text: str,
    *,
    alert_code: str | None = None,
    reported_urgency: str | None = None,
) -> UrgencyDecision:
    return classify_fault_message(
        FaultMessage(
            message_id="test-001",
            alert_code=alert_code,
            message_text=message_text,
            customer_reported_urgency=reported_urgency,
        )
    )


@pytest.mark.parametrize(
    "code", ["BRK_w003", "BRK_a001", "ABS_w002", "EPAS_w202", "EPAS_a191", "ESP_w003"]
)
def test_all_known_safety_codes_trigger_towing(code: str) -> None:
    result = classify("", alert_code=code)

    assert result.urgency == UrgencyLevel.SAFETY_URGENT
    assert result.fix_path == FixPath.STOP_DRIVING_AND_TOW
    assert result.matched_alert_code == code
    assert result.matched_symptoms == []
    assert code in result.reason


@pytest.mark.parametrize("missing_code", [None, "NONE"])
@pytest.mark.parametrize(
    ("description", "symptom"),
    [
        ("The brake pedal feels spongy.", "soft brake pedal"),
        ("The brake pedal slowly sinks to the floor.", "sinking brake pedal"),
        ("Braking is delayed when I press the pedal.", "delayed or ineffective braking"),
        ("I stepped on the brake and it didn't slow down.", "delayed or ineffective braking"),
        ("I get a grinding sound when I brake.", "grinding during braking"),
        ("When braking, the vehicle shudders.", "shuddering during braking"),
        ("The car pulls to the left as I come to a stop.", "pulling to one side during braking"),
        ("The steering suddenly feels very heavy.", "heavy or unstable steering"),
        ("The steering wheel wobbles on the motorway.", "heavy or unstable steering"),
    ],
)
def test_text_recovers_safety_symptoms_without_a_code(
    missing_code: str | None, description: str, symptom: str
) -> None:
    result = classify(description, alert_code=missing_code)

    assert result.urgency == UrgencyLevel.SAFETY_URGENT
    assert result.fix_path == FixPath.STOP_DRIVING_AND_TOW
    assert result.matched_alert_code is None
    assert symptom in result.matched_symptoms
    assert symptom in result.reason
    assert not result.human_review_required


def test_customer_minimization_and_no_warning_do_not_override_safety() -> None:
    result = classify(
        "No dashboard warning and no rush, but my brake pedal sinks at red lights.",
        reported_urgency="not urgent",
    )

    assert result.urgency == UrgencyLevel.SAFETY_URGENT
    assert result.matched_symptoms == ["sinking brake pedal"]


def test_safety_symptoms_override_a_routine_code() -> None:
    result = classify("The touchscreen froze. The steering feels heavy.", alert_code="UI_a004")

    assert result.urgency == UrgencyLevel.SAFETY_URGENT
    assert result.fix_path == FixPath.STOP_DRIVING_AND_TOW
    assert result.matched_alert_code == "UI_a004"
    assert result.matched_symptoms == ["heavy or unstable steering"]


@pytest.mark.parametrize("code", ["WIP_w001", None])
def test_wiper_fault_with_seriously_obstructed_visibility(code: str | None) -> None:
    result = classify(
        "My wipers stopped working in heavy rain and I could barely see the road.",
        alert_code=code,
    )

    assert result.urgency == UrgencyLevel.SAFETY_URGENT
    assert result.fix_path == FixPath.STOP_DRIVING_AND_TOW
    assert result.matched_symptoms == ["seriously obstructed driver visibility"]


def test_wiper_code_supplies_context_for_description_of_obstructed_view() -> None:
    result = classify("My view of the road is seriously obstructed in rain.", alert_code="WIP_w001")

    assert result.urgency == UrgencyLevel.SAFETY_URGENT
    assert result.fix_path == FixPath.STOP_DRIVING_AND_TOW


@pytest.mark.parametrize(
    ("code", "description", "path"),
    [
        (
            "WIP_w001",
            "A blade squeaks and leaves light streaks. I can still see clearly.",
            FixPath.ROUTINE_SERVICE_APPOINTMENT,
        ),
        (
            "UI_a004",
            "The touchscreen freezes until I restart the software.",
            FixPath.REMOTE_DIAGNOSIS_OR_OTA,
        ),
        (
            "CP_a004",
            "The charge port door latch takes two tries to close.",
            FixPath.ROUTINE_SERVICE_APPOINTMENT,
        ),
        (
            "HVAC_w003",
            "The cabin heater blows cold air even at the highest setting.",
            FixPath.ROUTINE_SERVICE_APPOINTMENT,
        ),
    ],
)
def test_routine_alert_codes_get_their_specified_fix_paths(
    code: str, description: str, path: FixPath
) -> None:
    result = classify(description, alert_code=code)

    assert result.urgency == UrgencyLevel.ROUTINE
    assert result.fix_path == path
    assert result.matched_alert_code == code
    assert not result.human_review_required


@pytest.mark.parametrize(
    ("code", "description"),
    [
        (None, "The brakes feel odd sometimes. I can't describe what changed."),
        ("CP_a004", "The steering felt strange once. Can someone call me to discuss it?"),
        ("NONE", "An ABS warning appeared briefly. I didn't catch the details."),
        ("WIP_w001", "The wipers are streaking and causing reduced visibility in rain."),
    ],
)
def test_ambiguous_high_risk_reports_are_referred_for_review(
    code: str | None, description: str
) -> None:
    result = classify(description, alert_code=code, reported_urgency="routine")

    assert result.urgency == UrgencyLevel.SOON
    assert result.fix_path == FixPath.EARLIER_SERVICE_APPOINTMENT
    assert result.human_review_required
    assert result.matched_symptoms == []
    assert "review" in result.reason.lower()


@pytest.mark.parametrize(
    "description",
    [
        "The brake pedal is not soft. The cabin fan squeaks.",
        "There is no grinding or shuddering when braking.",
        "No brake warning. Please inspect the pads at my regular service.",
        "The accelerator pedal feels soft.",
        "The car didn't stop charging at the scheduled time.",
        "The wipers squeak and leave streaks. I cannot see the touchscreen in sunlight.",
    ],
)
def test_negated_symptoms_and_unrelated_words_do_not_escalate(description: str) -> None:
    result = classify(description)

    assert result.urgency == UrgencyLevel.ROUTINE
    assert result.fix_path == FixPath.ROUTINE_SERVICE_APPOINTMENT
    assert result.matched_symptoms == []
    assert not result.human_review_required


def test_customer_reported_urgency_alone_does_not_change_routine_service() -> None:
    result = classify("The charge port latch sticks.", alert_code="CP_a004", reported_urgency="urgent")

    assert result.urgency == UrgencyLevel.ROUTINE


def test_alert_matching_is_case_insensitive_and_returns_the_canonical_code() -> None:
    result = classify("The alert appeared this morning.", alert_code=" brk_W003 ")

    assert result.matched_alert_code == "BRK_w003"
    assert result.urgency == UrgencyLevel.SAFETY_URGENT


def test_unrecognized_code_does_not_count_as_a_match() -> None:
    result = classify("The seat upholstery has a tear.", alert_code="unrecognized")

    assert result.matched_alert_code is None
    assert result.urgency == UrgencyLevel.ROUTINE
