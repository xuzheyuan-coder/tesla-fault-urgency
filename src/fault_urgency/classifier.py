"""Run the safety, review and routine routing rules in order."""

from .models import FaultMessage, FixPath, UrgencyDecision, UrgencyLevel
from .symptoms import assess_message_text

_SAFETY_ALERT_CODES = frozenset(
    {"BRK_w003", "BRK_a001", "ABS_w002", "EPAS_w202", "EPAS_a191", "ESP_w003"}
)

_ROUTINE_ROUTES = {
    "HVAC_w003": (
        FixPath.ROUTINE_SERVICE_APPOINTMENT,
        "Schedule service for the heating or climate-control issue.",
    ),
    "UI_a004": (
        FixPath.REMOTE_DIAGNOSIS_OR_OTA,
        "Start with remote diagnosis or software troubleshooting for the touchscreen issue.",
    ),
    "CP_a004": (
        FixPath.ROUTINE_SERVICE_APPOINTMENT,
        "Schedule service for the charge port door or latch.",
    ),
    "WIP_w001": (
        FixPath.ROUTINE_SERVICE_APPOINTMENT,
        "Schedule routine wiper maintenance.",
    ),
}

_CANONICAL_CODES = {
    code.casefold(): code for code in (*_SAFETY_ALERT_CODES, *_ROUTINE_ROUTES)
}


def classify_fault_message(message: FaultMessage) -> UrgencyDecision:
    """Return an urgency, a fix path and the evidence supporting the decision."""
    code = _CANONICAL_CODES.get(message.alert_code.casefold()) if message.alert_code else None
    safety_alert = code in _SAFETY_ALERT_CODES
    symptoms, review_signals = assess_message_text(
        message.message_text, wiper_alert=code == "WIP_w001"
    )

    if safety_alert or symptoms:
        evidence = []
        if safety_alert:
            evidence.append(f"Safety alert {code}.")
        if symptoms:
            evidence.append(f"Reported safety symptoms include {', '.join(symptoms)}.")
        return UrgencyDecision(
            urgency=UrgencyLevel.SAFETY_URGENT,
            fix_path=FixPath.STOP_DRIVING_AND_TOW,
            reason=" ".join(evidence),
            matched_alert_code=code,
            matched_symptoms=symptoms,
        )

    if review_signals:
        return UrgencyDecision(
            urgency=UrgencyLevel.SOON,
            fix_path=FixPath.EARLIER_SERVICE_APPOINTMENT,
            reason=f"Human review is required for the reported {' and '.join(review_signals)}.",
            matched_alert_code=code,
            human_review_required=True,
        )

    fix_path, reason = _ROUTINE_ROUTES.get(
        code,
        (
            FixPath.ROUTINE_SERVICE_APPOINTMENT,
            "The message supports routine service based on the available fault information.",
        ),
    )
    return UrgencyDecision(
        urgency=UrgencyLevel.ROUTINE,
        fix_path=fix_path,
        reason=reason,
        matched_alert_code=code,
    )
