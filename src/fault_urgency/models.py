"""Input and output schemas for fault-message triage."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class UrgencyLevel(StrEnum):
    SAFETY_URGENT = "safety_urgent"
    SOON = "soon"
    ROUTINE = "routine"


class FixPath(StrEnum):
    STOP_DRIVING_AND_TOW = "stop_driving_and_tow"
    EARLIER_SERVICE_APPOINTMENT = "earlier_service_appointment"
    REMOTE_DIAGNOSIS_OR_OTA = "remote_diagnosis_or_ota"
    ROUTINE_SERVICE_APPOINTMENT = "routine_service_appointment"


_ALLOWED_FIX_PATHS = {
    UrgencyLevel.SAFETY_URGENT: {FixPath.STOP_DRIVING_AND_TOW},
    UrgencyLevel.SOON: {FixPath.EARLIER_SERVICE_APPOINTMENT},
    UrgencyLevel.ROUTINE: {
        FixPath.REMOTE_DIAGNOSIS_OR_OTA,
        FixPath.ROUTINE_SERVICE_APPOINTMENT,
    },
}


class FaultMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message_id: str = Field(min_length=1, description="Identifier for the fault message.")
    alert_code: str | None = Field(
        default=None,
        description="Vehicle alert code. A blank value or NONE represents a missing code.",
    )
    message_text: str = Field(description="Customer description or service advisor notes.")
    customer_reported_urgency: str | None = Field(
        default=None,
        description="Optional customer assessment retained as context for service staff.",
    )

    @field_validator("alert_code")
    @classmethod
    def normalize_missing_code(cls, value: str | None) -> str | None:
        if not value or value.casefold() == "none":
            return None
        return value

    @model_validator(mode="after")
    def require_classification_input(self) -> "FaultMessage":
        if not self.message_text and self.alert_code is None:
            raise ValueError("Provide message text or a vehicle alert code.")
        return self


class UrgencyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    urgency: UrgencyLevel = Field(description="Urgency assigned by the triage rules.")
    fix_path: FixPath = Field(description="Recommended service path for this decision.")
    reason: str = Field(min_length=1, description="Explanation of the evidence used.")
    matched_alert_code: str | None = Field(
        default=None,
        description="Canonical alert code when the supplied code is recognized.",
    )
    matched_symptoms: list[str] = Field(
        default_factory=list,
        description="Safety-critical symptoms identified in the message text.",
    )
    human_review_required: bool = Field(
        default=False,
        description="Whether a reviewer must assess an unclear safety concern.",
    )

    @model_validator(mode="after")
    def validate_fix_path(self) -> "UrgencyDecision":
        if self.fix_path not in _ALLOWED_FIX_PATHS[self.urgency]:
            raise ValueError(
                f"Fix path '{self.fix_path}' is incompatible with urgency '{self.urgency}'."
            )
        return self
