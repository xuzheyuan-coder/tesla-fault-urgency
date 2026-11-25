"""Structured fault-message triage for the Zero service simulation."""

from .ai_classifier import OpenRouterAssessment, assess_with_openrouter, classify_with_openrouter
from .classifier import classify_fault_message
from .models import FaultMessage, FixPath, UrgencyDecision, UrgencyLevel
from .openrouter import OpenRouterError

__all__ = [
    "FaultMessage",
    "FixPath",
    "OpenRouterAssessment",
    "OpenRouterError",
    "UrgencyDecision",
    "UrgencyLevel",
    "assess_with_openrouter",
    "classify_fault_message",
    "classify_with_openrouter",
]
