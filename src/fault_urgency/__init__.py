"""Structured fault-message triage for the Zero service simulation."""

from .classifier import classify_fault_message
from .models import FaultMessage, FixPath, UrgencyDecision, UrgencyLevel

__all__ = [
    "FaultMessage",
    "FixPath",
    "UrgencyDecision",
    "UrgencyLevel",
    "classify_fault_message",
]
