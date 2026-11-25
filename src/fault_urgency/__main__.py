"""Classify a JSON report, test OpenRouter, or process a local CSV batch."""

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

from pydantic import ValidationError

from .ai_classifier import assess_with_openrouter
from .batch import BatchInputError, DEFAULT_CHECKPOINT, run_csv_batch
from .models import FaultMessage
from .openrouter import OpenRouterError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify fault messages using local rules and OpenRouter.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Path to a local FaultMessage JSON file.")
    source.add_argument("--smoke-test", action="store_true", help="Send one synthetic touchscreen report.")
    source.add_argument("--csv", type=Path, help="Plan a CSV batch without sending requests by default.")
    parser.add_argument("--checkpoint", type=Path, help=f"SQLite batch state (default: {DEFAULT_CHECKPOINT}).")
    parser.add_argument("--execute", action="store_true", help="Enable real requests for a CSV batch.")
    parser.add_argument("--max-requests", type=int, help="Maximum number of requests for this execution.")
    args = parser.parse_args(argv)

    if args.csv is not None:
        if args.execute != (args.max_requests is not None):
            parser.error("CSV execution requires both --execute and --max-requests.")
        try:
            summary = run_csv_batch(
                args.csv, checkpoint=args.checkpoint or DEFAULT_CHECKPOINT, max_requests=args.max_requests
            )
        except BatchInputError as error:
            print(str(error), file=sys.stderr)
            return 1
        except (OSError, sqlite3.Error, ValidationError):
            print("Unable to use the SQLite checkpoint or write the batch reports.", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("Batch interrupted. Resume with the same CSV and checkpoint.", file=sys.stderr)
            return 130
        print(json.dumps(asdict(summary), indent=2))
        return 2 if args.execute and (summary.pause_reason or summary.unconfirmed_requests) else 0
    if args.execute or args.max_requests is not None or args.checkpoint is not None:
        parser.error("--execute, --max-requests, and --checkpoint require --csv.")

    if args.smoke_test:
        message = FaultMessage(
            message_id="openrouter-smoke-test",
            alert_code="UI_a004",
            message_text="The touchscreen freezes when using navigation after a software update.",
        )
    else:
        try:
            message = FaultMessage.model_validate_json(args.input.read_bytes())
        except (OSError, ValidationError):
            print("Unable to load a valid FaultMessage JSON record from the input file.", file=sys.stderr)
            return 1

    try:
        assessment = assess_with_openrouter(message)
    except OpenRouterError as error:
        print(json.dumps({"assessment_status": "request_error", "error": str(error)}), file=sys.stderr)
        return 1

    print(assessment.decision.model_dump_json(indent=2))
    diagnostics = asdict(assessment.diagnostics) if assessment.diagnostics is not None else {}
    print(
        json.dumps({"assessment_status": assessment.status, "issue": assessment.issue, **diagnostics}),
        file=sys.stderr,
    )
    if args.smoke_test:
        if assessment.diagnostics is not None:
            print("OpenRouter reported zero cost for this request.", file=sys.stderr)
        return 0 if assessment.status == "classified" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
