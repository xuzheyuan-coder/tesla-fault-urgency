# Define the urgency rules for fault messages

A local Python classifier for the Zero Tesla service simulation. A validated
`FaultMessage` produces an `UrgencyDecision` with separate urgency and fix-path
fields, a reason, and the evidence behind the decision.

## Decision rules

The classifier applies these rules from top to bottom.

| Evidence | Urgency | Fix path |
| --- | --- | --- |
| A recognized safety alert or a credible safety-critical symptom | `safety_urgent` | `stop_driving_and_tow` |
| An unclear brake, steering, or wiper visibility concern | `soon` | `earlier_service_appointment`, with human review required |
| `UI_a004` touchscreen or software issue | `routine` | `remote_diagnosis_or_ota` |
| `HVAC_w003` heating hardware, `CP_a004` charge port door or latch, `WIP_w001` ordinary wiper maintenance, or other messages without a safety or review signal | `routine` | `routine_service_appointment` |

The safety alerts are `BRK_w003`, `BRK_a001`, `ABS_w002`, `EPAS_w202`,
`EPAS_a191`, and `ESP_w003`. The text rules look for a soft or sinking brake
pedal, delayed braking, grinding or shuddering while braking, pulling to one side
while braking, heavy or unstable steering, and wiper problems that seriously
obstruct driver visibility.

Credible symptoms trigger the same escalation when the customer reports low
urgency or mentions that no dashboard warning appeared. The input treats blank
codes and `NONE` as missing. Code matching is case-insensitive and decisions
return the canonical spelling of a recognized code. The text rules match English
descriptions.

## Run the tests

Use Python 3.11 or later. From the repository root, install the package and test
dependencies in a virtual environment. On Windows PowerShell, run the following
commands.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest
~~~

On macOS or Linux, use `.venv/bin/python` for the last two commands. The tests
use short example messages and run independently of the Zero CSV.

## Classify a message

~~~python
from fault_urgency import FaultMessage, classify_fault_message

message = FaultMessage(
    message_id="example-001",
    alert_code="NONE",
    message_text="The brake pedal slowly sinks to the floor.",
    customer_reported_urgency="not urgent",
)

decision = classify_fault_message(message)
print(decision.model_dump_json(indent=2))
~~~

This report receives `safety_urgent` and `stop_driving_and_tow`. The decision
identifies the sinking brake pedal as evidence.
