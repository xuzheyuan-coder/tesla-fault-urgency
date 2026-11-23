# Define the urgency rules for fault messages

A local Python classifier for the Zero Tesla service simulation. A validated
`FaultMessage` produces an `UrgencyDecision` with separate urgency and fix-path
fields, a reason, and the evidence behind the decision.

The simulated service team receives fault reports in arrival order. A serious
brake or steering issue can be buried among routine service requests. The
classifier identifies reports that need immediate attention and recommends a
service path based on the alert code and the customer's description.

## Decision rules

The classifier applies these rules from top to bottom.

| Evidence | Urgency | Fix path |
| --- | --- | --- |
| A recognized safety alert or a credible safety-critical symptom | `safety_urgent` | `stop_driving_and_tow` |
| An unclear brake, steering, or wiper visibility concern | `soon` | `earlier_service_appointment`, with human review required |
| `UI_a004` touchscreen or software issue | `routine` | `remote_diagnosis_or_ota` |
| `HVAC_w003` heating hardware, `CP_a004` charge port door or latch, `WIP_w001` ordinary wiper maintenance, or other messages without a safety or review signal | `routine` | `routine_service_appointment` |

### Safety alert codes

| System | Codes |
| --- | --- |
| Brakes | `BRK_w003`, `BRK_a001` |
| Anti-lock braking | `ABS_w002` |
| Electric power steering | `EPAS_w202`, `EPAS_a191` |
| Stability control | `ESP_w003` |

The input treats blank codes and `NONE` as missing. Code matching is
case-insensitive, and decisions return the canonical spelling of a recognized
code.

### Hidden safety symptoms

The text rules look for a soft or sinking brake pedal, delayed or ineffective
braking, grinding or shuddering while braking, pulling to one side while
braking, and heavy or unstable steering. Wiper faults also trigger
`safety_urgent` when the description indicates that driver visibility is
seriously obstructed.

Credible symptoms trigger escalation when the code is missing, when the
dashboard shows no warning, or when the customer describes the issue as
non-urgent. Ambiguous safety concerns go to human review through the `soon`
route.

## Models and architecture

Both models use Pydantic v2. Incorrect types, missing required fields, and
unexpected fields raise `ValidationError`. The decision model also validates
that the fix path is compatible with the urgency.

### Input — `FaultMessage`

| Field | Type and purpose |
| --- | --- |
| `message_id` | Required nonblank string identifying the message. |
| `alert_code` | Optional string, defaulting to `None`. |
| `message_text` | Required string containing the customer description or advisor notes. An empty string is accepted when an alert code is present. |
| `customer_reported_urgency` | Optional string that records the customer's assessment, defaulting to `None`. |

### Output — `UrgencyDecision`

| Field | Type and purpose |
| --- | --- |
| `urgency` | Required `UrgencyLevel` enum value. |
| `fix_path` | Required `FixPath` enum value. |
| `reason` | Required nonblank string explaining the decision. |
| `matched_alert_code` | The canonical recognized alert code, or `None`. |
| `matched_symptoms` | A list of matched safety symptoms, empty by default. |
| `human_review_required` | A boolean review flag, `False` by default. |

`models.py` defines and validates the input and output. `classifier.py` applies
the alert-code and routing rules through `classify_fault_message()`.
`symptoms.py` matches English symptom and review signals with context checks
for wording such as negated symptoms. The `tests/` directory covers both the
models and classifier.

## Install and test

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

Run this example in a Python session using the virtual environment.

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

## Verification results

All **86 pytest tests passed** on Windows with Python 3.15.0, Pydantic 2.14.4,
and pytest 9.2.0. The suite covers all six safety codes, hidden safety symptoms,
routine routes, human review, field validation, and JSON serialization.

The classifier was also run locally on all 300 records in
`Overnight Fault Message Batch (16 November 2026).csv`. The run mapped the
message ID, vehicle alert code, and description into `FaultMessage`. The
hand-checked safety labels were used to verify escalation.

| Batch result | Count |
| --- | ---: |
| Messages evaluated | 300 |
| `safety_urgent` decisions | 105 |
| `soon` decisions | 0 |
| `routine` decisions | 195 |
| Hand-checked safety faults escalated to `safety_urgent` and towing | 60 of 60 |
| Hand-checked safety faults with no alert code escalated | 37 of 37 |

### Limits of this verification

The text rules cover specific English phrasing. Other languages, unusual
wording, and incomplete descriptions can produce missed matches. The
original batch informed the text rules, so its results describe that batch.
The remaining 240 records lack human verification, which prevents a reliable
false-positive rate from being calculated. The raw batch stays local because
it includes customer-name and vehicle-identifier fields.

## Human-directed improvement

After the first AI-generated implementation, a follow-up request asked for a
validation audit and tests for incorrect types. The original models in local
commit `6ca3521` allowed Pydantic to coerce some invalid inputs. For example,
`UrgencyDecision.model_validate()` accepted the string `"false"` for
`human_review_required` and silently converted it to the boolean `False`.

The follow-up enabled strict Pydantic validation and added regression tests
for both models. The same input now raises a `ValidationError` with type
`bool_type` at `human_review_required`. An actual boolean remains valid, and the
decision accepts valid enum strings for JSON round trips. The change surfaces
malformed records at the model boundary before they enter the workflow.

## Original implementation prompt

The following is the original English development request from the project
conversation.

<details>
<summary>Read the development prompt</summary>

~~~markdown
Create a new Python repository that implements the Tesla fault-message urgency rules as a structured Pydantic v2 model.



Do not build a web application or connect to external APIs. Focus on a clean, testable Python rule engine.

### 1. Define urgency levels

Create an enum called `UrgencyLevel` with exactly three values:

- `safety_urgent`
- `soon`
- `routine`

### 2. Define four fix paths

Create an enum called `FixPath` with four values:

- `stop_driving_and_tow`
- `earlier_service_appointment`
- `remote_diagnosis_or_ota`
- `routine_service_appointment`

Keep urgency and fix path as separate fields.



The mapping should be:

- safety_urgent → stop_driving_and_tow
- soon → earlier_service_appointment
- routine → remote_diagnosis_or_ota OR routine_service_appointment, depending on the issue.

### 3. Define safety alert codes

The following six alert codes are associated with hand-checked safety faults:

- BRK_w003
- BRK_a001
- ABS_w002
- EPAS_w202
- EPAS_a191
- ESP_w003

These cover brake, ABS, electric power steering, and stability control issues.

### 4. Implement the hidden safety rule

The classification must examine both alert codes and actual message text.



A message must be classified as safety_urgent when it describes credible safety-critical symptoms, even if:

- No alert code is present.
- The alert code is NONE.
- The customer explicitly says the issue is not urgent.
- There is no dashboard warning.

Recognize symptoms including:

- Soft or sinking brake pedals.
- Delayed braking.
- Grinding or shuddering during braking.
- Vehicle pulling to one side during braking.
- Heavy or unstable steering.

Also, WIP_w001 is normally a routine maintenance code, but wiper problems that seriously obstruct driver visibility must be escalated to safety_urgent.



Do not let customer-reported urgency override actual safety symptoms.

### 5. Define non-safety categories

Use these codes and their intended treatment:

- HVAC_w003: Heating and climate control.
- UI_a004: Touchscreen, software, and user interface.
- CP_a004: Charge port door and latch.
- WIP_w001: Routine wiper blade and arm issues.

Treatment:

- UI_a004: Prefer remote diagnosis, software reset, or OTA troubleshooting.
- HVAC_w003: Normal service appointment for persistent hardware problems.
- CP_a004: Routine service appointment.
- WIP_w001: Routine service appointment unless visibility is seriously compromised.

### 6. Create Pydantic models

Create:



`FaultMessage`

- message_id
- alert_code (optional)
- message_text
- customer_reported_urgency (optional)

`UrgencyDecision`

- urgency
- fix_path
- reason
- matched_alert_code
- matched_symptoms
- human_review_required

Use Pydantic Field descriptions, enums, type validation, and model validators where appropriate.

### 7. Implement classification logic

Create a deterministic classification function that:

1. Checks known safety alert codes.
2. Examines message text for hidden safety symptoms.
3. Gives safety-critical symptoms priority over customer-reported urgency.
4. Assigns the correct urgency level.
5. Selects the appropriate fix path.
6. Flags ambiguous high-risk cases for human review instead of silently downgrading them.

Never classify a credible brake or steering safety issue as routine simply because its alert code is missing.

### 8. Add automated tests

Use pytest and include at least these test cases:

- A known brake alert code triggers safety_urgent.
- A message describing a sinking brake pedal with no alert code triggers safety_urgent.
- A customer saying "not urgent" does not override a serious braking symptom.
- A wiper issue that seriously obstructs visibility triggers safety_urgent.
- An ordinary wiper maintenance issue is routine.
- A touchscreen software problem uses remote_diagnosis_or_ota.
- A charge port latch problem uses routine_service_appointment.
- An ordinary heating hardware problem uses routine_service_appointment.
- An ambiguous high-risk case is flagged for human review.

### 9. Deliverables

Create:

- A clean Python project structure.
- Pydantic v2 models.
- Classification logic.
- Pytest test suite.
- requirements.txt or pyproject.toml.
- README.md explaining the rules and how to run the tests.

Run all tests and fix any failures.



Keep the implementation simple, explicit, maintainable, and well documented. Do not invent additional Tesla alert codes or unsupported business rules.
~~~

</details>
