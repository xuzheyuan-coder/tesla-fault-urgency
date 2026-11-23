"""Match reported safety symptoms in English customer and advisor messages."""

import re
from collections.abc import Iterable

_CLAUSE_BREAK = re.compile(r"[.!?;\n]+|\b(?:but|however)\b")
_NEGATION = r"(?:no|not|never|without|\w+n't)"
_WORD_GAP = rf"(?:\s+(?!(?:{_NEGATION}|and|or)\b)[\w'-]+){{0,5}}\s+"
_NEGATED_PREFIX = re.compile(
    rf"\b{_NEGATION}\s+"
    r"(?:(?:any|longer|signs? of|notice(?:d)?|feel(?:ing)?|have|"
    r"hear(?:d)?|report(?:ed)?|experienc(?:e|ed|ing))\s+)*"
    r"(?:[\w'-]+\s+(?:or|nor)\s+)?$"
)

_PEDAL = r"\b(?<!accelerator )(?<!gas )(?<!throttle )(?<!clutch )(?:brake )?pedal\b"
_SOFT = r"(?:soft(?:er)?|spongy|spongier|mushy)"
_SINKING = (
    r"(?:sink(?:s|ing)?|sank|lower than usual|"
    r"(?:go(?:es|ing)?|went|drop(?:s|ped|ping)?) "
    r"(?:(?:all the way |straight )?to the floor|(?:down )?too far))"
)
_STEERING = r"\bsteering(?: wheel)?\b"
_STEERING_STATE = (
    r"(?:heav(?:y|ier)|stiff(?:er)?|unstable|loose|unresponsive|"
    r"wobbl(?:e|es|ed|ing)|shimm(?:y|ies|ied|ying)|"
    r"jerk(?:s|ed|ing)?|lock(?:s|ed|ing) up)"
)
_VEHICLE_STOP = (
    r"(?:slow down|stop(?=\s*(?:$|,)| (?:at|when|before|in time|quickly|properly|right away)\b))"
)

_DIRECT_SYMPTOMS = (
    (
        "soft brake pedal",
        re.compile(rf"{_PEDAL}{_WORD_GAP}{_SOFT}\b|\b{_SOFT}\s+{_PEDAL}"),
    ),
    (
        "sinking brake pedal",
        re.compile(rf"{_PEDAL}{_WORD_GAP}{_SINKING}\b|\bsinking\s+{_PEDAL}"),
    ),
    (
        "delayed or ineffective braking",
        re.compile(
            r"\b(?:delay(?:ed)?|lag) (?:in |when )?braking\b"
            rf"|\bbrak(?:e|es|ing){_WORD_GAP}(?:delay(?:ed|s)?|lag(?:s|ged|ging)?|fail(?:s|ed)?)\b"
            rf"|\b(?:stopping|braking|brakes?){_WORD_GAP}"
            r"(?:takes?|took) (?:too long|(?:a (?:bit|lot) |much )?longer)\b"
            rf"|{_PEDAL}{_WORD_GAP}(?:go(?:es|ing)?|went) (?:a bit |really )?long\b"
            rf"|\b(?:car|vehicle) (?:won't|wouldn't|doesn't|didn't|cannot|can't) {_VEHICLE_STOP}\b"
            r"|\b(?:can't|cannot|couldn't|could not) stop (?:the|my) (?:car|vehicle)\b"
            r"|\b(?:brakes (?:don't|didn't|won't) work|lost (?:the )?brakes|lost braking)\b"
        ),
    ),
    (
        "heavy or unstable steering",
        re.compile(
            rf"{_STEERING}{_WORD_GAP}{_STEERING_STATE}\b|\b{_STEERING_STATE}\s+{_STEERING}"
            r"|\b(?:hard|difficult|unable) to steer\b"
            r"|\b(?:cannot|can't|could not|couldn't) (?:steer|turn the (?:steering )?wheel)\b"
            r"|\b(?:lost|loss of) (?:electric )?power steering\b"
        ),
    ),
)

_BRAKING_CONTEXT = re.compile(
    rf"\bbrak(?:e|es|ed|ing)\b|{_PEDAL}"
    r"|\bslow(?:s|ed|ing)? (?:down|for the lights)\b"
    r"|\b(?:com(?:e|es|ing)|came) to (?:a )?(?:stop|halt)\b"
    r"|\b(?:when|while|as) (?:i |we |the car |the vehicle )?stop(?:s|ped|ping)?\b"
    r"|\bstopped at (?:the )?(?:traffic )?lights\b"
)
_PULLING = re.compile(
    r"\b(?:pull(?:s|ed|ing)?|tug(?:s|ged|ging)?|drift(?:s|ed|ing)?|veer(?:s|ed|ing)?) "
    r"(?:(?:gently|slightly|a little|a bit|hard) )?(?:(?:to|towards?) )?(?:the )?"
    r"(?:left|right|one side|one direction)\b"
)
_BRAKING_SYMPTOMS = (
    (
        "delayed or ineffective braking",
        re.compile(
            rf"\b(?:it|car|vehicle) (?:wouldn't|doesn't|didn't) {_VEHICLE_STOP}\b"
            r"|\b(?:kept rolling|took (?:a few|several) seconds to stop)\b"
        ),
    ),
    (
        "grinding during braking",
        re.compile(r"\b(?:grind(?:s|ing)?|scrap(?:e|es|ing))\b"),
    ),
    (
        "shuddering during braking",
        re.compile(r"\b(?:shudder(?:s|ed|ing)?|vibrat(?:e|es|ed|ing|ion))\b"),
    ),
    ("pulling to one side during braking", _PULLING),
)
_REPEATED_STEERING_CORRECTION = re.compile(
    r"\b(?:have to|need to) (?:keep correcting|constantly correct)\b"
)

_WIPERS = re.compile(r"\bwipers?\b")
_WIPER_PROBLEM = re.compile(
    r"\b(?:fail(?:ed|s|ing|ure)?|stopp?(?:ed|s|ing)?|quit|died|stuck|brok(?:e|en)|"
    r"(?:won't|no longer) (?:work|move|clear)|(?:can't|couldn't) clear|not (?:working|clearing)|"
    r"smear(?:s|ed|ing)?|streak(?:s|ed|ing)?|"
    r"block(?:s|ed|ing)?|obstruct(?:s|ed|ing)?|obscur(?:e|es|ed|ing))\b"
)
_VIEW = r"(?:visibility|view|vision|line of sight)"
_SEVERE_VISIBILITY = re.compile(
    r"\b(?:can(?:not|'t)|could(?: not|n't)|unable to|barely|hardly|impossible to) "
    r"(?:clearly )?see\b(?:\s+(?:the )?(?:road(?:way)?|traffic|ahead|anything|out|"
    r"at all|in front|where i(?:'m| am) going|through (?:the |my )?(?:windshield|"
    r"windscreen|glass))\b|\s*$)"
    rf"|\b{_VIEW}{_WORD_GAP}(?:(?:severely|seriously|badly|completely) "
    r"(?:impaired|reduced|compromised)|dangerously (?:poor|low)|blocked|obscured|obstructed|zero)\b"
    rf"|\b(?:block(?:s|ed|ing)?|obstruct(?:s|ed|ing)?|obscur(?:e|es|ed|ing)) (?:most of )?"
    rf"(?:my |the |the driver's )?{_VIEW}\b"
)
_UNCLEAR_VISIBILITY = re.compile(
    r"\b(?:poor|reduced|limited) visibility\b"
    r"|\bvisibility (?:is|was) (?:poor|reduced|limited)\b"
    r"|\b(?:harder|difficult) to see (?:the road|ahead)\b"
)

_RISK_SYSTEM = (
    r"(?:brak(?:e|es|ing)(?: pedal)?|(?:power )?steering(?: wheel)?|abs|stability control)"
)
_VAGUE_CONCERN = (
    r"(?:issue|problem|concern|warning|"
    r"(?:feels?|felt|seems?|seemed) (?:off|odd|wrong|weird|different|strange|unusual|not right))"
)
_UNCLEAR_CONTROL = re.compile(
    rf"\b{_RISK_SYSTEM}{_WORD_GAP}{_VAGUE_CONCERN}\b"
    rf"|\b(?:warning|issue|problem|concern|something wrong) (?:with|about|for) "
    rf"(?:(?:the|my) )?{_RISK_SYSTEM}\b"
)


def _reports(pattern: re.Pattern[str], clauses: Iterable[str]) -> bool:
    return any(
        not _NEGATED_PREFIX.search(clause, 0, match.start())
        for clause in clauses
        for match in pattern.finditer(clause)
    )


def assess_message_text(
    message_text: str, *, wiper_alert: bool
) -> tuple[list[str], list[str]]:
    """Return confirmed symptom matches and any remaining signals for review."""
    normalized = message_text.casefold().replace("\u2019", "'")
    clauses = [" ".join(clause.split()) for clause in _CLAUSE_BREAK.split(normalized)]
    braking_clauses = [clause for clause in clauses if _BRAKING_CONTEXT.search(clause)]

    symptoms = [name for name, pattern in _DIRECT_SYMPTOMS if _reports(pattern, clauses)]
    symptoms.extend(
        name
        for name, pattern in _BRAKING_SYMPTOMS
        if name not in symptoms and _reports(pattern, braking_clauses)
    )
    if (
        "heavy or unstable steering" not in symptoms
        and _reports(_PULLING, clauses)
        and _reports(_REPEATED_STEERING_CORRECTION, clauses)
    ):
        symptoms.append("heavy or unstable steering")

    wiper_issue = wiper_alert or any(
        _reports(_WIPERS, (clause,)) and _reports(_WIPER_PROBLEM, (clause,))
        for clause in clauses
    )
    if wiper_issue and _reports(_SEVERE_VISIBILITY, clauses):
        symptoms.append("seriously obstructed driver visibility")

    review_signals = []
    if not symptoms:
        if _reports(_UNCLEAR_CONTROL, clauses):
            review_signals.append("brake or steering safety concern")
        if wiper_issue and _reports(_UNCLEAR_VISIBILITY, clauses):
            review_signals.append("wiper visibility concern")
    return symptoms, review_signals
