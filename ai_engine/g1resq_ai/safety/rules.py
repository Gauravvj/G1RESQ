"""Rule-based safety layer: a deterministic urgency floor.

The LLM may raise urgency above the floor but never lower it. This guards
against the most dangerous failure (under-triage). It runs in well under a
millisecond, fully offline, so red flags can be shown before the model answers.

These lists are a starting point, not a finished medical vocabulary. Every rule
must be backed by a test, and the Hindi/Hinglish terms need review by a native
speaker. Matching is intentionally generous: over-triage is safer than missing
a life-threatening report.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from ..schemas import max_urgency


@dataclass(frozen=True)
class Rule:
    name: str
    floor: str                 # "critical" or "high"
    pattern: re.Pattern
    negatable: bool = False    # ignore the match if directly preceded by no/not/nobody...


def _rule(name: str, floor: str, regex: str, negatable: bool = False) -> Rule:
    return Rule(name, floor, re.compile(regex, re.IGNORECASE), negatable)


RULES: list[Rule] = [
    # ---------- critical: immediate threat to life ----------
    _rule("not_breathing", "critical",
          r"\b(?:not|isn'?t|doesn'?t|didn'?t|stopped|stops|no longer|can'?t|cannot|unable to|barely|hardly)"
          r"\s+(?:be\s+|able to\s+)?breath(?:e|es|ing)\b"),
    _rule("cardiac", "critical",
          r"\b(?:no\s+pulse|without\s+a\s+pulse|pulseless|cardiac\s+arrest|heart\s+attack|heart\s+stopped)\b"),
    _rule("unconscious", "critical",
          r"\b(?:unconscious|unresponsive|not\s+responding|not\s+responsive|not\s+waking|won'?t\s+wake)\b"),
    _rule("severe_bleeding", "critical",
          r"\b(?:bleeding\s+(?:heavily|badly|profusely|a\s+lot)|heavy\s+bleeding|severe\s+bleeding"
          r"|(?:lot|lots)\s+of\s+blood|blood\s+everywhere|bleeding\s+out"
          r"|won'?t\s+stop\s+bleeding|bleeding\s+won'?t\s+stop)\b", negatable=True),
    _rule("trapped", "critical",
          r"\b(?:trapped|stuck\s+under|pinned\s+(?:under|down)|buried|under\s+(?:the\s+)?(?:rubble|debris)"
          r"|can'?t\s+get\s+out|cannot\s+get\s+out)\b", negatable=True),
    _rule("drowning", "critical", r"\b(?:drown(?:ing|ed)?|swept\s+away)\b", negatable=True),
    # Hinglish (romanized Hindi)
    _rule("hinglish_breathing", "critical", r"\b(?:saans|sans|saas)\s+(?:nahi|nahin|nhi|band|ruk)"),
    _rule("hinglish_unconscious", "critical", r"\b(?:behosh|bezosh|hosh\s+nahi|hosh\s+nhi)\b"),
    _rule("hinglish_trapped", "critical",
          r"\b(?:phans\w*|malb[ae]|dab(?:a|e|i)?\s+(?:hua|hue|hui|gaya|gaye|gayi))\b"),
    _rule("hinglish_drowning", "critical", r"\bdoob\w*"),
    _rule("hinglish_heavy_blood", "critical", r"\b(?:bahut|bohot|bahot|zyada|jyada)\s+(?:khoon|khun)\b"),
    _rule("hinglish_heart", "critical", r"\bdil\s+ka\s+daura\b"),
    # Hindi (Devanagari). No \b here: word boundaries are unreliable around combining marks.
    _rule("hindi_critical", "critical",
          r"(?:सांस|साँस)\s*(?:नहीं|बंद|रुक)|बेहोश|फंस|मलबा|मलबे|डूब|दिल का दौरा|बहुत खून"),

    # ---------- high: serious, needs a fast response ----------
    _rule("bleeding", "high", r"\b(?:bleed(?:ing|s)?|blood)\b", negatable=True),
    _rule("fracture", "high",
          r"\b(?:fractur\w+|broken\s+(?:leg|arm|bone|bones|back|neck|rib|ribs|spine|skull))\b"),
    _rule("chest_pain", "high",
          r"\b(?:chest\s+(?:pain|tightness)|pain\s+in\s+(?:the\s+|my\s+|his\s+|her\s+)?chest)\b"),
    _rule("breathing_difficulty", "high",
          r"\b(?:(?:difficulty|trouble|problems?|struggling)\s+(?:in\s+|with\s+|to\s+)?breath(?:e|ing)"
          r"|hard\s+to\s+breathe|short(?:ness)?\s+of\s+breath|breathless\w*|gasping)\b"),
    _rule("seizure", "high", r"\b(?:seizures?|convulsion\w*)\b"),
    _rule("stroke", "high", r"\b(?:stroke|paralys\w+|slurred\s+speech)\b"),
    _rule("poisoning", "high", r"\b(?:poison\w*|overdos\w+)\b"),
    _rule("severe_burns", "high", r"\b(?:severe(?:ly)?\s+burn\w*|badly\s+burn\w*|burn\w*\s+badly)\b"),
    _rule("electric_shock", "high", r"\b(?:electrocut\w*|electric\s+shock|live\s+wire)\b"),
    _rule("snakebite", "high", r"\b(?:snake\s*bite\w*|bitten\s+by\s+(?:a\s+)?snake)\b"),
    _rule("fainting", "high",
          r"\b(?:faint(?:ed|ing)?|passed\s+out|(?:he|she|they|someone|somebody|person|man|woman|child|boy|girl"
          r"|baby|father|mother|brother|sister|husband|wife|grandfather|grandmother|papa|mummy)"
          r"\s+(?:has\s+|have\s+|just\s+|suddenly\s+)*collapsed)\b"),
    _rule("gas_or_blast", "high",
          r"\b(?:gas\s+leak\w*|explosion|exploded|blast|cylinder\s+(?:blast|burst|leak\w*))\b", negatable=True),
    _rule("fire_spreading", "high",
          r"\b(?:fire\s+(?:is\s+)?spreading|spreading\s+fire|on\s+fire|caught\s+fire)\b", negatable=True),
    _rule("structure_collapse", "high",
          r"\b(?:(?:building|house|roof|wall|bridge|ceiling|structure)\s+(?:has\s+|have\s+|just\s+)?collaps\w*"
          r"|collapsed\s+(?:building|house|roof|wall|bridge)|landslide|mudslide|cave-?in)\b", negatable=True),
    _rule("flood_rising", "high",
          r"\b(?:(?:water|flood\w*)\s+(?:level\s+)?(?:is\s+)?(?:rising|entering|entered)"
          r"|rising\s+(?:water|flood\w*)|flash\s+flood)\b"),
    _rule("hinglish_high", "high",
          r"\b(?:(?:khoon|khun)\b|aag\s+lag\w*|bhukamp|zalzala|dhamaka|haddi\s+(?:toot|tut)\w*"
          r"|paani\s+(?:bhar|badh|chadh)\w*)"),
    _rule("hindi_high", "high", r"खून|आग\s*लग|भूकंप|धमाका|हड्डी\s*टूट|पानी\s*(?:भर|बढ)"),
]

# "no", "not", "nobody"... directly before the match (at most 2 words between).
# A negation never reaches across "and", "but", "or"..., so "not breathing and
# bleeding" does not cancel the bleeding. "no" followed by a digit is an
# address ("house no 12"), not a negation.
_NEGATOR = re.compile(
    r"\b(?:no(?!\.?\s*\d)|not|nobody|never|without|none|nothing)\b"
    r"(?:\s+(?!(?:and|but|or|with|also|while|then|plus|because|so)\b)\w+){0,2}\s*$",
    re.IGNORECASE,
)


def _negated(text: str, start: int) -> bool:
    return bool(_NEGATOR.search(text[max(0, start - 40):start]))


def _clean(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\u2019", "'")


@dataclass
class SafetyResult:
    floor: Optional[str] = None
    red_flags: list[str] = field(default_factory=list)


def check_safety(text: str) -> SafetyResult:
    text = _clean(text)
    floor: Optional[str] = None
    flags: list[str] = []
    for rule in RULES:
        for match in rule.pattern.finditer(text):
            if rule.negatable and _negated(text, match.start()):
                continue
            floor = max_urgency(floor, rule.floor)
            flags.append(f"{rule.name}: {match.group(0)}")
            break
    return SafetyResult(floor=floor, red_flags=flags)