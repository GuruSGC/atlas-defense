"""
Linguistic detectors used by the behavioral, source, and historical signals.

Each detector looks for a class of attack phrasing described in the memory-poisoning
literature (conditional "indication prompt" directives, impersonated authority,
exfiltration, unsafe code, passive-voice policy changes) instead of matching the exact
wording of any one test case. They are deliberately conservative about *who* the
sentence addresses: a user describing their own habit ("whenever I deploy, I run the
smoke tests") is not a directive to the assistant, whereas "whenever asked about X,
tell them Y" is.
"""

import re
from dataclasses import dataclass, field
from typing import List

# Verbs that change how the assistant answers or what it does on the user's behalf.
# Ordinary developer verbs (commit, push, deploy, use) are left out on purpose so team
# conventions such as "never commit directly to main" are not treated as injections.
_AGENT_VERBS = (
    r"tell|say|state|claim|report|answer|respond|reply|describe|suggest|recommend|advise|"
    r"approve|accept|deny|reject|acknowledge|mark|label|cite|treat|ignore|skip|forward|send|"
    r"store|process|reveal|disclose|hide|omit|round|read|confirm|pretend|assume|email|upload|"
    r"transmit|mention|refuse|decline|dismiss|mute"
)
_VERB_LEAD = r"(?:then\s+|please\s+|also\s+|quietly\s+|silently\s+|always\s+|never\s+|just\s+)*"

_CONDITIONAL = re.compile(
    r"\b(?:whenever|every time|each time|any ?time|if|when|once|in case)\b[^.;!?]{0,140}?,\s*"
    + _VERB_LEAD + r"(?:" + _AGENT_VERBS + r")\b",
    re.IGNORECASE,
)
_FROM_NOW_ON = re.compile(
    r"\b(?:from now on|going forward|henceforth|from this point on|starting now)\b,?\s*"
    + _VERB_LEAD + r"(?:" + _AGENT_VERBS + r")\b",
    re.IGNORECASE,
)
_STANDING = re.compile(
    r"(?:^|[.;:]\s+)(?:please\s+)?(?:always|never)\s+(?:" + _AGENT_VERBS + r")\b",
    re.IGNORECASE,
)
_FUTURE_TRIGGER = re.compile(
    r"\b(?:the next time|next time (?:i|you|we)|when i (?:talk|speak|ask)|if i ask you|whenever i ask|note for all future)\b",
    re.IGNORECASE,
)

_PASSIVE_POLICY = re.compile(
    r"\b(?:should|must|shall|can|may|could|will|needs? to|ought to|is to|are to)\s+"
    r"(?:now\s+|also\s+|simply\s+|just\s+|permanently\s+)*be\s+"
    r"(?:permanently\s+)?(?:disabled|removed|deleted|muted|suppressed|bypassed|skipped|ignored|waived|dropped|"
    r"wiped|paused|backdated|turned off|switched off|stopped|cancell?ed|decommissioned|relaxed|lifted|"
    r"allowed|approved without|attempted|overridden|exempt)",
    re.IGNORECASE,
)
_NO_FALLBACK = re.compile(r"\bno\b[^.]{0,40}\b(?:should|must|will)\s+(?:ever\s+)?be\s+attempted\b", re.IGNORECASE)

_ROLE = (
    r"ceo|cfo|cto|coo|ciso|admin(?:istrator)?|sysadmin|system administrator|security (?:lead|officer|team)|"
    r"incident commander|platform owner|manager|director|physician|doctor|dr\.?|nurse|pharmacist|accountant|"
    r"controller|supervisor|owner|officer|editor(?:-in-chief)?|head of [a-z]+|lead|relationship manager|"
    r"fraud department|it department|hr department|support team|customer support|compliance team|auditor"
)
_FIRST_PERSON_ROLE = re.compile(
    r"\b(?:as|i am|i'm|this is)\s+(?:the|a|an|your|our)?\s*(?:[\w'.-]+\s+){0,4}?(?:" + _ROLE + r")\b",
    re.IGNORECASE,
)
_THIRD_PARTY_ROLE = re.compile(
    r"\b(?:the|our|my|user's|by|from|per|dr\.?|mr\.?|ms\.?|mrs\.?)\s+(?:[\w'.-]+\s+){0,3}?(?:" + _ROLE + r")\b|\bdr\.?\s+[A-Z][a-z]+",
    re.IGNORECASE,
)
_APPROVAL = re.compile(
    r"\b(?:approv|authori[sz]|confirm|sign(?:ed|ing)? off|clear(?:ed|ing)?|permit|sanction|order(?:ed|ing)?|"
    r"instruct|direct(?:ed|ing)?|request(?:ed|ing)?|green-?lit|agree(?:d|s)?)\w*",
    re.IGNORECASE,
)
_RISKY_ACTION = re.compile(
    r"\b(?:remov(?:e|ed|ing)|delet(?:e|ed|ing)|disabl(?:e|ed|ing)|skip(?:ped|ping)?|bypass(?:ed|ing)?|"
    r"rais(?:e|ed|ing)|doubl(?:e|ed|ing)|backdat\w*|waiv(?:e|ed|ing)|paus(?:e|ed|ing)|suspend(?:ed|ing)?|"
    r"overrid(?:e|den|ing)|turn(?:ed|ing)? off|switch(?:ed|ing)? off|mut(?:e|ed|ing)|auto-?approv\w*|"
    r"revis(?:e|ed|ing) (?:upward|downward)|lower(?:ed|ing)? the (?:limit|threshold)|no longer require\w*|"
    r"reset(?:ting)?|unlimited)\b"
    r"|\bwithout (?:mfa|2fa|approval|review|confirmation|checking|verification)\b|\bdual approval\b",
    re.IGNORECASE,
)

_EXFIL = re.compile(
    r"\b(?:send|forward|email|mail|post|upload|transmit|copy|share|leak|exfiltrate|relay)\b[^.]{0,90}?"
    r"(?:\bto\b|\bat\b|\bvia\b)\s*(?:[\w-]+\s+){0,3}(?:https?://\S+|[\w.+-]+@[\w-]+\.[\w.-]+|www\.\S+|[\w-]+\.(?:com|net|org|io|ru|example)\b)",
    re.IGNORECASE,
)
_CODE_SAFETY = re.compile(
    r"\beval\s*\(|\bexec\s*\(|\bos\.system\b|\bsubprocess\.\w+|shell\s*=\s*true|\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r|"
    r"\bpickle\.loads?\b|\bchmod\s+(?:-r\s+)?777\b|\bdrop\s+table\b|(?:curl|wget)\s[^|]{0,120}\|\s*(?:ba|z)?sh\b|"
    r"\bbase64\s+-d\b|\bnc\s+-e\b|/etc/(?:passwd|shadow)",
    re.IGNORECASE,
)
_MEMORY_OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget|discard)\b\s+(?:any |all |the |your )?(?:earlier|previous|prior|older|existing)\s+"
    r"(?:notes?|memor(?:y|ies)|records?|instructions?|information|entries|rules)\b",
    re.IGNORECASE,
)
_INJECTION_MARKERS = re.compile(
    r"\[injected_override\]|always prepend|ignore (?:all|prior|previous) instructions|system prompt|admin override|"
    r"bypass safety|exfiltrate|\bdarknet\b|malicious-db|critical directive|security rule override|"
    r"override all|\bdelete_all_files\b|-rm\b|code to delete a file|developer mode|jailbreak|\bdan mode\b|"
    r"secret (?:rule|instruction|directive)",
    re.IGNORECASE,
)
_DESTRUCTIVE_IMPERATIVE = re.compile(
    r"(?:^|[.,;:]\s*|\bthen\s+|\band\s+)(?:please\s+|immediately\s+)*(?:delete|remove|erase|wipe|format|destroy|shut ?down|terminate|kill)\s+"
    r"(?:(?:all|every|the entire|any|everything)\b|(?:the\s+)?(?:project|environment|cluster|production|database|servers?|services?|instances?)\b|"
    r"(?:the\s+)?[\w']+\s+(?:environment|cluster|database|server|service|instance)\b)",
    re.IGNORECASE,
)
_SECURITY_OFF = re.compile(
    r"\b(?:disable|bypass|turn off|skip|circumvent|override|ignore)\b(?:\s+[\w-]+){0,3}?\s+"
    r"(?:security|authori[sz]ation|authentication|auth|checks?|scans?|mfa|2fa|controls?|safeguards?|guardrails?|filters?)\b|"
    r"\bsecurity scans? (?:are|is|were) disabled\b",
    re.IGNORECASE,
)


@dataclass
class Findings:
    flags: List[str] = field(default_factory=list)

    def add(self, name: str) -> None:
        if name not in self.flags:
            self.flags.append(name)

    @property
    def any(self) -> bool:
        return bool(self.flags)

    def __contains__(self, name: str) -> bool:
        return name in self.flags


def directive_findings(text: str) -> Findings:
    """Behavioral red flags: text that tries to steer the assistant or the system."""
    out = Findings()
    t = text.strip()
    if _CONDITIONAL.search(t):
        out.add("conditional_directive")
    if _FROM_NOW_ON.search(t):
        out.add("standing_directive")
    if _STANDING.search(t):
        out.add("standing_directive")
    if _FUTURE_TRIGGER.search(t):
        out.add("future_trigger")
    if _PASSIVE_POLICY.search(t) or _NO_FALLBACK.search(t):
        out.add("passive_policy_change")
    if (_FIRST_PERSON_ROLE.search(t) or _THIRD_PARTY_ROLE.search(t)) and _RISKY_ACTION.search(t):
        if _APPROVAL.search(t) or _FIRST_PERSON_ROLE.search(t):
            out.add("authority_with_risky_action")
    if _EXFIL.search(t):
        out.add("exfiltration")
    if _CODE_SAFETY.search(t):
        out.add("unsafe_code")
    if _MEMORY_OVERRIDE.search(t):
        out.add("memory_override")
    if _INJECTION_MARKERS.search(t):
        out.add("injection_marker")
    if _DESTRUCTIVE_IMPERATIVE.search(t):
        out.add("destructive_command")
    if _SECURITY_OFF.search(t):
        out.add("security_control_off")
    return out


def has_authority_claim(text: str) -> bool:
    """Self-identifying or third-party authority assertion (used by source reliability)."""
    return bool(_FIRST_PERSON_ROLE.search(text))


REVISION_MARKERS = [
    r"\bno longer\b", r"\bnot\b.{0,20}\banymore\b", r"\bwas changed to\b", r"\bchanged from\b",
    r"\bwas updated to\b", r"\breversed\b", r"\bmisdiagnos", r"\bquietly\b", r"\bwas moved\b",
    r"\bmoved up\b", r"\bwas switched\b", r"\bdisabled entirely\b", r"\bremoved from\b",
    r"\bincreased from\b", r"\bdecreased from\b", r"\brevised (upward|downward)\b",
    r"\bdiscontinu", r"\bstop taking\b", r"\bskip(ping|ped)?\b", r"\bwas reset to\b",
    r"\bcorrection\b", r"\bupdate:\s", r"\bswitch(ed)? to\b", r"\bwas raised to\b",
    r"\bwas increased\b", r"\bpermanently changed\b", r"\bpermanently disabled\b",
    r"\bwas approved\b", r"\brecommended (discontinuing|moving|switching|skipping)\b",
    r"\bwas (?:set|lowered|reduced|replaced|renamed|corrected|amended|revised|rewritten) (?:to|from|by)\b",
    r"\bwere (?:amended|corrected|revised|updated|changed|decommissioned|removed|discontinued)\b",
    r"\bas previously (?:stated|reported|noted|mentioned)\b", r"\bnot \w+(?: \w+)? as (?:previously|before|earlier)\b",
    r"\bdespite earlier\b", r"\bdecommission", r"\bwaived\b",
]
_REVISION = [re.compile(p, re.IGNORECASE) for p in REVISION_MARKERS]


def has_revision_language(text: str) -> bool:
    return any(p.search(text) for p in _REVISION)


# Preference-like and domain "slots". A contradiction needs the candidate and the stored
# memory to talk about the same slot with different values, otherwise a statement such as
# "the frontend is built with TypeScript" would clash with "the user prefers Python".
SLOT_WORDS = {
    "preference": r"\b(?:prefer\w*|favou?rite|default|primary|main|standard|chosen|choice)\b",
    "language": r"\b(?:language|programming|coding)\b",
    "database": r"\b(?:database|db)\b",
    "deployment": r"\b(?:deploy\w*|target|hosting|production)\b",
    "region": r"\b(?:region|location|datacenter)\b",
    "style": r"\b(?:style|format\w*|indent\w*)\b",
}
_SLOT_RE = {k: re.compile(v, re.IGNORECASE) for k, v in SLOT_WORDS.items()}


def slots_in(text: str) -> set:
    return {k for k, r in _SLOT_RE.items() if r.search(text)}


def shares_slot(a: str, b: str) -> bool:
    sa, sb = slots_in(a), slots_in(b)
    if sa & sb:
        return True
    # "default language" vs "prefers X" are both statements about a stored preference
    return bool({"preference", "language"} & sa and {"preference", "language"} & sb)


_STOP = {
    "the", "and", "for", "with", "that", "this", "these", "those", "from", "have", "has", "had", "will", "would",
    "should", "could", "must", "shall", "than", "then", "when", "whenever", "also", "into", "over", "under", "about",
    "after", "before", "still", "just", "only", "more", "most", "very", "user", "users", "their", "them", "they",
    "your", "yours", "ours", "were", "been", "being", "does", "doing", "done", "such", "some", "same", "other",
    "each", "every", "both", "while", "where", "which", "what", "who", "whom", "whose", "there", "here",
}


def _stem(w: str) -> str:
    for suf in ("ings", "ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def content_words(text: str) -> set:
    words = re.findall(r"[a-z][a-z0-9+#.-]*", text.lower())
    return {_stem(w.strip(".-")) for w in words if len(w) >= 4 and w not in _STOP}


def is_related(a: str, b: str) -> bool:
    """Backend-independent relatedness: a shared slot or a shared content word.

    Cosine similarity is not used here because both the offline hash embedding and the
    live embedding model give unrelated sentences a high similarity, so no fixed
    threshold separates related from unrelated text."""
    return shares_slot(a, b) or bool(content_words(a) & content_words(b))
