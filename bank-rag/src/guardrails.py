"""Input/output guardrails: PII masking, prompt-injection screening, topic restriction, citation check."""
import re

PII_PATTERNS = {
    "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "CARD": re.compile(r"\b\d(?:[ -]?\d){12,15}\b"),
    "PHONE": re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "IFSC": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
}

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"disregard (the |your )?(system|previous) (prompt|instructions)",
    r"reveal (your |the )?(system )?prompt",
    r"you are now",
    r"pretend (to be|you are)",
    r"developer mode",
    r"jailbreak",
]

# Light keyword gate for banking scope. Retrieval score is the second gate (see pipeline).
BANKING_HINTS = {
    "loan", "emi", "interest", "account", "savings", "card", "credit", "debit", "upi", "kyc",
    "fee", "charge", "charges", "penalty", "limit", "refund", "dispute", "fraud", "cheque",
    "neft", "rtgs", "imps", "deposit", "fd", "tenure", "prepayment", "foreclosure", "bank",
    "statement", "grievance", "complaint", "ombudsman", "rbi", "balance", "minimum", "atm",
    "transaction", "reward", "points", "cashback", "gst", "closure", "nominee", "mobile",
}

REFUSAL_OFF_TOPIC = "I can only answer questions about the bank's products, policies and charges."
REFUSAL_INJECTION = "I can't follow that request. Please ask a question about banking products or policies."
NO_ANSWER = "I couldn't find this in the available documents. Please contact customer care for help."


def mask_pii(text: str) -> tuple[str, list[str]]:
    found = []
    # order matters: mask card/aadhaar before phone so digit runs are not double-matched
    for label in ["CARD", "AADHAAR", "PAN", "IFSC", "EMAIL", "PHONE"]:
        text, n = PII_PATTERNS[label].subn(f"[{label}]", text)
        if n:
            found.append(label)
    return text, found


def is_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)


def looks_on_topic(text: str) -> bool:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(words & BANKING_HINTS)


def check_citations(answer: str, n_sources: int) -> bool:
    """Every [n] in the answer must point at a real source, and there must be at least one."""
    cites = [int(c) for c in re.findall(r"\[(\d+)\]", answer)]
    return bool(cites) and all(1 <= c <= n_sources for c in cites)
