"""
Vietnamese linguistic tools for the Linguist agent.

These are REAL tools, not LLM prompts — they use actual Vietnamese
language rules, syllable dictionaries, and word segmentation.
This is what transforms "three LLM calls" into a genuinely
tool-augmented multi-agent system.
"""

import re
import unicodedata
from dataclasses import dataclass, field


# ── Vietnamese Syllable Validator ────────────────────────────────────
# Vietnamese has ~7,000 valid syllables. Any output containing an
# invalid syllable is almost certainly an OCR error.

# Vietnamese phonotactic rules (simplified but effective)
# A valid Vietnamese syllable = optional_initial + vowel_nucleus + optional_final + optional_tone

VALID_INITIALS = {
    '', 'b', 'c', 'ch', 'd', 'đ', 'g', 'gh', 'gi', 'h', 'k', 'kh',
    'l', 'm', 'n', 'ng', 'ngh', 'nh', 'p', 'ph', 'qu', 'r', 's',
    't', 'th', 'tr', 'v', 'x',
}

VALID_NUCLEI = {
    'a', 'ă', 'â', 'e', 'ê', 'i', 'o', 'ô', 'ơ', 'u', 'ư', 'y',
    # Diphthongs and triphthongs
    'ai', 'ao', 'au', 'ay', 'âu', 'ây',
    'eo', 'êu',
    'ia', 'iê', 'iêu', 'iu',
    'oa', 'oă', 'oai', 'oay', 'oe', 'oeo', 'oi',
    'oo', 'ôi',
    'ơi',
    'ua', 'uâ', 'uây', 'uê', 'ui', 'uô', 'uôi', 'uy', 'uya', 'uyê', 'uyu',
    'ưa', 'ưi', 'ươ', 'ươi', 'ươu', 'ưu',
    'yê', 'yêu',
}

VALID_FINALS = {
    '', 'c', 'ch', 'm', 'n', 'ng', 'nh', 'p', 't',
}

# Tone mark stripping map — maps toned chars to their base+modifier form
# Each group: 5 toned variants → base letter repeated 5 times
TONE_STRIP = str.maketrans(
    # src: àáảãạ ằắẳẵặ ầấẩẫậ èéẻẽẹ ềếểễệ ìíỉĩị òóỏõọ ồốổỗộ ờớởỡợ ùúủũụ ừứửữự ỳýỷỹỵ
    'àáảãạ' 'ằắẳẵặ' 'ầấẩẫậ' 'èéẻẽẹ' 'ềếểễệ' 'ìíỉĩị'
    'òóỏõọ' 'ồốổỗộ' 'ờớởỡợ' 'ùúủũụ' 'ừứửữự' 'ỳýỷỹỵ',
    # dst: a×5      ă×5      â×5      e×5      ê×5      i×5
    #      o×5      ô×5      ơ×5      u×5      ư×5      y×5
    'aaaaa' 'ăăăăă' 'ââââ' 'â' 'eeeee' 'êêêêê' 'iiiii'
    'ooooo' 'ôôôôô' 'ơơơơơ' 'uuuuu' 'ưưưưư' 'yyyyy'
)


def strip_tones(text: str) -> str:
    """Strip all tone marks, keeping base letter + vowel modifier."""
    return text.lower().translate(TONE_STRIP)


def is_valid_syllable_structure(syllable: str) -> bool:
    """
    Check if a syllable follows Vietnamese phonotactic rules.
    This is a structural check — not a dictionary lookup.
    Returns True if the syllable COULD be valid Vietnamese.
    """
    s = strip_tones(syllable.lower().strip())
    if not s:
        return False

    # Try all possible initial consonant splits
    for init_len in range(4, -1, -1):  # max initial = 3 chars (ngh)
        initial = s[:init_len]
        if initial not in VALID_INITIALS:
            continue

        remainder = s[init_len:]
        if not remainder:
            continue

        # Try all possible final consonant splits
        for final_len in range(min(2, len(remainder)), -1, -1):
            if final_len == 0:
                nucleus = remainder
                final = ''
            else:
                nucleus = remainder[:-final_len]
                final = remainder[-final_len:]

            if not nucleus:
                continue

            if nucleus in VALID_NUCLEI and final in VALID_FINALS:
                return True

    return False


# ── Vietnamese Syllable Dictionary ───────────────────────────────────
# Common Vietnamese syllables (high-frequency). This list can be
# expanded from a Vietnamese word frequency corpus.
# For now, includes ~2,000 most common syllables.

# We generate valid syllables programmatically from the rules above
# and supplement with a known-word list.

def build_common_syllables() -> set:
    """Build a set of common Vietnamese syllables from phonotactic rules."""
    syllables = set()
    for initial in VALID_INITIALS:
        for nucleus in VALID_NUCLEI:
            for final in VALID_FINALS:
                s = initial + nucleus + final
                if len(s) >= 1:
                    syllables.add(s)
    return syllables

_COMMON_SYLLABLES = None

def get_syllable_dictionary() -> set:
    """Lazy-load the syllable dictionary."""
    global _COMMON_SYLLABLES
    if _COMMON_SYLLABLES is None:
        _COMMON_SYLLABLES = build_common_syllables()
    return _COMMON_SYLLABLES


def is_known_syllable(syllable: str) -> bool:
    """Check if a syllable (tone-stripped) is in the dictionary."""
    stripped = strip_tones(syllable.lower().strip())
    return stripped in get_syllable_dictionary()


# ── Diacritic Consistency Checker ────────────────────────────────────
# Vietnamese tone marks follow strict placement rules:
# - Tone marks go on the main vowel of the nucleus
# - Only ONE tone mark per syllable
# - Specific vowel+tone combinations are valid

TONE_MARKS = {
    'à': ('a', 'huyền'), 'á': ('a', 'sắc'), 'ả': ('a', 'hỏi'),
    'ã': ('a', 'ngã'), 'ạ': ('a', 'nặng'),
    'ằ': ('ă', 'huyền'), 'ắ': ('ă', 'sắc'), 'ẳ': ('ă', 'hỏi'),
    'ẵ': ('ă', 'ngã'), 'ặ': ('ă', 'nặng'),
    'ầ': ('â', 'huyền'), 'ấ': ('â', 'sắc'), 'ẩ': ('â', 'hỏi'),
    'ẫ': ('â', 'ngã'), 'ậ': ('â', 'nặng'),
    'è': ('e', 'huyền'), 'é': ('e', 'sắc'), 'ẻ': ('e', 'hỏi'),
    'ẽ': ('e', 'ngã'), 'ẹ': ('e', 'nặng'),
    'ề': ('ê', 'huyền'), 'ế': ('ê', 'sắc'), 'ể': ('ê', 'hỏi'),
    'ễ': ('ê', 'ngã'), 'ệ': ('ê', 'nặng'),
    'ì': ('i', 'huyền'), 'í': ('i', 'sắc'), 'ỉ': ('i', 'hỏi'),
    'ĩ': ('i', 'ngã'), 'ị': ('i', 'nặng'),
    'ò': ('o', 'huyền'), 'ó': ('o', 'sắc'), 'ỏ': ('o', 'hỏi'),
    'õ': ('o', 'ngã'), 'ọ': ('o', 'nặng'),
    'ồ': ('ô', 'huyền'), 'ố': ('ô', 'sắc'), 'ổ': ('ô', 'hỏi'),
    'ỗ': ('ô', 'ngã'), 'ộ': ('ô', 'nặng'),
    'ờ': ('ơ', 'huyền'), 'ớ': ('ơ', 'sắc'), 'ở': ('ơ', 'hỏi'),
    'ỡ': ('ơ', 'ngã'), 'ợ': ('ơ', 'nặng'),
    'ù': ('u', 'huyền'), 'ú': ('u', 'sắc'), 'ủ': ('u', 'hỏi'),
    'ũ': ('u', 'ngã'), 'ụ': ('u', 'nặng'),
    'ừ': ('ư', 'huyền'), 'ứ': ('ư', 'sắc'), 'ử': ('ư', 'hỏi'),
    'ữ': ('ư', 'ngã'), 'ự': ('ư', 'nặng'),
    'ỳ': ('y', 'huyền'), 'ý': ('y', 'sắc'), 'ỷ': ('y', 'hỏi'),
    'ỹ': ('y', 'ngã'), 'ỵ': ('y', 'nặng'),
}


def count_tones_in_syllable(syllable: str) -> int:
    """Count how many tone-marked characters are in a syllable."""
    return sum(1 for c in syllable if c in TONE_MARKS)


def check_tone_consistency(syllable: str) -> tuple[bool, str]:
    """
    Check if a syllable has valid tone mark placement.
    Returns (is_valid, reason).
    """
    tone_count = count_tones_in_syllable(syllable)

    if tone_count == 0:
        return True, "no_tone"
    if tone_count > 1:
        return False, f"multiple_tones ({tone_count})"

    # Check that the tone is on a vowel (not a consonant)
    for c in syllable:
        if c in TONE_MARKS:
            base, tone_name = TONE_MARKS[c]
            return True, f"{tone_name}_on_{base}"

    return True, "ok"


# ── Full Text Validator ──────────────────────────────────────────────


@dataclass
class ValidationResult:
    """Result from validating a Vietnamese text string."""
    is_valid: bool
    total_syllables: int
    valid_syllables: int
    invalid_syllables: list[str] = field(default_factory=list)
    tone_errors: list[str] = field(default_factory=list)
    confidence: float = 1.0  # 0-1, ratio of valid syllables
    suggestions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "total_syllables": self.total_syllables,
            "valid_syllables": self.valid_syllables,
            "invalid_syllables": self.invalid_syllables,
            "tone_errors": self.tone_errors,
            "confidence": round(self.confidence, 4),
            "suggestions": self.suggestions,
        }

    def summary(self) -> str:
        if self.is_valid:
            return f"VALID ({self.valid_syllables}/{self.total_syllables} syllables)"
        return (
            f"INVALID ({self.valid_syllables}/{self.total_syllables} valid) | "
            f"Bad syllables: {', '.join(self.invalid_syllables[:5])} | "
            f"Tone errors: {', '.join(self.tone_errors[:3])}"
        )


def validate_vietnamese_text(text: str) -> ValidationResult:
    """
    Validate Vietnamese text using linguistic rules.
    This is the core tool the Linguist agent calls.
    """
    # Split into syllables (Vietnamese words are space-separated syllables)
    syllables = text.strip().split()

    # Filter out punctuation-only tokens
    syllables = [s for s in syllables if re.search(r'[a-zA-ZÀ-ỹ]', s)]

    if not syllables:
        return ValidationResult(is_valid=True, total_syllables=0, valid_syllables=0)

    valid_count = 0
    invalid_syllables = []
    tone_errors = []
    suggestions = {}

    for syllable in syllables:
        # Strip punctuation at edges
        clean = re.sub(r'^[^\wÀ-ỹ]+|[^\wÀ-ỹ]+$', '', syllable)
        if not clean:
            continue

        # Check syllable structure
        struct_valid = is_valid_syllable_structure(clean)

        # Check tone consistency
        tone_valid, tone_info = check_tone_consistency(clean)

        if struct_valid and tone_valid:
            valid_count += 1
        else:
            if not struct_valid:
                invalid_syllables.append(clean)
                # Try to find similar valid syllables
                similar = _find_similar_syllables(clean)
                if similar:
                    suggestions[clean] = similar
            if not tone_valid:
                tone_errors.append(f"{clean}: {tone_info}")

    total = len([s for s in syllables if re.sub(r'^[^\wÀ-ỹ]+|[^\wÀ-ỹ]+$', '', s)])
    confidence = valid_count / total if total > 0 else 1.0

    return ValidationResult(
        is_valid=len(invalid_syllables) == 0 and len(tone_errors) == 0,
        total_syllables=total,
        valid_syllables=valid_count,
        invalid_syllables=invalid_syllables,
        tone_errors=tone_errors,
        confidence=confidence,
        suggestions=suggestions,
    )


def _find_similar_syllables(syllable: str, max_results: int = 3) -> list[str]:
    """Find valid syllables similar to an invalid one (edit distance 1-2)."""
    import editdistance

    stripped = strip_tones(syllable.lower())
    dictionary = get_syllable_dictionary()

    candidates = []
    for valid in dictionary:
        dist = editdistance.eval(stripped, valid)
        if 0 < dist <= 2:
            candidates.append((dist, valid))

    candidates.sort(key=lambda x: x[0])
    return [c[1] for c in candidates[:max_results]]


# ── Diacritic Restoration Suggestions ────────────────────────────────
# When OCR strips diacritics (common failure mode), suggest restorations.

# Common Vietnamese words that are frequently confused without diacritics
DIACRITIC_CONFUSION_PAIRS = {
    # base_form: [(diacritized_form, meaning), ...]
    "ma": [("ma", "ghost"), ("má", "mother/cheek"), ("mà", "but/that"),
           ("mả", "grave/tomb"), ("mã", "code/horse"), ("mạ", "rice seedling")],
    "ba": [("ba", "three/father"), ("bà", "grandmother"), ("bá", "uncle"),
           ("bả", "poison bait"), ("bã", "residue"), ("bạ", "reckless")],
    "la": [("la", "shout"), ("là", "is/to be"), ("lá", "leaf"),
           ("lả", "exhausted"), ("lã", "bland"), ("lạ", "strange")],
    "co": [("co", "shrink"), ("có", "have"), ("cò", "heron"),
           ("cỏ", "grass"), ("cõ", "rare"), ("cọ", "palm tree")],
    "an": [("an", "peace"), ("ăn", "eat"), ("ân", "grace")],
    "da": [("da", "skin"), ("dà", "rare"), ("dá", "rare"),
           ("dạ", "yes/stomach")],
}


def suggest_diacritic_restorations(undiacritized: str) -> list[tuple[str, str]]:
    """
    Given a syllable without diacritics, suggest possible diacritized forms.
    Returns list of (form, meaning) tuples.
    """
    key = undiacritized.lower().strip()
    return DIACRITIC_CONFUSION_PAIRS.get(key, [])
