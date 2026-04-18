"""
Evaluation metrics for Vietnamese HTR.
Includes standard CER/WER plus novel DER (Diacritic Error Rate).

DER isolates diacritic-specific accuracy — the metric that matters most
for Vietnamese where a missing tone mark changes word meaning entirely.
"""

import unicodedata
from dataclasses import dataclass
from jiwer import cer, wer
import editdistance


# Vietnamese diacritic character sets
VIETNAMESE_DIACRITICS = set(
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợ"
    "ùúủũụưứừửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
    "ÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"
)

# Tone marks only (sắc, huyền, hỏi, ngã, nặng)
TONE_MARK_CHARS = set(
    "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợ"
    "ùúủũụừứửữựỳýỷỹỵ"
    "ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢ"
    "ÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ"
)

# Vowel modifier chars (ă, â, ê, ô, ơ, ư, đ) — base forms without tone
VOWEL_MODIFIER_CHARS = set("ăâêôơưđĂÂÊÔƠƯĐ")

# Base letter to all its diacritical variants
DIACRITIC_FAMILIES = {
    'a': 'àáảãạăắằẳẵặâấầẩẫậ',
    'e': 'èéẻẽẹêếềểễệ',
    'i': 'ìíỉĩị',
    'o': 'òóỏõọôốồổỗộơớờởỡợ',
    'u': 'ùúủũụưứừửữự',
    'y': 'ỳýỷỹỵ',
    'd': 'đ',
}


@dataclass
class HTRMetrics:
    """Complete metrics for a Vietnamese HTR evaluation."""
    cer: float          # Character Error Rate (standard)
    wer: float          # Word Error Rate (standard)
    der: float          # Diacritic Error Rate (proposed)
    tone_er: float      # Tone mark error rate
    modifier_er: float  # Vowel modifier error rate
    n_samples: int
    n_chars: int
    n_diacritic_chars: int
    n_tone_chars: int

    def to_dict(self) -> dict:
        return {
            "CER": f"{self.cer:.4f}",
            "WER": f"{self.wer:.4f}",
            "DER": f"{self.der:.4f}",
            "Tone_ER": f"{self.tone_er:.4f}",
            "Modifier_ER": f"{self.modifier_er:.4f}",
            "n_samples": self.n_samples,
            "n_chars": self.n_chars,
            "n_diacritic_chars": self.n_diacritic_chars,
        }

    def __str__(self) -> str:
        lines = [
            f"CER:         {self.cer:.4f} ({self.cer*100:.2f}%)",
            f"WER:         {self.wer:.4f} ({self.wer*100:.2f}%)",
            f"DER:         {self.der:.4f} ({self.der*100:.2f}%)",
            f"Tone ER:     {self.tone_er:.4f} ({self.tone_er*100:.2f}%)",
            f"Modifier ER: {self.modifier_er:.4f} ({self.modifier_er*100:.2f}%)",
            f"Samples: {self.n_samples} | Chars: {self.n_chars} | Diacritic chars: {self.n_diacritic_chars}",
        ]
        return "\n".join(lines)


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """Standard Character Error Rate."""
    if not predictions or not references:
        return 1.0
    return cer(references, predictions)


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """Standard Word Error Rate."""
    if not predictions or not references:
        return 1.0
    return wer(references, predictions)


def compute_der(predictions: list[str], references: list[str]) -> float:
    """
    Diacritic Error Rate (DER) — proposed metric.

    Computes edit distance only over diacritic-bearing characters,
    ignoring non-diacritical characters entirely. This isolates
    the diacritic-specific accuracy that matters most for Vietnamese.

    Method:
    1. Extract only diacritic characters from both prediction and reference
       (preserving order)
    2. Compute character-level edit distance on these filtered sequences
    3. Normalize by the number of diacritic characters in the reference
    """
    total_ref_diacritics = 0
    total_edit_distance = 0

    for pred, ref in zip(predictions, references):
        # Extract diacritic characters only
        ref_diacritics = [c for c in ref if c in VIETNAMESE_DIACRITICS]
        pred_diacritics = [c for c in pred if c in VIETNAMESE_DIACRITICS]

        total_ref_diacritics += len(ref_diacritics)

        if ref_diacritics:
            dist = editdistance.eval(ref_diacritics, pred_diacritics)
            total_edit_distance += dist

    if total_ref_diacritics == 0:
        return 0.0

    return total_edit_distance / total_ref_diacritics


def compute_tone_error_rate(predictions: list[str], references: list[str]) -> float:
    """Error rate specifically for tone-marked characters."""
    total_ref = 0
    total_errors = 0

    for pred, ref in zip(predictions, references):
        ref_tones = [c for c in ref if c in TONE_MARK_CHARS]
        pred_tones = [c for c in pred if c in TONE_MARK_CHARS]

        total_ref += len(ref_tones)
        if ref_tones:
            total_errors += editdistance.eval(ref_tones, pred_tones)

    return total_errors / total_ref if total_ref > 0 else 0.0


def compute_modifier_error_rate(predictions: list[str], references: list[str]) -> float:
    """Error rate specifically for vowel modifier characters (ă, â, ê, ô, ơ, ư, đ)."""
    total_ref = 0
    total_errors = 0

    for pred, ref in zip(predictions, references):
        # Strip tone marks to isolate modifier errors
        ref_base = [_strip_tone(c) for c in ref if _strip_tone(c) in VOWEL_MODIFIER_CHARS]
        pred_base = [_strip_tone(c) for c in pred if _strip_tone(c) in VOWEL_MODIFIER_CHARS]

        total_ref += len(ref_base)
        if ref_base:
            total_errors += editdistance.eval(ref_base, pred_base)

    return total_errors / total_ref if total_ref > 0 else 0.0


def _strip_tone(char: str) -> str:
    """Strip tone mark from a Vietnamese character, keeping the base + modifier."""
    # Map: ắ → ă, ấ → â, ế → ê, etc.
    tone_to_base = {
        'ằ': 'ă', 'ắ': 'ă', 'ẳ': 'ă', 'ẵ': 'ă', 'ặ': 'ă',
        'ầ': 'â', 'ấ': 'â', 'ẩ': 'â', 'ẫ': 'â', 'ậ': 'â',
        'ề': 'ê', 'ế': 'ê', 'ể': 'ê', 'ễ': 'ê', 'ệ': 'ê',
        'ồ': 'ô', 'ố': 'ô', 'ổ': 'ô', 'ỗ': 'ô', 'ộ': 'ô',
        'ờ': 'ơ', 'ớ': 'ơ', 'ở': 'ơ', 'ỡ': 'ơ', 'ợ': 'ơ',
        'ừ': 'ư', 'ứ': 'ư', 'ử': 'ư', 'ữ': 'ư', 'ự': 'ư',
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
    }
    return tone_to_base.get(char.lower(), char.lower())


def evaluate(predictions: list[str], references: list[str]) -> HTRMetrics:
    """Full evaluation suite for Vietnamese HTR."""
    assert len(predictions) == len(references), \
        f"Mismatch: {len(predictions)} predictions vs {len(references)} references"

    # Count statistics
    n_chars = sum(len(r) for r in references)
    n_diacritic = sum(1 for r in references for c in r if c in VIETNAMESE_DIACRITICS)
    n_tone = sum(1 for r in references for c in r if c in TONE_MARK_CHARS)

    return HTRMetrics(
        cer=compute_cer(predictions, references),
        wer=compute_wer(predictions, references),
        der=compute_der(predictions, references),
        tone_er=compute_tone_error_rate(predictions, references),
        modifier_er=compute_modifier_error_rate(predictions, references),
        n_samples=len(references),
        n_chars=n_chars,
        n_diacritic_chars=n_diacritic,
        n_tone_chars=n_tone,
    )
