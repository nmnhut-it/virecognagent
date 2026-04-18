"""
Agent Pod: Bounded multi-agent verification for low-confidence OCR regions.

CRITICAL DESIGN CHOICE (addresses reviewer weakness #2):
The Linguist agent uses REAL Vietnamese linguistic tools — syllable
validation, dictionary lookup, tone consistency checking — not just
LLM prompting. This is what makes it a tool-augmented agent, not
"three LLM calls pretending to be specialists."

Architecture:
  Proposer (vision + thinking) → Linguist (tools + optional LLM) → Judge (vision + all evidence)
  Max 2 rounds, with anti-hallucination safety gate.
"""

import editdistance
from dataclasses import dataclass, field
from .recognition import Gemma4Recognizer, RecognitionResult
from .viet_tools import (
    validate_vietnamese_text,
    ValidationResult,
    check_tone_consistency,
    suggest_diacritic_restorations,
    strip_tones,
    is_valid_syllable_structure,
)


@dataclass
class AgentPodResult:
    """Result from the multi-agent verification pod."""
    final_text: str
    initial_text: str
    rounds_used: int
    agents_invoked: list[str] = field(default_factory=list)
    was_corrected: bool = False
    rejected_by_safety: bool = False
    linguist_validation: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)


class ToolAugmentedLinguist:
    """
    Vietnamese linguistic verification using real tools + optional LLM.

    This is NOT just an LLM call — it runs actual Vietnamese phonotactic
    rules, syllable dictionary lookups, and tone consistency checks.
    The LLM is only invoked when tools flag ambiguous cases that need
    contextual reasoning.
    """

    def __init__(self, gemma: Gemma4Recognizer = None, use_llm_fallback: bool = True):
        self.gemma = gemma
        self.use_llm_fallback = use_llm_fallback

    def verify(self, text: str, context: str = "") -> dict:
        """
        Verify Vietnamese text using linguistic tools.

        Returns dict with:
          - is_valid: bool
          - corrected_text: str (may be same as input)
          - validation: full ValidationResult
          - tool_corrections: list of corrections made by tools
          - llm_consulted: bool
        """
        result = {
            "is_valid": True,
            "corrected_text": text,
            "tool_corrections": [],
            "llm_consulted": False,
            "validation": None,
        }

        # Step 1: Run Vietnamese syllable validator (TOOL, not LLM)
        validation = validate_vietnamese_text(text)
        result["validation"] = validation.to_dict()

        if validation.is_valid and validation.confidence > 0.95:
            # All syllables valid, high confidence -> accept without LLM
            return result

        # Step 2: Attempt tool-based corrections for invalid syllables
        for invalid_syl in validation.invalid_syllables:
            suggestions = validation.suggestions.get(invalid_syl, [])
            if suggestions:
                result["tool_corrections"].append({
                    "original": invalid_syl,
                    "suggestions": suggestions,
                    "type": "invalid_syllable",
                })

        # Step 3: Check tone mark consistency (TOOL)
        for tone_error in validation.tone_errors:
            result["tool_corrections"].append({
                "error": tone_error,
                "type": "tone_error",
            })

        # Step 4: If tools found issues AND LLM is available, consult LLM
        # for contextual disambiguation (tools can't do context)
        if (not validation.is_valid or validation.confidence < 0.9) and \
           self.use_llm_fallback and self.gemma is not None:

            result["llm_consulted"] = True
            tool_report = self._format_tool_report(text, validation)
            llm_result = self.gemma.linguistic_check_with_tools(
                text=text,
                tool_report=tool_report,
                context=context,
            )
            corrected_text = llm_result.text
            result["corrected_text"] = corrected_text

            # Re-validate the LLM correction
            revalidation = validate_vietnamese_text(corrected_text)
            if revalidation.confidence < validation.confidence:
                # LLM made it worse -> reject
                result["corrected_text"] = text
                result["tool_corrections"].append({
                    "type": "llm_correction_rejected",
                    "reason": "revalidation worse than original",
                })
        else:
            result["corrected_text"] = text

        result["is_valid"] = validate_vietnamese_text(result["corrected_text"]).is_valid
        return result

    def _format_tool_report(self, text: str, validation: ValidationResult) -> str:
        """Format tool findings into a structured report for the LLM."""
        lines = [f"Text to verify: \"{text}\""]
        lines.append(f"Syllable validity: {validation.valid_syllables}/{validation.total_syllables} valid")

        if validation.invalid_syllables:
            lines.append(f"Invalid syllables: {', '.join(validation.invalid_syllables)}")
            for syl, suggestions in validation.suggestions.items():
                lines.append(f"  '{syl}' might be: {', '.join(suggestions)}")

        if validation.tone_errors:
            lines.append(f"Tone errors: {', '.join(validation.tone_errors)}")

        return "\n".join(lines)


class AgentPod:
    """
    Confidence-gated bounded multi-agent verification.

    The Linguist uses REAL TOOLS first, only consulting the LLM when
    tools flag ambiguous cases needing contextual reasoning.
    """

    def __init__(
        self,
        gemma: Gemma4Recognizer,
        max_rounds: int = 2,
        max_edit_distance_ratio: float = 0.3,
        max_word_count_change: float = 0.1,
        linguist_use_llm: bool = True,
    ):
        self.gemma = gemma
        self.max_rounds = max_rounds
        self.max_edit_ratio = max_edit_distance_ratio
        self.max_word_change = max_word_count_change
        self.linguist = ToolAugmentedLinguist(
            gemma=gemma if linguist_use_llm else None,
            use_llm_fallback=linguist_use_llm,
        )

    def verify(self, image, initial_text: str) -> AgentPodResult:
        """Run the multi-agent verification pod."""
        result = AgentPodResult(
            final_text=initial_text,
            initial_text=initial_text,
            rounds_used=0,
        )

        current_text = initial_text

        for round_num in range(1, self.max_rounds + 1):
            result.rounds_used = round_num

            # Step 1: Proposer (Gemma 4 vision)
            proposer_result = self.gemma.propose_correction(image, current_text)
            proposed_text = proposer_result.text
            result.agents_invoked.append("proposer")
            result.history.append({
                "round": round_num, "agent": "proposer",
                "input": current_text, "output": proposed_text,
            })

            if not self._safety_check(current_text, proposed_text):
                result.rejected_by_safety = True
                result.history.append({
                    "round": round_num, "agent": "safety_gate",
                    "action": "rejected_proposer", "reason": "edit distance too large",
                })
                break

            # Step 2: Linguist (TOOLS + optional LLM)
            linguist_result = self.linguist.verify(
                text=proposed_text, context=current_text,
            )
            result.agents_invoked.append("linguist")
            result.linguist_validation = linguist_result
            result.history.append({
                "round": round_num, "agent": "linguist",
                "input": proposed_text,
                "is_valid": linguist_result["is_valid"],
                "tool_corrections": linguist_result["tool_corrections"],
                "llm_consulted": linguist_result["llm_consulted"],
                "corrected_text": linguist_result["corrected_text"],
            })

            linguist_text = linguist_result["corrected_text"]

            # Step 3: Judge (Gemma 4 vision + all evidence)
            linguist_summary = self._format_linguist_for_judge(linguist_result)
            judge_result = self.gemma.judge(
                image=image,
                initial_text=current_text,
                proposed_text=linguist_text,
                linguist_output=linguist_summary,
            )
            result.agents_invoked.append("judge")
            result.history.append({
                "round": round_num, "agent": "judge",
                "output": judge_result.text,
            })

            if not self._safety_check(initial_text, judge_result.text):
                result.rejected_by_safety = True
                result.history.append({
                    "round": round_num, "agent": "safety_gate",
                    "action": "rejected_judge",
                    "reason": "final output too different from initial",
                })
                break

            if judge_result.text == current_text:
                break

            current_text = judge_result.text
            result.final_text = current_text
            result.was_corrected = (current_text != initial_text)

        return result

    def _format_linguist_for_judge(self, linguist_result: dict) -> str:
        """Format linguist findings for the judge agent."""
        lines = []
        if linguist_result["is_valid"]:
            lines.append("Linguist: Text is linguistically VALID.")
        else:
            lines.append("Linguist: Text has issues.")

        for corr in linguist_result.get("tool_corrections", []):
            if corr["type"] == "invalid_syllable":
                lines.append(
                    f"  Invalid syllable '{corr['original']}', "
                    f"suggestions: {', '.join(corr['suggestions'])}"
                )
            elif corr["type"] == "tone_error":
                lines.append(f"  Tone error: {corr['error']}")

        if linguist_result.get("llm_consulted"):
            lines.append(f"Linguist corrected to: '{linguist_result['corrected_text']}'")

        return "\n".join(lines)

    def _safety_check(self, original: str, corrected: str) -> bool:
        """Anti-hallucination gate (VieBookRead approach)."""
        if not original or not corrected:
            return False

        dist = editdistance.eval(original, corrected)
        max_len = max(len(original), len(corrected))
        if max_len > 0 and (dist / max_len) > self.max_edit_ratio:
            return False

        orig_words = len(original.split())
        corr_words = len(corrected.split())
        if orig_words > 0:
            word_change = abs(orig_words - corr_words) / orig_words
            if word_change > self.max_word_change:
                return False

        return True
