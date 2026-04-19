"""
Text recognition backends: VietOCR (fast) and Gemma 4 E4B (slow/accurate).
Implements dual-path recognition with confidence estimation.
"""

import io
import base64
import numpy as np
from PIL import Image
from dataclasses import dataclass


@dataclass
class RecognitionResult:
    """Result from a recognition attempt."""
    text: str
    confidence: float
    method: str  # "vietocr", "gemma4", "gemma4_think"
    raw_output: str = ""


# ── VietOCR Backend ──────────────────────────────────────────────────


class VietOCRRecognizer:
    """Vietnamese-specific OCR using VGG + Transformer."""

    def __init__(self, config_name: str = "vgg_transformer", device: str = "cuda"):
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg

        config = Cfg.load_config_from_name(config_name)
        config["device"] = device
        config["predictor"]["beamsearch"] = True

        self.predictor = Predictor(config)
        self.device = device

    def recognize(self, image: np.ndarray | Image.Image) -> RecognitionResult:
        """Recognize Vietnamese text from a cropped line image."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        text = self.predictor.predict(image)

        # VietOCR doesn't expose confidence natively
        # Use prediction length as a rough proxy (short outputs = less certain)
        confidence = min(1.0, len(text.strip()) / 5.0) if text.strip() else 0.0

        return RecognitionResult(
            text=text.strip(),
            confidence=confidence,
            method="vietocr"
        )

    def recognize_with_confidence(
        self,
        image: np.ndarray | Image.Image,
        n_samples: int = 3,
        temperature: float = 0.3,
    ) -> RecognitionResult:
        """Self-consistency based confidence: run N times, measure agreement."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Multiple predictions
        predictions = []
        for _ in range(n_samples):
            text = self.predictor.predict(image)
            predictions.append(text.strip())

        # Use majority vote
        from collections import Counter
        counts = Counter(predictions)
        best_text, best_count = counts.most_common(1)[0]

        # Confidence = agreement ratio
        confidence = best_count / n_samples

        return RecognitionResult(
            text=best_text,
            confidence=confidence,
            method="vietocr"
        )


# ── Gemma 4 E4B Backend ─────────────────────────────────────────────


class Gemma4Recognizer:
    """Gemma 4 E4B vision-language model for Vietnamese HTR."""

    SYSTEM_PROMPT_RECOGNIZER = (
        "You are a Vietnamese handwriting OCR system. "
        "Read the handwritten Vietnamese text in the image exactly as written. "
        "Preserve ALL diacritical marks (dấu) precisely — every accent, "
        "circumflex, breve, horn, and tone mark matters. "
        "Output ONLY the transcribed text, nothing else."
    )

    SYSTEM_PROMPT_PROPOSER = (
        "You are an expert Vietnamese handwriting reader. "
        "The initial OCR read this text as: '{initial_text}'. "
        "Look at the image again very carefully, paying special attention to: "
        "- Tone marks (sắc, huyền, hỏi, ngã, nặng) "
        "- Vowel modifiers (circumflex â/ê/ô, breve ă, horn ơ/ư) "
        "- Easily confused pairs: ă/â, ơ/ô, é/è/ê "
        "Propose corrections if any diacritics were misread. "
        "Output ONLY the corrected text."
    )

    SYSTEM_PROMPT_LINGUIST = (
        "You are a Vietnamese language expert. You do NOT see the image. "
        "Check if this Vietnamese text is linguistically valid: '{text}' "
        "For each word, verify: "
        "1. Is it a valid Vietnamese syllable? "
        "2. Are the diacritics consistent (correct tone mark placement)? "
        "3. Does it make sense in context with surrounding words? "
        "If you find errors, suggest corrections. "
        "Output format: VALID if text is correct, or CORRECTED: <your correction>"
    )

    SYSTEM_PROMPT_JUDGE = (
        "You are the final judge for Vietnamese handwriting OCR. "
        "Initial reading: '{initial_text}' "
        "Proposer's correction: '{proposed_text}' "
        "Linguist's assessment: '{linguist_output}' "
        "Look at the image one final time. Consider both the visual evidence "
        "and the linguistic assessment. Output ONLY the final correct text."
    )

    def __init__(
        self,
        model_name: str = "google/gemma-4-E4B-it",
        quantization: str = "4bit",
        visual_token_budget: int = 1120,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.visual_token_budget = visual_token_budget
        self.device = device
        self.model = None
        self.processor = None
        self._quantization = quantization

    def load(self):
        """Lazy load the model (heavy, only when needed)."""
        if self.model is not None:
            return

        from transformers import AutoProcessor, AutoModelForImageTextToText
        import torch

        print(f"Loading {self.model_name} ({self._quantization})...")

        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }

        if self._quantization == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        elif self._quantization == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, **load_kwargs
        )
        print(f"Model loaded successfully.")

    def recognize(
        self,
        image: np.ndarray | Image.Image,
        thinking: bool = False,
        max_new_tokens: int = 512,
    ) -> RecognitionResult:
        """Zero-shot Vietnamese handwriting recognition."""
        self.load()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        messages = self._build_messages(
            system_prompt=self.SYSTEM_PROMPT_RECOGNIZER,
            user_text="Read the Vietnamese handwritten text in this image.",
            image=image,
            thinking=thinking,
        )

        text = self._generate(messages, max_new_tokens)

        return RecognitionResult(
            text=text.strip(),
            confidence=0.7,  # default for zero-shot
            method="gemma4_think" if thinking else "gemma4",
            raw_output=text,
        )

    def propose_correction(
        self,
        image: np.ndarray | Image.Image,
        initial_text: str,
        max_new_tokens: int = 512,
    ) -> RecognitionResult:
        """Proposer agent: re-examine image given initial OCR output."""
        self.load()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = self.SYSTEM_PROMPT_PROPOSER.format(initial_text=initial_text)
        messages = self._build_messages(
            system_prompt=prompt,
            user_text="Look at this handwriting image and correct any diacritic errors.",
            image=image,
            thinking=True,
        )

        text = self._generate(messages, max_new_tokens)

        return RecognitionResult(
            text=text.strip(),
            confidence=0.8,
            method="gemma4_proposer",
            raw_output=text,
        )

    def linguistic_check(self, text: str, max_new_tokens: int = 256) -> RecognitionResult:
        """Linguist agent: text-only Vietnamese validity check."""
        self.load()

        prompt = self.SYSTEM_PROMPT_LINGUIST.format(text=text)
        messages = self._build_messages(
            system_prompt=prompt,
            user_text=f"Check this Vietnamese text: {text}",
            image=None,
            thinking=True,
        )

        output = self._generate(messages, max_new_tokens)

        # Parse output
        if output.strip().upper().startswith("VALID"):
            return RecognitionResult(text=text, confidence=0.9, method="gemma4_linguist", raw_output=output)
        elif "CORRECTED:" in output.upper():
            corrected = output.split(":", 1)[-1].strip()
            return RecognitionResult(text=corrected, confidence=0.85, method="gemma4_linguist", raw_output=output)
        else:
            return RecognitionResult(text=output.strip(), confidence=0.75, method="gemma4_linguist", raw_output=output)

    def judge(
        self,
        image: np.ndarray | Image.Image,
        initial_text: str,
        proposed_text: str,
        linguist_output: str,
        max_new_tokens: int = 512,
    ) -> RecognitionResult:
        """Judge agent: final decision using image + all evidence."""
        self.load()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = self.SYSTEM_PROMPT_JUDGE.format(
            initial_text=initial_text,
            proposed_text=proposed_text,
            linguist_output=linguist_output,
        )

        messages = self._build_messages(
            system_prompt=prompt,
            user_text="Make the final decision on the correct text.",
            image=image,
            thinking=True,
        )

        text = self._generate(messages, max_new_tokens)

        return RecognitionResult(
            text=text.strip(),
            confidence=0.9,
            method="gemma4_judge",
            raw_output=text,
        )

    # ── NEW BASELINES (Reviewer Weakness #1) ─────────────────────────
    # These prove the multi-agent triad adds value beyond simpler alternatives.

    def recognize_comprehensive(
        self,
        image: np.ndarray | Image.Image,
        max_new_tokens: int = 512,
    ) -> RecognitionResult:
        """
        BASELINE: Single comprehensive prompt.
        Tests whether one good prompt matches the 3-agent triad.
        """
        self.load()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = (
            "You are an expert Vietnamese handwriting OCR system. "
            "Read the handwritten text in this image. Then:\n"
            "1. Verify every diacritic mark (sắc, huyền, hỏi, ngã, nặng)\n"
            "2. Check vowel modifiers (â/ă, ê, ô/ơ, ư)\n"
            "3. Ensure each word is a valid Vietnamese syllable\n"
            "4. Consider context to disambiguate similar diacritics\n"
            "Output ONLY the final corrected Vietnamese text."
        )

        messages = self._build_messages(
            system_prompt=prompt,
            user_text="Read and verify this Vietnamese handwriting.",
            image=image,
            thinking=True,
        )

        text = self._generate(messages, max_new_tokens)

        return RecognitionResult(
            text=text.strip(),
            confidence=0.8,
            method="gemma4_comprehensive",
            raw_output=text,
        )

    def recognize_cot(
        self,
        image: np.ndarray | Image.Image,
        max_new_tokens: int = 1024,
    ) -> RecognitionResult:
        """
        BASELINE: Chain-of-thought prompting.
        Tests whether structured reasoning matches multi-agent deliberation.
        """
        self.load()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = (
            "You are a Vietnamese handwriting OCR expert. "
            "Think step by step:\n"
            "Step 1: Read the raw text from the image\n"
            "Step 2: For each word, identify the base letter and any diacritics\n"
            "Step 3: Check if each syllable is valid Vietnamese\n"
            "Step 4: If any syllable is invalid, look at the image again and correct\n"
            "Step 5: Output the final text\n\n"
            "Format your response as:\n"
            "RAW: <initial reading>\n"
            "CHECKED: <list any corrections>\n"
            "FINAL: <corrected text>"
        )

        messages = self._build_messages(
            system_prompt=prompt,
            user_text="Read this Vietnamese handwriting step by step.",
            image=image,
            thinking=True,
        )

        text = self._generate(messages, max_new_tokens)

        # Parse FINAL: line if present
        final_text = text
        if "FINAL:" in text.upper():
            for line in text.split("\n"):
                if line.strip().upper().startswith("FINAL:"):
                    final_text = line.split(":", 1)[-1].strip()
                    break

        return RecognitionResult(
            text=final_text.strip(),
            confidence=0.8,
            method="gemma4_cot",
            raw_output=text,
        )

    def recognize_self_consistency(
        self,
        image: np.ndarray | Image.Image,
        n_samples: int = 5,
        temperature: float = 0.3,
        max_new_tokens: int = 512,
    ) -> RecognitionResult:
        """
        BASELINE: Self-consistency (majority vote over N runs).
        Tests whether voting matches the Proposer-Linguist-Judge triad.
        """
        self.load()
        import torch

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        predictions = []
        for _ in range(n_samples):
            messages = self._build_messages(
                system_prompt=self.SYSTEM_PROMPT_RECOGNIZER,
                user_text="Read the Vietnamese handwritten text in this image.",
                image=image,
                thinking=False,
            )

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                )

            new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
            text = self.processor.decode(new_tokens, skip_special_tokens=True)
            if "<|channel|>" in text:
                text = text.split("<|channel|>")[-1]
            predictions.append(text.strip())

        # Majority vote
        from collections import Counter
        counts = Counter(predictions)
        best_text, best_count = counts.most_common(1)[0]
        confidence = best_count / n_samples

        return RecognitionResult(
            text=best_text,
            confidence=confidence,
            method=f"gemma4_sc{n_samples}",
            raw_output=str(predictions),
        )

    # ── Tool-augmented linguistic check ──────────────────────────────

    SYSTEM_PROMPT_LINGUIST_TOOLS = (
        "You are a Vietnamese language expert. You do NOT see the image.\n"
        "A Vietnamese syllable validator tool has analyzed the text and found issues.\n\n"
        "Tool findings:\n{tool_report}\n\n"
        "Original context: '{context}'\n\n"
        "Using the tool findings AND your Vietnamese language knowledge, "
        "correct the text. Focus on:\n"
        "- Fixing invalid syllables using the tool's suggestions\n"
        "- Resolving tone mark errors\n"
        "- Ensuring contextual coherence\n"
        "Output ONLY the corrected text, nothing else."
    )

    def linguistic_check_with_tools(
        self,
        text: str,
        tool_report: str,
        context: str = "",
        max_new_tokens: int = 256,
    ) -> RecognitionResult:
        """Linguist agent: LLM informed by tool findings."""
        self.load()

        prompt = self.SYSTEM_PROMPT_LINGUIST_TOOLS.format(
            tool_report=tool_report,
            context=context,
        )
        messages = self._build_messages(
            system_prompt=prompt,
            user_text=f"Correct this text using the tool findings: {text}",
            image=None,
            thinking=True,
        )

        output = self._generate(messages, max_new_tokens)

        return RecognitionResult(
            text=output.strip(),
            confidence=0.85,
            method="gemma4_linguist_tools",
            raw_output=output,
        )

    def _build_messages(
        self,
        system_prompt: str,
        user_text: str,
        image: Image.Image | None = None,
        thinking: bool = False,
    ) -> list[dict]:
        """Build chat messages for Gemma 4."""
        # System message with optional thinking control
        system_content = system_prompt
        if thinking:
            system_content = "<|think|>\n" + system_content

        messages = [{"role": "system", "content": [{"type": "text", "text": system_content}]}]

        # User message with optional image
        user_content = []
        if image is not None:
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": user_text})

        messages.append({"role": "user", "content": user_content})
        return messages

    def _generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        """Generate text from messages.

        Uses two-step processor call (apply_chat_template then processor(text, images=...))
        so that image content in messages actually reaches the model — the single-call
        tokenize=True path drops images silently on Gemma 4.
        """
        import torch

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        images = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "image" and "image" in c:
                        images.append(c["image"])

        proc_kwargs = {"text": [text], "return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        inputs = self.processor(**proc_kwargs).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode only new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True)

        # Strip thinking tags if present
        if "<|channel|>" in text:
            parts = text.split("<|channel|>")
            text = parts[-1] if len(parts) > 1 else text

        return text.strip()
