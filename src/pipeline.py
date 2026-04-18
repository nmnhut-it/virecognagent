"""
ViRecognAgent Pipeline: confidence-gated multi-agent Vietnamese HTR.

Outputs STRUCTURED JSON per document — each region includes bounding box,
recognized text, confidence, agent trace, and diacritic analysis.

Pipeline flow:
  Image → Preprocess → Detect (PaddleOCR DB) → Recognize (VietOCR/Gemma4)
       → Confidence Gate → [Agent Pod if LOW] → Post-correct → Structured output
"""

import time
import json
import numpy as np
from PIL import Image
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

from .preprocessing import preprocess_image, crop_text_region
from .detection import TextDetector, TextRegion
from .recognition import VietOCRRecognizer, Gemma4Recognizer, RecognitionResult
from .agents import AgentPod, AgentPodResult


# Vietnamese diacritic character set for per-region analysis
_VIET_DIACRITICS = set(
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợ"
    "ùúủũụưứừửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
    "ÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"
)


@dataclass
class DiacriticInfo:
    """Diacritic analysis for a single text region."""
    count: int = 0
    chars: list[str] = field(default_factory=list)
    corrections_made: int = 0
    der_risk: str = "low"  # low / medium / high

    def to_dict(self) -> dict:
        return {"count": self.count, "chars": self.chars,
                "corrections_made": self.corrections_made, "der_risk": self.der_risk}


@dataclass
class AgentTrace:
    """Full trace of agent pod activity for a single region."""
    initial_text: str = ""
    proposer_text: str = ""
    linguist_verdict: str = ""
    judge_text: str = ""
    rounds: int = 0
    safety_rejected: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RegionOutput:
    """Structured output for a single detected text region."""
    id: int
    bbox: list
    text: str
    confidence: float
    method: str
    agent_activated: bool = False
    agent_trace: Optional[AgentTrace] = None
    diacritics: Optional[DiacriticInfo] = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "bbox": self.bbox,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "agent_activated": self.agent_activated,
        }
        if self.agent_trace and self.agent_activated:
            d["agent_trace"] = self.agent_trace.to_dict()
        if self.diacritics:
            d["diacritics"] = self.diacritics.to_dict()
        return d


@dataclass
class StructuredOutput:
    """Complete structured output for one document image."""
    document: str
    regions: list[RegionOutput] = field(default_factory=list)
    full_text: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "document": self.document,
            "regions": [r.to_dict() for r in self.regions],
            "full_text": self.full_text,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


# Keep backward-compatible alias
PipelineResult = StructuredOutput


class ViRecognAgentPipeline:
    """
    Main pipeline combining deterministic OCR with multi-agent verification.

    Usage:
        pipeline = ViRecognAgentPipeline(mode="full")
        result = pipeline.process("handwriting.png")
        print(result.final_text)
    """

    MODES = {
        "gemma4_zeroshot":      "Gemma 4 E4B zero-shot only",
        "gemma4_think":         "Gemma 4 E4B with thinking ON",
        "gemma4_comprehensive": "Gemma 4 single comprehensive prompt (baseline)",
        "gemma4_cot":           "Gemma 4 chain-of-thought (baseline)",
        "gemma4_sc":            "Gemma 4 self-consistency N=5 (baseline)",
        "vietocr":              "VietOCR only",
        "vietocr_ensemble":     "VietOCR ensemble voting N=5 (baseline)",
        "two_stage":            "VietOCR + Gemma 4 post-correction",
        "full":                 "VietOCR + confidence gate + Tool-augmented Agent Pod",
        "ablation_no_linguist": "Full pipeline minus Linguist agent",
        "ablation_no_judge":    "Full pipeline minus Judge agent",
        "ablation_tools_only":  "Full pipeline, Linguist uses tools ONLY (no LLM fallback)",
    }

    def __init__(
        self,
        mode: str = "full",
        confidence_threshold: float = 0.85,
        max_rounds: int = 2,
        gemma_model: str = "google/gemma-4-E4B-it",
        gemma_quantization: str = "4bit",
        visual_token_budget: int = 1120,
        device: str = "cuda",
    ):
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.device = device

        print(f"Initializing ViRecognAgent pipeline: {self.MODES.get(mode, mode)}")

        # Always init detector
        self.detector = TextDetector()

        # Init recognizers based on mode
        self.vietocr = None
        self.gemma = None
        self.agent_pod = None

        if mode in ("vietocr", "vietocr_ensemble", "two_stage", "full",
                   "ablation_no_linguist", "ablation_no_judge", "ablation_tools_only"):
            print("  Loading VietOCR...")
            self.vietocr = VietOCRRecognizer(device=device)

        if mode not in ("vietocr", "vietocr_ensemble"):
            print("  Initializing Gemma 4 E4B (lazy load on first use)...")
            self.gemma = Gemma4Recognizer(
                model_name=gemma_model,
                quantization=gemma_quantization,
                visual_token_budget=visual_token_budget,
                device=device,
            )

        if mode in ("full", "ablation_no_linguist", "ablation_no_judge", "ablation_tools_only"):
            linguist_use_llm = (mode != "ablation_tools_only")
            self.agent_pod = AgentPod(
                gemma=self.gemma,
                max_rounds=max_rounds,
                linguist_use_llm=linguist_use_llm,
            )

        print(f"  Pipeline ready. Confidence threshold: {confidence_threshold}")

    def process(self, image_path: str) -> StructuredOutput:
        """Process a single document image → structured JSON output."""
        start_time = time.time()

        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))

        # Step 1: Preprocess
        preprocessed = preprocess_image(image)

        # Step 2: Detect text regions
        detected = self.detector.detect(image)

        output = StructuredOutput(document=image_path)

        if not detected:
            output.metadata = self._build_metadata([], 0, time.time() - start_time)
            return output

        # Step 3: Process each region → structured RegionOutput
        n_agent_activations = 0

        for i, region in enumerate(detected):
            crop = region.crop
            if crop is None:
                continue

            region_out = self._process_line(
                crop, region_idx=i, bbox=region.bbox
            )
            output.regions.append(region_out)

            if region_out.agent_activated:
                n_agent_activations += 1

        # Assemble full text and metadata
        output.full_text = "\n".join(r.text for r in output.regions)
        output.metadata = self._build_metadata(
            output.regions, n_agent_activations, time.time() - start_time
        )

        return output

    def process_single_line(
        self, image: np.ndarray | Image.Image, region_idx: int = 0
    ) -> RegionOutput:
        """Process a single line image → structured RegionOutput."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        h, w = image.shape[:2]
        bbox = [[0, 0], [w, 0], [w, h], [0, h]]
        return self._process_line(image, region_idx=region_idx, bbox=bbox)

    def _process_line(
        self, crop: np.ndarray, region_idx: int = 0, bbox: list = None
    ) -> RegionOutput:
        """Process a single text line → structured RegionOutput with full trace."""
        bbox = bbox or []
        text = ""
        confidence = 0.0
        method = self.mode
        agent_activated = False
        agent_trace = None
        initial_text = ""

        # ── Mode: Gemma 4 zero-shot ──
        if self.mode == "gemma4_zeroshot":
            result = self.gemma.recognize(crop, thinking=False)
            text, confidence, method = result.text, result.confidence, "gemma4_zeroshot"

        # ── Mode: Gemma 4 with thinking ──
        elif self.mode == "gemma4_think":
            result = self.gemma.recognize(crop, thinking=True)
            text, confidence, method = result.text, result.confidence, "gemma4_think"

        # ── BASELINE: Single comprehensive prompt ──
        elif self.mode == "gemma4_comprehensive":
            result = self.gemma.recognize_comprehensive(crop)
            text, confidence, method = result.text, result.confidence, "gemma4_comprehensive"

        # ── BASELINE: Chain-of-thought ──
        elif self.mode == "gemma4_cot":
            result = self.gemma.recognize_cot(crop)
            text, confidence, method = result.text, result.confidence, "gemma4_cot"

        # ── BASELINE: Self-consistency voting ──
        elif self.mode == "gemma4_sc":
            result = self.gemma.recognize_self_consistency(crop, n_samples=5)
            text, confidence, method = result.text, result.confidence, f"gemma4_sc5"

        # ── Mode: VietOCR only ──
        elif self.mode == "vietocr":
            result = self.vietocr.recognize(crop)
            text, confidence, method = result.text, result.confidence, "vietocr"

        # ── BASELINE: VietOCR ensemble voting ──
        elif self.mode == "vietocr_ensemble":
            result = self.vietocr.recognize_with_confidence(crop, n_samples=5)
            text, confidence, method = result.text, result.confidence, "vietocr_ensemble5"

        # ── Mode: Two-stage (VietOCR + Gemma post-correction) ──
        elif self.mode == "two_stage":
            vietocr_result = self.vietocr.recognize_with_confidence(crop, n_samples=3)
            corrected = self.gemma.linguistic_check(vietocr_result.text)
            text = corrected.text
            confidence = vietocr_result.confidence
            method = "two_stage"
            initial_text = vietocr_result.text

        # ── Full pipeline with confidence gating ──
        else:
            vietocr_result = self.vietocr.recognize_with_confidence(crop, n_samples=3)
            confidence = vietocr_result.confidence
            initial_text = vietocr_result.text

            if vietocr_result.confidence >= self.confidence_threshold:
                # HIGH confidence → accept with lightweight post-correction
                text = self._lightweight_postcorrect(vietocr_result.text)
                method = "vietocr_high_conf"

            else:
                # LOW confidence → Agent Pod
                agent_activated = True

                if self.agent_pod is not None:
                    pod_result = self.agent_pod.verify(crop, vietocr_result.text)
                    text = pod_result.final_text
                    method = f"agent_pod_r{pod_result.rounds_used}"

                    # Build agent trace from pod history
                    agent_trace = AgentTrace(
                        initial_text=vietocr_result.text,
                        rounds=pod_result.rounds_used,
                        safety_rejected=pod_result.rejected_by_safety,
                    )
                    # Extract proposer/linguist/judge outputs from history
                    for entry in pod_result.history:
                        if entry.get("agent") == "proposer":
                            agent_trace.proposer_text = entry.get("output", "")
                        elif entry.get("agent") == "linguist":
                            agent_trace.linguist_verdict = entry.get("raw", entry.get("output", ""))
                        elif entry.get("agent") == "judge":
                            agent_trace.judge_text = entry.get("output", "")

                else:
                    result = self.gemma.recognize(crop, thinking=True)
                    text = result.text
                    method = "gemma4_fallback"

        # Build diacritic analysis
        diacritics = self._analyze_diacritics(text, initial_text)

        return RegionOutput(
            id=region_idx,
            bbox=bbox,
            text=text,
            confidence=confidence,
            method=method,
            agent_activated=agent_activated,
            agent_trace=agent_trace,
            diacritics=diacritics,
        )

    def _analyze_diacritics(self, text: str, initial_text: str = "") -> DiacriticInfo:
        """Analyze diacritic content of recognized text."""
        diac_chars = [c for c in text if c in _VIET_DIACRITICS]
        count = len(diac_chars)
        unique_chars = sorted(set(diac_chars))

        # Count corrections (chars in final text not in initial)
        corrections = 0
        if initial_text and initial_text != text:
            initial_diac = set(c for c in initial_text if c in _VIET_DIACRITICS)
            final_diac = set(c for c in text if c in _VIET_DIACRITICS)
            corrections = len(final_diac - initial_diac)

        # Risk assessment based on diacritic density
        text_len = max(len(text), 1)
        diac_ratio = count / text_len
        if diac_ratio > 0.3:
            risk = "high"
        elif diac_ratio > 0.15:
            risk = "medium"
        else:
            risk = "low"

        return DiacriticInfo(
            count=count,
            chars=unique_chars,
            corrections_made=corrections,
            der_risk=risk,
        )

    def _build_metadata(
        self, regions: list[RegionOutput], n_activations: int, elapsed: float
    ) -> dict:
        """Build document-level metadata."""
        n_regions = len(regions)
        total_diacritics = sum(r.diacritics.count for r in regions if r.diacritics)
        avg_conf = (
            sum(r.confidence for r in regions) / n_regions if n_regions else 0.0
        )

        return {
            "total_regions": n_regions,
            "agent_activation_rate": round(n_activations / n_regions, 4) if n_regions else 0,
            "agent_activations": n_activations,
            "avg_confidence": round(avg_conf, 4),
            "total_diacritic_chars": total_diacritics,
            "processing_time_seconds": round(elapsed, 3),
            "pipeline_mode": self.mode,
            "confidence_threshold": self.confidence_threshold,
        }

    def _lightweight_postcorrect(self, text: str) -> str:
        """Quick rule-based post-correction for high-confidence outputs."""
        # Basic Vietnamese-specific corrections
        # (expand this with more rules as you discover common errors)
        corrections = {
            "  ": " ",     # double space
            " ,": ",",     # space before comma
            " .": ".",     # space before period
        }
        for old, new in corrections.items():
            text = text.replace(old, new)
        return text.strip()
