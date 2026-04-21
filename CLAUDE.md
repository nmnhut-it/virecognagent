# CLAUDE.md — ViRecognAgent

## Project Overview

**ViRecognAgent** is a confidence-gated multi-agent pipeline for Vietnamese handwritten text recognition (HTR). It uses a deterministic OCR backbone (PaddleOCR detection + VietOCR recognition) with multi-agent LLM verification loops (Gemma 4 E4B) that activate only on low-confidence regions.

This is a research project for an academic paper targeting ICDAR/ICFHR venues.

## Architecture

```
Image → Preprocess (OpenCV) → Detect (PaddleOCR DB)
     → Recognize fast (VietOCR)
     → Confidence Gate
          ├── HIGH (≥0.85) → lightweight post-correct → output
          └── LOW  (<0.85) → Agent Pod:
                    Proposer (Gemma 4 vision, thinking ON)
                    → Linguist (Vietnamese tools + optional LLM)
                    → Judge (Gemma 4 vision + all evidence)
                    → [max 2 rounds] → output
     → Structured JSON output per region
```

## Key Design Decisions

1. **Tool-augmented Linguist**: The Linguist agent uses REAL Vietnamese linguistic tools (syllable validator, tone checker, dictionary lookup in `src/viet_tools.py`), NOT just LLM prompting. The LLM is only consulted when tools flag ambiguous cases.

2. **Single model family**: All LLM agents use Gemma 4 E4B, differentiated by system prompt, thinking mode, and vision toggle.

3. **Structured output**: Every region outputs JSON with bbox, text, confidence, agent trace, and diacritic analysis.

4. **DER metric**: Novel Diacritic Error Rate metric in `src/metrics.py` — CER computed only over diacritic-bearing characters.

## Project Structure

```
src/
  __init__.py          # Package init
  preprocessing.py     # OpenCV: deskew, CLAHE, binarize (192px height)
  detection.py         # PaddleOCR DB wrapper (tuned for handwriting)
  recognition.py       # VietOCR + Gemma 4 E4B backends + baselines
  viet_tools.py        # Vietnamese syllable validator, tone checker, dictionary
  agents.py            # Tool-augmented Agent Pod (Proposer/Linguist/Judge)
  pipeline.py          # Main orchestrator with confidence gating + structured output
  metrics.py           # CER, WER, DER, Tone ER, Modifier ER
  calibration.py       # Confidence gate calibration (ECE, MCE, reliability diagrams)

scripts/
  download_datasets.py # Downloads 5CD-AI/Viet-Handwriting-OCR from HuggingFace
  run_experiments.py   # Runs all baselines + ablations + sweeps

notebooks/
  PaddleOCR_VietWiki.ipynb      # Simple baseline: stock PaddleOCR on Viet-Wiki-Handwriting
  ViRecognAgent_Pipeline.ipynb  # Main Colab notebook (full multi-agent pipeline)

configs/
  default.yaml         # All configuration parameters
```

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets
python scripts/download_datasets.py --dataset viet_handwriting

# Run a single experiment
python scripts/run_experiments.py --mode vietocr --max-samples 100

# Run all experiments
python scripts/run_experiments.py --mode all --max-samples 100

# Available modes:
#   vietocr, vietocr_ensemble, gemma4_zeroshot, gemma4_think,
#   gemma4_comprehensive, gemma4_cot, gemma4_sc, two_stage,
#   full, ablation_no_linguist, ablation_no_judge, ablation_tools_only
```

## Important Notes

- **Gemma 4 E4B** (`google/gemma-4-E4B-it`) is very new (April 2026). Requires `transformers>=4.50.0`. Use 4-bit quantization to fit on T4 GPU.
- **PaddleOCR** needs `paddlepaddle` installed separately. On GPU: `pip install paddlepaddle-gpu`.
- **VietOCR** (`pip install vietocr`) — downloads pretrained weights automatically on first use.
- The visual token budget for Gemma 4 should be set to **1120** for handwriting (maximum detail).
- Vietnamese diacritics are the core challenge: 67 diacritical characters, 18 variants of "a" alone.
- Output is always structured JSON — see `StructuredOutput` and `RegionOutput` in `pipeline.py`.

## Testing

When making changes, verify:
1. `src/viet_tools.py` — run `validate_vietnamese_text("Xin chào Việt Nam")` should return valid
2. `src/metrics.py` — run `evaluate(["Xin chào"], ["Xin chao"])` should show DER > 0
3. `src/pipeline.py` — all 12 modes should initialize without errors
4. Structured output — `RegionOutput.to_dict()` and `StructuredOutput.to_json()` should produce valid JSON

## Code Style

- Python 3.10+, type hints everywhere
- Dataclasses for structured data
- All Vietnamese text handled as UTF-8
- Error messages should be descriptive — this is research code that needs to be debuggable

## Git Workflow

- Main branch: `main`
- Commit messages: descriptive, e.g. "fix: handle edge case in syllable validator for qu- initials"
- Tag experiment results: `git tag exp-baseline-v1`

## Priorities

1. Get the pipeline running end-to-end on 5CD-AI/Viet-Handwriting-OCR
2. Produce baseline numbers (VietOCR, Gemma 4 zero-shot)
3. Run the full pipeline and compare against baselines
4. Generate paper figures (comparison chart, threshold sweep, calibration plot)
