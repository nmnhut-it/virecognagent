# ViRecognAgent

**A Confidence-Gated Multi-Agent Pipeline for Vietnamese Handwritten Text Recognition**

> First multi-agent OCR system for any language — using a single open model family (Gemma 4 E4B) specialized into role-specific agents via system prompts for Vietnamese diacritic verification.

## Architecture

```
Image → Preprocess (OpenCV)
     → Detect text lines (PaddleOCR DB)
     → Recognize (VietOCR / Gemma 4 E4B zero-shot)
     → Confidence Gate
          ├── HIGH (≥ threshold) → Lightweight post-correct → Final text
          └── LOW  (< threshold) → Agent Pod:
                    Proposer (Gemma 4 vision, thinking ON)
                    → Linguist (Gemma 4 text-only, Vietnamese phonotactic check)
                    → Judge (Gemma 4 vision + linguistic evidence)
                    → [max 2 rounds] → Final text
```

## Quick Start

### Option 1: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/virecognagent/blob/main/notebooks/ViRecognAgent_Pipeline.ipynb)

### Option 2: Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/virecognagent.git
cd virecognagent
pip install -r requirements.txt
python scripts/download_datasets.py
python scripts/run_baseline.py --model gemma4 --dataset viet-handwriting
python scripts/run_pipeline.py --threshold 0.85 --max-rounds 2
```

## Datasets

| Dataset | Size | Type | Access |
|---------|------|------|--------|
| **5CD-AI/Viet-Handwriting-OCR** | 23,403 images | Diverse handwriting | [HuggingFace](https://huggingface.co/datasets/5CD-AI/Viet-Handwriting-OCR) |
| **Cinnamon AI Marathon** | 1,838 images | Handwritten addresses | [Google Drive](https://drive.google.com/drive/folders/1Qa2YA6w6V5MaNV-qxqZwlkPsJ5lEBajc) |
| **HANDS-VNOnDB** | 7,296 lines | Online handwriting | [Request access](https://sites.google.com/view/icfhr2018-vohtr-vnondb/database-tools) |

## Experiments

| Experiment | Description |
|-----------|-------------|
| `baseline_gemma4` | Gemma 4 E4B zero-shot (thinking OFF) |
| `baseline_gemma4_think` | Gemma 4 E4B zero-shot (thinking ON) |
| `baseline_vietocr` | VietOCR standalone |
| `two_stage` | VietOCR + Gemma 4 post-correction |
| `full_pipeline` | VietOCR + confidence gate + Agent Pod |
| `ablation_no_linguist` | Remove Linguist agent |
| `ablation_no_judge` | Remove Judge agent |
| `threshold_sweep` | Vary threshold 0.5 → 0.9 |
| `rounds_sweep` | Vary max rounds 1 / 2 / 3 |

## Metrics

- **CER** — Character Error Rate (standard)
- **WER** — Word Error Rate (standard)
- **DER** — Diacritic Error Rate (proposed) — CER computed only over diacritic-bearing characters
- **Latency** — seconds per image
- **Agent Activation Rate** — % of lines routed to Agent Pod

## Citation

```bibtex
@inproceedings{virecognagent2026,
  title={ViRecognAgent: A Confidence-Gated Multi-Agent Pipeline for Vietnamese Handwritten Text Recognition},
  author={Your Name},
  year={2026}
}
```

## License

Apache 2.0
