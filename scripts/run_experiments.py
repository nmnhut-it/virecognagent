"""
Run experiments for ViRecognAgent paper.
Evaluates all pipeline modes and ablations on Vietnamese handwriting datasets.
"""

import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import ViRecognAgentPipeline
from src.metrics import evaluate, HTRMetrics


def load_dataset(data_dir: str, split: str = "test", max_samples: int = None):
    """Load evaluation dataset."""
    # Try HuggingFace-style split directories
    labels_path = os.path.join(data_dir, split, "labels.json")
    if not os.path.exists(labels_path):
        labels_path = os.path.join(data_dir, f"{split}.json")
    if not os.path.exists(labels_path):
        # Try loading directly from HuggingFace
        from datasets import load_dataset as hf_load
        ds = hf_load("5CD-AI/Viet-Handwriting-OCR", split=split)
        samples = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            samples.append({"image": item["image"], "text": item.get("text", item.get("label", ""))})
        return samples

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if max_samples:
        labels = labels[:max_samples]

    # Load images
    base_dir = os.path.dirname(labels_path)
    for label in labels:
        img_path = os.path.join(base_dir, label["image"])
        if os.path.exists(img_path):
            label["image"] = Image.open(img_path).convert("RGB")
        else:
            label["image"] = None

    return [l for l in labels if l["image"] is not None]


def run_experiment(
    mode: str,
    data_dir: str,
    split: str = "test",
    max_samples: int = None,
    confidence_threshold: float = 0.85,
    max_rounds: int = 2,
    output_dir: str = "outputs",
    gemma_quantization: str = "4bit",
):
    """Run a single experiment configuration."""

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {mode}")
    print(f"  Dataset: {data_dir} | Split: {split}")
    print(f"  Threshold: {confidence_threshold} | Max rounds: {max_rounds}")
    print(f"{'='*70}\n")

    # Load data
    samples = load_dataset(data_dir, split, max_samples)
    print(f"Loaded {len(samples)} samples\n")

    # Init pipeline
    pipeline = ViRecognAgentPipeline(
        mode=mode,
        confidence_threshold=confidence_threshold,
        max_rounds=max_rounds,
        gemma_quantization=gemma_quantization,
    )

    # Run recognition
    predictions = []
    references = []
    timings = []
    details = []

    for sample in tqdm(samples, desc=f"Running {mode}"):
        image = sample["image"]
        gt_text = sample["text"]

        start = time.time()
        region_out = pipeline.process_single_line(image)
        elapsed = time.time() - start

        pred_text = region_out.text
        predictions.append(pred_text)
        references.append(gt_text)
        timings.append(elapsed)

        details.append({
            "ground_truth": gt_text,
            "prediction": pred_text,
            "confidence": region_out.confidence,
            "method": region_out.method,
            "agent_activated": region_out.agent_activated,
            "diacritics": region_out.diacritics.to_dict() if region_out.diacritics else {},
            "agent_trace": region_out.agent_trace.to_dict() if region_out.agent_trace else None,
            "time_seconds": elapsed,
        })

    # Compute metrics
    metrics = evaluate(predictions, references)

    # Compute timing stats
    avg_time = sum(timings) / len(timings) if timings else 0
    agent_rate = sum(1 for d in details if d["agent_activated"]) / len(details) if details else 0

    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS: {mode}")
    print(f"{'='*70}")
    print(metrics)
    print(f"\nAvg time/sample: {avg_time:.3f}s")
    print(f"Agent activation rate: {agent_rate:.1%}")
    print(f"{'='*70}\n")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(output_dir, f"{mode}_{timestamp}.json")

    result_data = {
        "experiment": mode,
        "timestamp": timestamp,
        "config": {
            "confidence_threshold": confidence_threshold,
            "max_rounds": max_rounds,
            "max_samples": max_samples,
            "split": split,
        },
        "metrics": metrics.to_dict(),
        "timing": {
            "avg_seconds_per_sample": avg_time,
            "total_seconds": sum(timings),
        },
        "agent_activation_rate": agent_rate,
        "details": details,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {result_path}")
    return metrics, result_data


def run_all_experiments(
    data_dir: str,
    split: str = "test",
    max_samples: int = None,
    output_dir: str = "outputs",
):
    """Run the complete experiment suite from the paper."""

    experiments = [
        # ── Single-model baselines (Reviewer Weakness #1) ──
        # These MUST be included to prove the triad adds value
        {"mode": "vietocr", "threshold": 0.85, "rounds": 2},
        {"mode": "vietocr_ensemble", "threshold": 0.85, "rounds": 2},
        {"mode": "gemma4_zeroshot", "threshold": 0.85, "rounds": 2},
        {"mode": "gemma4_think", "threshold": 0.85, "rounds": 2},
        {"mode": "gemma4_comprehensive", "threshold": 0.85, "rounds": 2},
        {"mode": "gemma4_cot", "threshold": 0.85, "rounds": 2},
        {"mode": "gemma4_sc", "threshold": 0.85, "rounds": 2},
        # ── Two-stage ──
        {"mode": "two_stage", "threshold": 0.85, "rounds": 2},
        # ── Full pipeline (main result) ──
        {"mode": "full", "threshold": 0.85, "rounds": 2},
        # ── Ablations ──
        {"mode": "ablation_no_linguist", "threshold": 0.85, "rounds": 2},
        {"mode": "ablation_no_judge", "threshold": 0.85, "rounds": 2},
        {"mode": "ablation_tools_only", "threshold": 0.85, "rounds": 2},
    ]

    # Threshold sweep
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        experiments.append({"mode": "full", "threshold": threshold, "rounds": 2})

    # Rounds sweep
    for rounds in [1, 2, 3]:
        experiments.append({"mode": "full", "threshold": 0.85, "rounds": rounds})

    all_results = []
    for exp in experiments:
        try:
            metrics, result = run_experiment(
                mode=exp["mode"],
                data_dir=data_dir,
                split=split,
                max_samples=max_samples,
                confidence_threshold=exp["threshold"],
                max_rounds=exp["rounds"],
                output_dir=output_dir,
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR in {exp['mode']}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    summary = [
        {
            "experiment": r["experiment"],
            "config": r["config"],
            "metrics": r["metrics"],
            "agent_activation_rate": r.get("agent_activation_rate", 0),
        }
        for r in all_results
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nAll results saved to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ViRecognAgent experiments")
    parser.add_argument("--mode", default="all", help="Experiment mode or 'all'")
    parser.add_argument("--data-dir", default="data/viet_handwriting")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    args = parser.parse_args()

    if args.mode == "all":
        run_all_experiments(
            data_dir=args.data_dir,
            split=args.split,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
        )
    else:
        run_experiment(
            mode=args.mode,
            data_dir=args.data_dir,
            split=args.split,
            max_samples=args.max_samples,
            confidence_threshold=args.threshold,
            max_rounds=args.max_rounds,
            output_dir=args.output_dir,
            gemma_quantization=args.quantization,
        )
