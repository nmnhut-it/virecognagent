"""
Confidence calibration for the confidence gate.

Addresses reviewer weakness #4: "Is the confidence signal actually calibrated?"
Provides calibration plots, false positive/negative analysis, and comparison
between confidence estimation methods.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CalibrationResult:
    """Calibration analysis results."""
    bin_edges: list[float]
    bin_accuracies: list[float]    # actual accuracy per bin
    bin_confidences: list[float]   # mean predicted confidence per bin
    bin_counts: list[int]          # samples per bin
    ece: float                     # Expected Calibration Error
    mce: float                     # Maximum Calibration Error
    false_positive_rate: float     # high conf but wrong
    false_negative_rate: float     # low conf but correct
    optimal_threshold: float       # threshold minimizing total error
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "ECE": round(self.ece, 4),
            "MCE": round(self.mce, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "optimal_threshold": round(self.optimal_threshold, 4),
            "n_samples": self.n_samples,
            "bins": [
                {
                    "range": f"{self.bin_edges[i]:.2f}-{self.bin_edges[i+1]:.2f}",
                    "accuracy": round(self.bin_accuracies[i], 4),
                    "confidence": round(self.bin_confidences[i], 4),
                    "count": self.bin_counts[i],
                }
                for i in range(len(self.bin_accuracies))
            ],
        }


def compute_calibration(
    confidences: list[float],
    is_correct: list[bool],
    n_bins: int = 10,
    threshold: float = 0.85,
) -> CalibrationResult:
    """
    Compute calibration metrics for the confidence gate.

    Args:
        confidences: predicted confidence per sample
        is_correct: whether OCR output matched ground truth (exact or CER < epsilon)
        n_bins: number of calibration bins
        threshold: the confidence threshold used for gating

    Returns:
        CalibrationResult with ECE, MCE, calibration curve, and optimal threshold
    """
    confidences = np.array(confidences)
    is_correct = np.array(is_correct, dtype=bool)
    n = len(confidences)

    # Bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        count = mask.sum()
        bin_counts.append(int(count))

        if count > 0:
            acc = is_correct[mask].mean()
            conf = confidences[mask].mean()
            bin_accuracies.append(float(acc))
            bin_confidences.append(float(conf))

            gap = abs(acc - conf)
            ece += gap * count / n
            mce = max(mce, gap)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((lo + hi) / 2)

    # False positive/negative analysis at the given threshold
    high_conf = confidences >= threshold
    low_conf = ~high_conf

    # False positive: high confidence but wrong
    fp = (high_conf & ~is_correct).sum()
    fp_rate = fp / high_conf.sum() if high_conf.sum() > 0 else 0

    # False negative: low confidence but actually correct
    fn = (low_conf & is_correct).sum()
    fn_rate = fn / low_conf.sum() if low_conf.sum() > 0 else 0

    # Find optimal threshold (minimizes fp_rate + fn_rate weighted)
    best_threshold = threshold
    best_cost = float('inf')
    for t in np.arange(0.3, 0.96, 0.05):
        h = confidences >= t
        l = ~h
        cost_fp = (h & ~is_correct).sum() / max(h.sum(), 1)
        cost_fn = (l & is_correct).sum() / max(l.sum(), 1)
        # Weight: false positives cost more (wrong output goes to user)
        # false negatives cost compute (unnecessary agent activation)
        total_cost = 2 * cost_fp + cost_fn
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = float(t)

    return CalibrationResult(
        bin_edges=bin_edges.tolist(),
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
        ece=float(ece),
        mce=float(mce),
        false_positive_rate=float(fp_rate),
        false_negative_rate=float(fn_rate),
        optimal_threshold=best_threshold,
        n_samples=n,
    )


def is_correct_match(prediction: str, reference: str, cer_threshold: float = 0.05) -> bool:
    """
    Determine if a prediction is 'correct' for calibration purposes.
    Uses CER < threshold rather than exact match (too strict for HTR).
    """
    if prediction == reference:
        return True

    import editdistance
    dist = editdistance.eval(prediction, reference)
    ref_len = max(len(reference), 1)
    cer = dist / ref_len
    return cer < cer_threshold


def plot_calibration(
    calibration: CalibrationResult,
    save_path: Optional[str] = None,
    title: str = "Confidence Calibration"
):
    """Generate calibration plot (reliability diagram)."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    midpoints = [(calibration.bin_edges[i] + calibration.bin_edges[i+1]) / 2
                 for i in range(len(calibration.bin_accuracies))]

    ax1.bar(midpoints, calibration.bin_accuracies, width=0.08, alpha=0.7,
            label="Actual accuracy", color="#2196F3")
    ax1.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Confidence")
    ax1.set_ylabel("Actual Accuracy (CER < 5%)")
    ax1.set_title(f"Reliability Diagram\nECE={calibration.ece:.4f}, MCE={calibration.mce:.4f}")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Confidence distribution with FP/FN zones
    ax2.bar(midpoints, calibration.bin_counts, width=0.08, alpha=0.7, color="#4CAF50")
    ax2.axvline(x=calibration.optimal_threshold, color='red', linestyle='--',
                label=f"Optimal threshold: {calibration.optimal_threshold:.2f}")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Sample Count")
    ax2.set_title(
        f"Confidence Distribution\n"
        f"FP rate: {calibration.false_positive_rate:.1%}, "
        f"FN rate: {calibration.false_negative_rate:.1%}"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig
