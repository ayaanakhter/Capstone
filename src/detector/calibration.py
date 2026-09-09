"""
Calibration analysis for the Lie Detector.

Measures how well-calibrated the model's confidence is and quantifies
the improvement when using Lie Score to recalibrate.

Key metrics:
    - ECE (Expected Calibration Error): weighted average of per-bin |accuracy - confidence|
    - MCE (Maximum Calibration Error): worst-case bin miscalibration
    - Reliability diagram: visual plot of confidence bins vs actual accuracy

The "confidence vs correctness" calibration plot is the visual centerpiece
of the project — it's the proof that the model "lies" about how sure it is.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""

    ece: float                         # Expected Calibration Error
    mce: float                         # Maximum Calibration Error
    bin_confidences: np.ndarray        # Mean confidence per bin
    bin_accuracies: np.ndarray         # Actual accuracy per bin
    bin_counts: np.ndarray             # Number of samples per bin
    bin_edges: np.ndarray              # Bin boundaries
    n_bins: int
    total_samples: int

    @property
    def overconfidence_ratio(self) -> float:
        """Fraction of bins where confidence exceeds accuracy."""
        valid = self.bin_counts > 0
        if not valid.any():
            return 0.0
        overconfident = (self.bin_confidences[valid] > self.bin_accuracies[valid]).sum()
        return float(overconfident / valid.sum())


class CalibrationAnalyzer:
    """
    Analyzes model calibration by comparing predicted confidence to actual correctness.

    Usage:
        analyzer = CalibrationAnalyzer(n_bins=15)

        # Accumulate predictions over many examples
        analyzer.add_predictions(confidences, correctness)
        analyzer.add_predictions(more_confidences, more_correctness)

        # Compute calibration metrics
        result = analyzer.compute()

        # Plot reliability diagram
        analyzer.plot_reliability_diagram("calibration_plot.png")
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self.all_confidences: list[float] = []
        self.all_correctness: list[bool] = []

    def add_predictions(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
    ) -> None:
        """
        Accumulate predictions for calibration analysis.

        Args:
            confidences: Per-token model confidence (max softmax prob), shape (N,)
            correctness: Per-token binary correctness label, shape (N,)
                         1 = model's top prediction was correct, 0 = incorrect
        """
        self.all_confidences.extend(confidences.tolist())
        self.all_correctness.extend(correctness.tolist())

    def compute(self) -> CalibrationResult:
        """Compute ECE, MCE, and per-bin statistics."""
        confidences = np.array(self.all_confidences)
        correctness = np.array(self.all_correctness)
        n = len(confidences)

        if n == 0:
            return CalibrationResult(
                ece=0.0, mce=0.0,
                bin_confidences=np.zeros(self.n_bins),
                bin_accuracies=np.zeros(self.n_bins),
                bin_counts=np.zeros(self.n_bins, dtype=int),
                bin_edges=np.linspace(0, 1, self.n_bins + 1),
                n_bins=self.n_bins,
                total_samples=0,
            )

        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_confidences = np.zeros(self.n_bins)
        bin_accuracies = np.zeros(self.n_bins)
        bin_counts = np.zeros(self.n_bins, dtype=int)

        # Assign each prediction to a bin
        bin_indices = np.digitize(confidences, bin_edges[1:-1])

        for b in range(self.n_bins):
            mask = bin_indices == b
            count = mask.sum()
            bin_counts[b] = count

            if count > 0:
                bin_confidences[b] = confidences[mask].mean()
                bin_accuracies[b] = correctness[mask].mean()

        # ECE = weighted average of per-bin calibration error
        valid = bin_counts > 0
        weights = bin_counts[valid] / n
        errors = np.abs(bin_accuracies[valid] - bin_confidences[valid])
        ece = float((weights * errors).sum())

        # MCE = maximum bin calibration error
        mce = float(errors.max()) if len(errors) > 0 else 0.0

        return CalibrationResult(
            ece=ece,
            mce=mce,
            bin_confidences=bin_confidences,
            bin_accuracies=bin_accuracies,
            bin_counts=bin_counts,
            bin_edges=bin_edges,
            n_bins=self.n_bins,
            total_samples=n,
        )

    def plot_reliability_diagram(
        self,
        save_path: Optional[str] = None,
        title: str = "Reliability Diagram — Confidence vs Accuracy",
        show_gap: bool = True,
        show_histogram: bool = True,
        figsize: tuple = (10, 8),
    ) -> plt.Figure:
        """
        Plot the reliability diagram (calibration plot).

        This is the visual centerpiece: bars showing per-bin accuracy vs a
        diagonal "perfect calibration" line. The gap between bars and the
        diagonal IS the model's "lie."
        """
        result = self.compute()

        fig, ax1 = plt.subplots(figsize=figsize)

        # Dark theme styling
        fig.patch.set_facecolor("#1a1a2e")
        ax1.set_facecolor("#16213e")

        valid = result.bin_counts > 0
        bin_centers = (result.bin_edges[:-1] + result.bin_edges[1:]) / 2
        bar_width = 1.0 / result.n_bins * 0.85

        # Perfect calibration line
        ax1.plot(
            [0, 1], [0, 1],
            "--", color="#e94560", linewidth=2, alpha=0.8,
            label="Perfect Calibration",
        )

        # Accuracy bars
        colors = []
        for i in range(result.n_bins):
            if result.bin_counts[i] == 0:
                colors.append("#333333")
            else:
                gap = result.bin_confidences[i] - result.bin_accuracies[i]
                if gap > 0.1:
                    colors.append("#e94560")  # Overconfident — red
                elif gap < -0.1:
                    colors.append("#0f3460")  # Underconfident — blue
                else:
                    colors.append("#00d2ff")  # Well-calibrated — cyan

        ax1.bar(
            bin_centers[valid],
            result.bin_accuracies[valid],
            width=bar_width,
            color=[c for c, v in zip(colors, valid) if v],
            edgecolor="#ffffff22",
            alpha=0.85,
            label="Accuracy",
            zorder=3,
        )

        # Gap fill (overconfidence region)
        if show_gap:
            for i in range(result.n_bins):
                if result.bin_counts[i] > 0 and result.bin_confidences[i] > result.bin_accuracies[i]:
                    ax1.bar(
                        bin_centers[i],
                        result.bin_confidences[i] - result.bin_accuracies[i],
                        bottom=result.bin_accuracies[i],
                        width=bar_width,
                        color="#e94560",
                        alpha=0.3,
                        zorder=2,
                    )

        ax1.set_xlabel("Confidence", fontsize=13, color="white", fontfamily="sans-serif")
        ax1.set_ylabel("Accuracy", fontsize=13, color="white", fontfamily="sans-serif")
        ax1.set_title(title, fontsize=15, color="white", fontweight="bold", pad=15)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.tick_params(colors="white")
        ax1.legend(loc="upper left", fontsize=11, facecolor="#16213e", edgecolor="white", labelcolor="white")

        # Add ECE/MCE annotation
        ax1.text(
            0.98, 0.05,
            f"ECE = {result.ece:.4f}\nMCE = {result.mce:.4f}\nN = {result.total_samples}",
            transform=ax1.transAxes,
            fontsize=11, color="#00d2ff",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f3460", alpha=0.8),
            fontfamily="monospace",
        )

        # Histogram overlay (sample counts per bin)
        if show_histogram:
            ax2 = ax1.twinx()
            ax2.bar(
                bin_centers,
                result.bin_counts,
                width=bar_width,
                color="#ffffff",
                alpha=0.08,
                zorder=1,
            )
            ax2.set_ylabel("Count", fontsize=11, color="#666666")
            ax2.tick_params(colors="#666666")
            max_count = max(result.bin_counts.max(), 1)
            ax2.set_ylim(0, max_count * 3)  # Keep histogram subtle

        # Grid
        ax1.grid(True, alpha=0.1, color="white")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

        return fig

    def reset(self) -> None:
        """Clear accumulated predictions."""
        self.all_confidences.clear()
        self.all_correctness.clear()


def compare_calibration(
    before: CalibrationResult,
    after: CalibrationResult,
) -> dict:
    """
    Compare calibration before and after Lie Score recalibration.

    Returns a summary dict with improvement metrics.
    """
    ece_improvement = (before.ece - after.ece) / before.ece * 100 if before.ece > 0 else 0
    mce_improvement = (before.mce - after.mce) / before.mce * 100 if before.mce > 0 else 0

    return {
        "ece_before": before.ece,
        "ece_after": after.ece,
        "ece_improvement_pct": ece_improvement,
        "mce_before": before.mce,
        "mce_after": after.mce,
        "mce_improvement_pct": mce_improvement,
        "overconfidence_before": before.overconfidence_ratio,
        "overconfidence_after": after.overconfidence_ratio,
    }
