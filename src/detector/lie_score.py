"""
Lie Score Aggregator — combines the 4 uncertainty signals into a single per-token
"Lie Score" that indicates how likely the model is hallucinating or overconfident.

The Lie Score is the core output of the detection engine:
    lie_score ∈ [0, 1]
    0 = model is likely telling the truth (low uncertainty, well-calibrated)
    1 = model is likely lying (high uncertainty despite high confidence)

The key insight: a high entropy or high gradient norm alone doesn't mean
the model is "lying" — it just means it's uncertain. The LIE is when the
model is confident (low entropy) but internally unstable (high gradient/jacobian)
or inconsistent (high MC variance). That gap is what we detect.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .signals import AllSignals, SignalResult


@dataclass
class LieScoreResult:
    """Per-token lie scores with alert flags."""

    scores: np.ndarray             # (num_tokens,) — aggregated lie score
    is_flagged: np.ndarray         # (num_tokens,) — boolean, True if score > threshold
    confidence: np.ndarray         # (num_tokens,) — model's own confidence (max softmax prob)
    confidence_gap: np.ndarray     # (num_tokens,) — gap between confidence and trustworthiness
    tokens: list[str]
    threshold: float
    num_flagged: int
    overall_lie_score: float       # Mean lie score across all tokens
    signal_contributions: dict     # Per-signal weighted contribution for interpretability

    @property
    def flagged_tokens(self) -> list[tuple[int, str, float]]:
        """Return list of (index, token, score) for all flagged tokens."""
        return [
            (i, self.tokens[i], self.scores[i])
            for i in range(len(self.tokens))
            if self.is_flagged[i]
        ]


@dataclass
class LieScoreWeights:
    """
    Weights for combining signals into the lie score.

    These can be hand-tuned or learned from a calibration set
    (via logistic regression in calibration.py).
    """

    entropy: float = 0.15
    gradient_norm: float = 0.30
    jacobian_norm: float = 0.30
    mc_variance: float = 0.25

    def as_dict(self) -> dict:
        return {
            "entropy": self.entropy,
            "gradient_norm": self.gradient_norm,
            "jacobian_norm": self.jacobian_norm,
            "mc_variance": self.mc_variance,
        }

    def normalize(self) -> "LieScoreWeights":
        """Normalize weights to sum to 1."""
        total = self.entropy + self.gradient_norm + self.jacobian_norm + self.mc_variance
        if total == 0:
            return LieScoreWeights(0.25, 0.25, 0.25, 0.25)
        return LieScoreWeights(
            entropy=self.entropy / total,
            gradient_norm=self.gradient_norm / total,
            jacobian_norm=self.jacobian_norm / total,
            mc_variance=self.mc_variance / total,
        )


class LieScoreAggregator:
    """
    Aggregates the 4 uncertainty signals into a single Lie Score per token.

    The lie score is NOT just a weighted sum of uncertainty signals.
    It specifically captures the DISCREPANCY between the model's expressed
    confidence and its internal uncertainty — this is the "lie."

    Formula:
        raw_uncertainty = w1·entropy + w2·grad_norm + w3·jacobian + w4·mc_variance
        confidence = max(softmax_probs) per token
        lie_score = sigmoid(α · (confidence × raw_uncertainty - β))

    The confidence × uncertainty product means:
    - High confidence + high uncertainty = HIGH lie score (the model is lying)
    - Low confidence + high uncertainty = LOWER lie score (the model admits uncertainty)
    - High confidence + low uncertainty = LOW lie score (the model is legitimately confident)
    """

    def __init__(
        self,
        weights: Optional[LieScoreWeights] = None,
        threshold: float = 0.5,
        alpha: float = 5.0,
        beta: float = 0.3,
    ):
        self.weights = (weights or LieScoreWeights()).normalize()
        self.threshold = threshold
        self.alpha = alpha   # Sigmoid steepness
        self.beta = beta     # Sigmoid center offset

    def compute(
        self,
        signals: AllSignals,
        confidence: np.ndarray,
    ) -> LieScoreResult:
        """
        Compute the lie score from extracted signals and model confidence.

        Args:
            signals: AllSignals from the SignalExtractor
            confidence: Per-token max softmax probability (the model's "confidence")
        """
        w = self.weights

        # Weighted uncertainty aggregate
        raw_uncertainty = (
            w.entropy * signals.entropy.values
            + w.gradient_norm * signals.gradient_norm.values
            + w.jacobian_norm * signals.jacobian_norm.values
            + w.mc_variance * signals.mc_variance.values
        )

        # The lie = confidence × uncertainty product
        # When the model is confident but internally uncertain, this is high
        lie_product = confidence * raw_uncertainty

        # Sigmoid normalization to [0, 1]
        scores = self._sigmoid(self.alpha * (lie_product - self.beta))

        # Flag tokens above threshold
        is_flagged = scores > self.threshold

        # Confidence gap: how much the model's confidence exceeds its trustworthiness
        trustworthiness = 1.0 - scores
        confidence_gap = np.maximum(0, confidence - trustworthiness)

        # Per-signal weighted contributions (for the dashboard breakdown)
        signal_contributions = {
            "entropy": w.entropy * signals.entropy.values,
            "gradient_norm": w.gradient_norm * signals.gradient_norm.values,
            "jacobian_norm": w.jacobian_norm * signals.jacobian_norm.values,
            "mc_variance": w.mc_variance * signals.mc_variance.values,
        }

        return LieScoreResult(
            scores=scores,
            is_flagged=is_flagged,
            confidence=confidence,
            confidence_gap=confidence_gap,
            tokens=signals.tokens,
            threshold=self.threshold,
            num_flagged=int(is_flagged.sum()),
            overall_lie_score=float(scores.mean()),
            signal_contributions=signal_contributions,
        )

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x)),
        )

    def update_weights(self, new_weights: LieScoreWeights) -> None:
        """Update signal weights (e.g., after learning from calibration data)."""
        self.weights = new_weights.normalize()

    def update_threshold(self, new_threshold: float) -> None:
        """Update the flagging threshold."""
        self.threshold = new_threshold


def compute_lie_scores(
    signals: AllSignals,
    softmax_probs: np.ndarray,
    weights: Optional[LieScoreWeights] = None,
    threshold: float = 0.5,
) -> LieScoreResult:
    """
    Convenience function: compute lie scores from signals and softmax probs.

    Args:
        signals: AllSignals from SignalExtractor
        softmax_probs: (num_tokens, vocab_size) softmax probability matrix
        weights: Optional custom weights
        threshold: Flagging threshold

    Returns:
        LieScoreResult with per-token scores, flags, and breakdowns
    """
    # Extract confidence (max prob per token)
    confidence = softmax_probs.max(axis=-1) if softmax_probs.ndim > 1 else softmax_probs

    aggregator = LieScoreAggregator(weights=weights, threshold=threshold)
    return aggregator.compute(signals, confidence)
