"""
Unit tests for the Lie Detector signal extractors.

Tests cover:
    1. SoftmaxEntropy — correctness on known distributions
    2. GradientNorm — non-zero output when grads available
    3. MCDropoutVariance — variance computation from stochastic passes
    4. LieScoreAggregator — scoring logic and flagging
    5. CalibrationAnalyzer — ECE computation on synthetic data
"""

import numpy as np
import torch
import torch.nn.functional as F
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detector.signals import SoftmaxEntropy, GradientNorm, MCDropoutVariance, SignalResult
from src.detector.lie_score import LieScoreAggregator, LieScoreWeights, compute_lie_scores
from src.detector.calibration import CalibrationAnalyzer
from src.detector.model import GenerationOutput


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_dummy_output(
    num_tokens: int = 5,
    vocab_size: int = 100,
    embed_dim: int = 32,
    seq_len: int = 10,
    uniform_probs: bool = False,
    confident_probs: bool = False,
    include_grads: bool = True,
    n_mc_passes: int = 0,
) -> GenerationOutput:
    """Create a synthetic GenerationOutput for testing."""
    if uniform_probs:
        probs = torch.ones(num_tokens, vocab_size) / vocab_size
    elif confident_probs:
        probs = torch.zeros(num_tokens, vocab_size)
        probs[:, 0] = 0.95
        probs[:, 1:] = 0.05 / (vocab_size - 1)
    else:
        logits = torch.randn(num_tokens, vocab_size)
        probs = F.softmax(logits, dim=-1)

    logits = torch.log(probs + 1e-10)

    grads = None
    if include_grads:
        grads = [torch.randn(1, seq_len + i, embed_dim) for i in range(num_tokens)]

    mc_logits = None
    if n_mc_passes > 0:
        mc_logits = [torch.randn(num_tokens, vocab_size) for _ in range(n_mc_passes)]

    tokens = [f"tok_{i}" for i in range(num_tokens)]

    return GenerationOutput(
        generated_text=" ".join(tokens),
        generated_token_ids=list(range(num_tokens)),
        generated_tokens=tokens,
        logits_sequence=logits,
        softmax_probs_sequence=probs,
        embedding_grads=grads,
        mc_logits=mc_logits,
        input_ids=torch.zeros(1, seq_len, dtype=torch.long),
        input_text="test prompt",
    )


# ─────────────────────────────────────────────
# Test SoftmaxEntropy
# ─────────────────────────────────────────────

class TestSoftmaxEntropy:
    def test_uniform_distribution_has_max_entropy(self):
        """Uniform distribution should have normalized entropy ≈ 1.0"""
        output = make_dummy_output(num_tokens=3, vocab_size=100, uniform_probs=True)
        signal = SoftmaxEntropy(normalize=True)
        result = signal.compute(output)

        assert result.name == "softmax_entropy"
        assert len(result.values) == 3
        np.testing.assert_allclose(result.values, 1.0, atol=0.01)

    def test_confident_distribution_has_low_entropy(self):
        """Highly confident distribution should have low entropy."""
        output = make_dummy_output(num_tokens=3, vocab_size=100, confident_probs=True)
        signal = SoftmaxEntropy(normalize=True)
        result = signal.compute(output)

        assert all(v < 0.2 for v in result.values)

    def test_output_shape_matches_tokens(self):
        """Signal values should have one entry per generated token."""
        for n in [1, 5, 20]:
            output = make_dummy_output(num_tokens=n)
            result = SoftmaxEntropy().compute(output)
            assert len(result.values) == n
            assert len(result.tokens) == n


# ─────────────────────────────────────────────
# Test GradientNorm
# ─────────────────────────────────────────────

class TestGradientNorm:
    def test_nonzero_when_grads_available(self):
        """Should produce non-zero values when gradients are provided."""
        output = make_dummy_output(num_tokens=5, include_grads=True)
        signal = GradientNorm(normalize=True)
        result = signal.compute(output)

        assert result.name == "gradient_norm"
        assert len(result.values) == 5
        assert any(v > 0 for v in result.values)

    def test_zeros_when_no_grads(self):
        """Should return zeros gracefully when no gradients are available."""
        output = make_dummy_output(num_tokens=3, include_grads=False)
        signal = GradientNorm()
        result = signal.compute(output)

        np.testing.assert_array_equal(result.values, [0.0, 0.0, 0.0])

    def test_normalized_range(self):
        """Normalized values should be in [0, 1]."""
        output = make_dummy_output(num_tokens=10, include_grads=True)
        result = GradientNorm(normalize=True).compute(output)

        assert all(0 <= v <= 1.0 for v in result.values)


# ─────────────────────────────────────────────
# Test MCDropoutVariance
# ─────────────────────────────────────────────

class TestMCDropoutVariance:
    def test_nonzero_with_mc_passes(self):
        """Should produce non-zero variance when MC passes are provided."""
        output = make_dummy_output(num_tokens=5, n_mc_passes=5)
        signal = MCDropoutVariance(normalize=True)
        result = signal.compute(output)

        assert result.name == "mc_dropout_variance"
        assert len(result.values) == 5
        assert any(v > 0 for v in result.values)

    def test_zeros_when_no_mc_data(self):
        """Should return zeros when no MC Dropout data is available."""
        output = make_dummy_output(num_tokens=3, n_mc_passes=0)
        result = MCDropoutVariance().compute(output)

        np.testing.assert_array_equal(result.values, [0.0, 0.0, 0.0])

    def test_identical_passes_give_zero_variance(self):
        """If all MC passes produce identical logits, variance should be ~0."""
        output = make_dummy_output(num_tokens=3, n_mc_passes=5)
        # Make all passes identical
        output.mc_logits = [output.mc_logits[0].clone() for _ in range(5)]
        result = MCDropoutVariance(normalize=False).compute(output)

        np.testing.assert_allclose(result.values, 0.0, atol=1e-6)


# ─────────────────────────────────────────────
# Test LieScoreAggregator
# ─────────────────────────────────────────────

class TestLieScoreAggregator:
    def test_high_confidence_high_uncertainty_means_lying(self):
        """
        When model is confident (high max prob) but signals show high uncertainty,
        the lie score should be HIGH.
        """
        from src.detector.signals import AllSignals

        num_tokens = 5
        tokens = [f"t_{i}" for i in range(num_tokens)]

        # All uncertainty signals maxed out
        high_uncertainty_signals = AllSignals(
            entropy=SignalResult("e", np.ones(num_tokens), tokens),
            gradient_norm=SignalResult("g", np.ones(num_tokens), tokens),
            jacobian_norm=SignalResult("j", np.ones(num_tokens), tokens),
            mc_variance=SignalResult("m", np.ones(num_tokens), tokens),
            tokens=tokens,
            num_tokens=num_tokens,
        )

        # Model is very confident
        confidence = np.ones(num_tokens) * 0.95

        aggregator = LieScoreAggregator(threshold=0.5)
        result = aggregator.compute(high_uncertainty_signals, confidence)

        # All tokens should have high lie score
        assert all(s > 0.5 for s in result.scores), f"Expected high lie scores, got {result.scores}"
        assert result.num_flagged == num_tokens

    def test_low_confidence_high_uncertainty_is_honest(self):
        """
        When model is NOT confident and uncertainty is high,
        lie score should be LOWER (model is honestly uncertain).
        """
        from src.detector.signals import AllSignals

        num_tokens = 5
        tokens = [f"t_{i}" for i in range(num_tokens)]

        signals = AllSignals(
            entropy=SignalResult("e", np.ones(num_tokens), tokens),
            gradient_norm=SignalResult("g", np.ones(num_tokens), tokens),
            jacobian_norm=SignalResult("j", np.ones(num_tokens), tokens),
            mc_variance=SignalResult("m", np.ones(num_tokens), tokens),
            tokens=tokens,
            num_tokens=num_tokens,
        )

        # Model has LOW confidence
        confidence = np.ones(num_tokens) * 0.1

        aggregator = LieScoreAggregator(threshold=0.5)
        result = aggregator.compute(signals, confidence)

        # Lie score should be lower than the confident case
        assert result.overall_lie_score < 0.5

    def test_confident_and_stable_is_truthful(self):
        """
        When model is confident AND internally stable (low uncertainty),
        lie score should be LOW.
        """
        from src.detector.signals import AllSignals

        num_tokens = 5
        tokens = [f"t_{i}" for i in range(num_tokens)]

        # All signals near zero — model is internally stable
        signals = AllSignals(
            entropy=SignalResult("e", np.zeros(num_tokens), tokens),
            gradient_norm=SignalResult("g", np.zeros(num_tokens), tokens),
            jacobian_norm=SignalResult("j", np.zeros(num_tokens), tokens),
            mc_variance=SignalResult("m", np.zeros(num_tokens), tokens),
            tokens=tokens,
            num_tokens=num_tokens,
        )

        confidence = np.ones(num_tokens) * 0.95

        aggregator = LieScoreAggregator(threshold=0.5)
        result = aggregator.compute(signals, confidence)

        assert all(s < 0.5 for s in result.scores)
        assert result.num_flagged == 0


# ─────────────────────────────────────────────
# Test CalibrationAnalyzer
# ─────────────────────────────────────────────

class TestCalibrationAnalyzer:
    def test_perfect_calibration_has_zero_ece(self):
        """A perfectly calibrated model should have ECE ≈ 0."""
        analyzer = CalibrationAnalyzer(n_bins=10)

        # Create perfectly calibrated predictions:
        # confidence 0.9 → 90% correct, confidence 0.5 → 50% correct, etc.
        np.random.seed(42)
        for conf_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
            n = 1000
            confidences = np.full(n, conf_level)
            correctness = np.random.binomial(1, conf_level, n).astype(float)
            analyzer.add_predictions(confidences, correctness)

        result = analyzer.compute()
        assert result.ece < 0.05, f"ECE should be near 0, got {result.ece}"

    def test_overconfident_model_has_high_ece(self):
        """A model that's always 90% confident but only 30% correct should have high ECE."""
        analyzer = CalibrationAnalyzer(n_bins=10)

        n = 1000
        confidences = np.full(n, 0.9)
        correctness = np.random.binomial(1, 0.3, n).astype(float)
        analyzer.add_predictions(confidences, correctness)

        result = analyzer.compute()
        assert result.ece > 0.4, f"ECE should be high, got {result.ece}"
        assert result.overconfidence_ratio > 0.5

    def test_empty_analyzer(self):
        """Should handle empty state gracefully."""
        analyzer = CalibrationAnalyzer()
        result = analyzer.compute()
        assert result.ece == 0.0
        assert result.total_samples == 0

    def test_reset_clears_state(self):
        """Reset should clear all accumulated predictions."""
        analyzer = CalibrationAnalyzer()
        analyzer.add_predictions(np.array([0.5]), np.array([1.0]))
        assert len(analyzer.all_confidences) == 1

        analyzer.reset()
        assert len(analyzer.all_confidences) == 0


# ─────────────────────────────────────────────
# Integration test
# ─────────────────────────────────────────────

class TestFullPipeline:
    def test_end_to_end_no_model(self):
        """
        Test the full pipeline from GenerationOutput → signals → lie scores
        using synthetic data (no actual GPT-2 model needed).
        """
        from src.detector.signals import AllSignals, SignalExtractor

        # Create dummy output with all signal data
        output = make_dummy_output(
            num_tokens=10,
            vocab_size=50,
            include_grads=True,
            n_mc_passes=5,
        )

        # Extract signals (skip Jacobian since it needs the model)
        extractor = SignalExtractor(compute_jacobian=False)
        signals = extractor.extract_all(output, model=None)

        assert signals.num_tokens == 10
        assert len(signals.entropy.values) == 10
        assert len(signals.gradient_norm.values) == 10
        assert len(signals.mc_variance.values) == 10

        # Compute lie scores
        confidence = output.softmax_probs_sequence.numpy().max(axis=-1)
        result = compute_lie_scores(signals, output.softmax_probs_sequence.numpy())

        assert len(result.scores) == 10
        assert all(0 <= s <= 1 for s in result.scores)
        assert len(result.tokens) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
