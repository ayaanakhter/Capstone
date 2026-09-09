"""
Lie Detector — Detection engine package.

Core modules:
    - model: GPT-2 wrapper with hook instrumentation
    - signals: 4 uncertainty signal extractors
    - lie_score: Aggregator + threshold alerting
    - calibration: ECE/MCE computation, reliability diagrams
"""

from .model import LieDetectorModel, TokenSignal, GenerationOutput
from .signals import (
    SignalExtractor,
    AllSignals,
    SignalResult,
    SoftmaxEntropy,
    GradientNorm,
    JacobianSpectralNorm,
    MCDropoutVariance,
)
from .lie_score import (
    LieScoreAggregator,
    LieScoreResult,
    LieScoreWeights,
    compute_lie_scores,
)
from .calibration import CalibrationAnalyzer, CalibrationResult, compare_calibration

__all__ = [
    "LieDetectorModel",
    "TokenSignal",
    "SignalExtractor",
    "AllSignals",
    "SignalResult",
    "SoftmaxEntropy",
    "GradientNorm",
    "JacobianSpectralNorm",
    "MCDropoutVariance",
    "LieScoreAggregator",
    "LieScoreResult",
    "LieScoreWeights",
    "compute_lie_scores",
    "CalibrationAnalyzer",
    "CalibrationResult",
    "compare_calibration",
]
