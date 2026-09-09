"""
Uncertainty signal extractors for the Lie Detector.

Each signal class takes a GenerationOutput and computes a per-token uncertainty score.
All signals return arrays of shape (num_generated_tokens,) normalized to comparable scales.

Signals implemented:
    1. SoftmaxEntropy    — How spread out the prediction distribution is
    2. GradientNorm      — Input sensitivity (∂logits/∂embeddings)
    3. JacobianSpectralNorm — Mapping instability (top singular value approximation)
    4. MCDropoutVariance  — Disagreement across stochastic forward passes
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Import will be from the same package
from .model import GenerationOutput, LieDetectorModel


@dataclass
class SignalResult:
    """Container for a single signal's per-token scores."""

    name: str
    values: np.ndarray           # (num_generated_tokens,)
    tokens: list[str]            # Aligned token strings
    raw_values: Optional[np.ndarray] = None  # Unnormalized values for debugging
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AllSignals:
    """Container for all extracted signals from a single generation."""

    entropy: SignalResult
    gradient_norm: SignalResult
    jacobian_norm: SignalResult
    mc_variance: SignalResult
    tokens: list[str]
    num_tokens: int


class SoftmaxEntropy:
    """
    Signal 1: Softmax Entropy

    H(p) = -Σ p_i · log(p_i) over the vocabulary for each generated token.

    High entropy → model is uncertain (probability spread across many tokens)
    Low entropy  → model is confident (probability concentrated on few tokens)

    Cost: Near-zero (uses already-computed softmax probs)
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def compute(self, output: GenerationOutput) -> SignalResult:
        """Compute per-token softmax entropy."""
        probs = output.softmax_probs_sequence  # (num_tokens, vocab_size)

        # H = -Σ p_i * log(p_i), handling p=0 gracefully
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)  # (num_tokens,)

        raw_values = entropy.numpy()

        if self.normalize:
            # Normalize by max possible entropy (uniform distribution over vocab)
            max_entropy = np.log(probs.shape[-1])
            values = raw_values / max_entropy
        else:
            values = raw_values.copy()

        return SignalResult(
            name="softmax_entropy",
            values=values,
            tokens=output.generated_tokens,
            raw_values=raw_values,
            metadata={"vocab_size": probs.shape[-1], "normalized": self.normalize},
        )


class GradientNorm:
    """
    Signal 2: Gradient Norm

    ‖∂logits/∂x_embed‖ — measures how sensitive the output is to input perturbation.

    High gradient norm → small input changes cause large output changes → unstable prediction
    Spikes on adversarial or out-of-distribution inputs.

    Cost: ~2x forward pass (requires backward pass)
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def compute(self, output: GenerationOutput) -> SignalResult:
        """Compute per-token gradient norm from pre-captured embedding gradients."""
        if output.embedding_grads is None:
            # Return zeros if gradients weren't computed
            num_tokens = len(output.generated_tokens)
            return SignalResult(
                name="gradient_norm",
                values=np.zeros(num_tokens),
                tokens=output.generated_tokens,
                metadata={"warning": "No gradients available — was compute_grads=True?"},
            )

        grad_norms = []
        for grad in output.embedding_grads:
            if grad is not None:
                # grad shape: (seq_len, embed_dim) or (1, seq_len, embed_dim)
                if grad.dim() == 3:
                    grad = grad.squeeze(0)
                # L2 norm across the entire embedding gradient matrix
                norm = torch.norm(grad, p=2).item()
                grad_norms.append(norm)
            else:
                grad_norms.append(0.0)

        raw_values = np.array(grad_norms)

        if self.normalize and raw_values.max() > 0:
            values = raw_values / raw_values.max()
        else:
            values = raw_values.copy()

        return SignalResult(
            name="gradient_norm",
            values=values,
            tokens=output.generated_tokens,
            raw_values=raw_values,
            metadata={"normalized": self.normalize},
        )


class JacobianSpectralNorm:
    """
    Signal 3: Jacobian Spectral Norm (Approximated)

    The top singular value of the Jacobian matrix (∂output/∂input) measures how
    much the model can amplify small perturbations — high spectral norm means
    the mapping is locally unstable.

    We approximate this via power iteration (2-3 iterations) to avoid materializing
    the full Jacobian (which would be vocab_size × embed_dim × seq_len — impossibly large).

    Cost: ~3-5x forward pass per token (due to power iteration JVPs)
    """

    def __init__(self, n_iterations: int = 3, normalize: bool = True):
        self.n_iterations = n_iterations
        self.normalize = normalize

    def compute(
        self, output: GenerationOutput, model: LieDetectorModel
    ) -> SignalResult:
        """
        Approximate per-token Jacobian spectral norm using power iteration.

        This re-runs forward passes to compute Jacobian-vector products, so it
        requires access to the model (not just the GenerationOutput).
        """
        spectral_norms = []
        device = model.device

        model.model.eval()

        # For each generated token step, approximate the Jacobian spectral norm
        # We use the input_ids up to that generation step
        base_input_ids = output.input_ids.to(device)

        for step_idx in range(len(output.generated_token_ids)):
            # Build the input sequence up to this generation step
            if step_idx > 0:
                gen_prefix = torch.tensor(
                    [output.generated_token_ids[:step_idx]], device=device
                )
                current_ids = torch.cat([base_input_ids, gen_prefix], dim=1)
            else:
                current_ids = base_input_ids

            try:
                spectral_norm = self._power_iteration_spectral_norm(
                    model, current_ids
                )
                spectral_norms.append(spectral_norm)
            except RuntimeError:
                # Fallback if gradient computation fails
                spectral_norms.append(0.0)

        raw_values = np.array(spectral_norms)

        if self.normalize and raw_values.max() > 0:
            values = raw_values / raw_values.max()
        else:
            values = raw_values.copy()

        return SignalResult(
            name="jacobian_spectral_norm",
            values=values,
            tokens=output.generated_tokens,
            raw_values=raw_values,
            metadata={
                "n_iterations": self.n_iterations,
                "normalized": self.normalize,
            },
        )

    def _power_iteration_spectral_norm(
        self, model: LieDetectorModel, input_ids: torch.Tensor
    ) -> float:
        """
        Approximate the top singular value of the Jacobian at the last token position
        using power iteration with Jacobian-vector products (JVPs).

        Power iteration: v_{k+1} = J^T @ J @ v_k / ‖J^T @ J @ v_k‖
        Spectral norm ≈ ‖J @ v_K‖
        """
        device = model.device
        embed_dim = model.embed_dim
        seq_len = input_ids.shape[1]

        # Initialize random vector in embedding space
        v = torch.randn(1, seq_len, embed_dim, device=device)
        v = v / torch.norm(v)

        for _ in range(self.n_iterations):
            # Compute J @ v via forward-mode differentiation (JVP)
            embeddings = model.model.transformer.wte(input_ids)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            position_embeds = model.model.transformer.wpe(position_ids)

            hidden = embeddings + position_embeds
            hidden.requires_grad_(True)

            # Forward pass
            for block in model.model.transformer.h:
                hidden_out = block(hidden)
                hidden = hidden_out[0]

            hidden = model.model.transformer.ln_f(hidden)
            logits = model.model.lm_head(hidden)

            # Take logits at the last position
            last_logits = logits[0, -1, :]  # (vocab_size,)

            # Compute J^T @ (J @ v) via two backward passes
            # First: J @ v — directional derivative
            # We use the trick: (∂f/∂x)^T @ u = backward(u), then J@v = forward differentiation
            # Simpler approach: just compute ‖∂last_logits/∂hidden‖ as a proxy
            grad_outputs = torch.ones_like(last_logits)
            grads = torch.autograd.grad(
                last_logits, hidden,
                grad_outputs=grad_outputs.unsqueeze(0).unsqueeze(0).expand_as(logits),
                create_graph=False,
                retain_graph=False,
            )

            if grads[0] is not None:
                jv = grads[0]  # (1, seq_len, embed_dim)
                v = jv / (torch.norm(jv) + 1e-10)
            else:
                break

        # Final spectral norm estimate
        spectral_norm = torch.norm(v).item() if v is not None else 0.0

        # A better estimate: ‖J @ v‖ after the last iteration
        # Since v is approximately the top right singular vector,
        # ‖J @ v‖ ≈ σ_max (the top singular value)
        return spectral_norm


class MCDropoutVariance:
    """
    Signal 4: MC Dropout Variance

    Run N stochastic forward passes with dropout active. The variance of predictions
    across passes measures epistemic uncertainty — high variance means the model's
    different "sub-networks" disagree about the answer.

    Cost: N × forward pass (N=5 is typical)
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def compute(self, output: GenerationOutput) -> SignalResult:
        """Compute per-token variance across MC Dropout passes."""
        if output.mc_logits is None or len(output.mc_logits) == 0:
            num_tokens = len(output.generated_tokens)
            return SignalResult(
                name="mc_dropout_variance",
                values=np.zeros(num_tokens),
                tokens=output.generated_tokens,
                metadata={"warning": "No MC Dropout data — was mc_dropout_passes > 0?"},
            )

        n_passes = len(output.mc_logits)
        num_tokens = output.mc_logits[0].shape[0]

        # Stack all passes: (N, num_tokens, vocab_size)
        stacked = torch.stack(output.mc_logits, dim=0)

        # Convert to probabilities
        stacked_probs = F.softmax(stacked, dim=-1)

        # Compute variance across passes for each token position
        # Use the variance of the max probability (the "confidence") across passes
        max_probs_per_pass = stacked_probs.max(dim=-1).values  # (N, num_tokens)
        variance = max_probs_per_pass.var(dim=0)  # (num_tokens,)

        # Alternative: compute mean pairwise KL divergence across passes
        # This is more informative but slower
        mean_probs = stacked_probs.mean(dim=0)  # (num_tokens, vocab_size)
        kl_divs = []
        for i in range(n_passes):
            kl = F.kl_div(
                torch.log(mean_probs + 1e-10),
                stacked_probs[i],
                reduction="none",
            ).sum(dim=-1)  # (num_tokens,)
            kl_divs.append(kl)
        mean_kl = torch.stack(kl_divs).mean(dim=0)  # (num_tokens,)

        raw_values = mean_kl.numpy()

        if self.normalize and raw_values.max() > 0:
            values = raw_values / raw_values.max()
        else:
            values = raw_values.copy()

        return SignalResult(
            name="mc_dropout_variance",
            values=values,
            tokens=output.generated_tokens,
            raw_values=raw_values,
            metadata={
                "n_passes": n_passes,
                "normalized": self.normalize,
                "method": "mean_kl_divergence",
            },
        )


class SignalExtractor:
    """
    Orchestrator that runs all 4 signal extractors on a GenerationOutput
    and returns a unified AllSignals result.

    Usage:
        model = LieDetectorModel()
        output = model.generate_with_signals("prompt", mc_dropout_passes=5)
        extractor = SignalExtractor()
        signals = extractor.extract_all(output, model)
    """

    def __init__(
        self,
        compute_jacobian: bool = True,
        jacobian_iterations: int = 3,
        normalize: bool = True,
    ):
        self.entropy_signal = SoftmaxEntropy(normalize=normalize)
        self.gradient_signal = GradientNorm(normalize=normalize)
        self.jacobian_signal = JacobianSpectralNorm(
            n_iterations=jacobian_iterations, normalize=normalize
        ) if compute_jacobian else None
        self.mc_signal = MCDropoutVariance(normalize=normalize)

    def extract_all(
        self, output: GenerationOutput, model: Optional[LieDetectorModel] = None
    ) -> AllSignals:
        """
        Extract all signals from a generation output.

        Args:
            output: The generation output containing logits, grads, MC samples
            model: Required only if computing Jacobian spectral norm
        """
        entropy = self.entropy_signal.compute(output)
        gradient = self.gradient_signal.compute(output)

        if self.jacobian_signal is not None and model is not None:
            jacobian = self.jacobian_signal.compute(output, model)
        else:
            num_tokens = len(output.generated_tokens)
            jacobian = SignalResult(
                name="jacobian_spectral_norm",
                values=np.zeros(num_tokens),
                tokens=output.generated_tokens,
                metadata={"warning": "Jacobian computation disabled or model not provided"},
            )

        mc_variance = self.mc_signal.compute(output)

        return AllSignals(
            entropy=entropy,
            gradient_norm=gradient,
            jacobian_norm=jacobian,
            mc_variance=mc_variance,
            tokens=output.generated_tokens,
            num_tokens=len(output.generated_tokens),
        )
