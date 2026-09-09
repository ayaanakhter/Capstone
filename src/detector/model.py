"""
LieDetectorModel — Universal HF wrapper with hook instrumentation for uncertainty signal extraction.

Loads Causal LMs from HuggingFace and provides methods for:
- Text generation with full internal state capture
- Embedding-level gradient computation using inputs_embeds
- MC Dropout inference with active dropout
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class TokenSignal:
    token: str
    confidence: float
    entropy: float
    gradient_norm: float
    mc_variance: float


@dataclass
class GenerationOutput:
    # Dummy class to preserve backwards compatibility for imports in signals.py
    pass


@dataclass
class UGDTokenSignal:
    """Token signal with UGD gate status."""
    token: str           # Emitted token (empty string if retracted)
    original_token: str  # Token the model originally wanted to emit
    status: str          # "accepted" | "warned" | "retracted"
    confidence: float
    entropy: float
    gradient_norm: float
    mc_variance: float
    risk_score: float    # 0–1 hallucination risk
    corrected_risk: float


class LieDetectorModel:
    """
    Model wrapper that enables deep introspection of internal states
    during generation, for the purpose of detecting overconfidence and hallucination.
    """

    def __init__(
        self,
        model_name: str = "./QWEN",
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading {model_name} in float32 precision for maximum speed...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.config: PretrainedConfig = self.model.config
        self.vocab_size = self.config.vocab_size
        self.embed_dim = getattr(self.config, 'hidden_size', getattr(self.config, 'n_embd', 0))
        self.n_layers = getattr(self.config, 'num_hidden_layers', getattr(self.config, 'n_layer', 0))
        self.n_heads = getattr(self.config, 'num_attention_heads', getattr(self.config, 'n_head', 0))

    # ─────────────────────────────────────────────
    # Standard streaming (no safety gate)
    # ─────────────────────────────────────────────
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 40,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        compute_grads: bool = True,
        mc_dropout_passes: int = 0,
    ):
        inputs = self._build_inputs(prompt)
        current_ids = inputs["input_ids"]

        for _ in range(max_new_tokens):
            step_logits, step_grads = self._forward_with_grads(current_ids, compute_grads=compute_grads)
            last_logits = step_logits[:, -1, :].float()
            last_probs = F.softmax(last_logits / temperature, dim=-1)

            confidence = last_probs.max().item()
            entropy = -(last_probs * torch.log(last_probs + 1e-10)).sum(dim=-1).item()

            grad_norm = 0.0
            if compute_grads and step_grads is not None:
                gt = step_grads.squeeze(0) if step_grads.dim() == 3 else step_grads
                grad_norm = torch.norm(gt, p=2).item()

            mc_variance = 0.0
            if mc_dropout_passes > 0:
                mc_list = self._mc_dropout_forward(current_ids, n_passes=mc_dropout_passes)
                stacked = torch.stack(mc_list, dim=0).float()
                mc_variance = F.softmax(stacked, dim=-1).max(dim=-1).values.var(dim=0).item()

            next_token_id = self._sample_token(last_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            token_str = self.tokenizer.decode([next_token_id.item()], skip_special_tokens=True)
            current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0).unsqueeze(0).to(self.device)], dim=1)

            yield TokenSignal(
                token=token_str,
                confidence=confidence,
                entropy=entropy,
                gradient_norm=grad_norm,
                mc_variance=mc_variance,
            )

    # ─────────────────────────────────────────────
    # UGD: Uncertainty-Gated Decoding
    # ─────────────────────────────────────────────
    def generate_stream_ugd(
        self,
        prompt: str,
        max_new_tokens: int = 40,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        mc_dropout_passes: int = 2,
        warn_threshold: float = 0.45,     # Highlight token orange but keep it
        retract_threshold: float = 0.70,  # Stop generation entirely
    ):
        """
        Uncertainty-Gated Decoding (UGD) — 3-tier system:
          - risk < warn_threshold                 → ✅ accepted  (white, emit normally)
          - warn_threshold ≤ risk < retract       → ⚠️ warned    (orange, emit but flag)
          - risk ≥ retract_threshold              → 🛑 retracted (stop generation)
        """
        inputs = self._build_inputs(prompt)
        current_ids = inputs["input_ids"]

        for _ in range(max_new_tokens):
            step_logits, step_grads = self._forward_with_grads(current_ids, compute_grads=True)
            last_logits = step_logits[:, -1, :].float()
            last_probs  = F.softmax(last_logits / temperature, dim=-1)

            confidence  = last_probs.max().item()
            entropy     = -(last_probs * torch.log(last_probs + 1e-10)).sum(dim=-1).item()

            grad_norm = 0.0
            if step_grads is not None:
                gt = step_grads.squeeze(0) if step_grads.dim() == 3 else step_grads
                grad_norm = torch.norm(gt, p=2).item()

            mc_variance = 0.0
            if mc_dropout_passes > 0:
                mc_list = self._mc_dropout_forward(current_ids, n_passes=mc_dropout_passes)
                stacked = torch.stack(mc_list, dim=0).float()
                mc_variance = F.softmax(stacked, dim=-1).max(dim=-1).values.var(dim=0).item()

            risk = self._compute_risk(confidence, entropy, grad_norm, mc_variance)

            next_token_id = self._sample_token(last_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
            token_str = self.tokenizer.decode([next_token_id.item()], skip_special_tokens=True)

            if risk >= retract_threshold:
                # 🛑 Retract — stop generation
                yield UGDTokenSignal(
                    token="", original_token=token_str, status="retracted",
                    confidence=confidence, entropy=entropy,
                    gradient_norm=grad_norm, mc_variance=mc_variance,
                    risk_score=risk, corrected_risk=0.0,
                )
                break

            # ✅ Accept or ⚠️ Warn — emit the token either way
            current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0).unsqueeze(0).to(self.device)], dim=1)
            status = "warned" if risk >= warn_threshold else "accepted"
            yield UGDTokenSignal(
                token=token_str, original_token=token_str, status=status,
                confidence=confidence, entropy=entropy,
                gradient_norm=grad_norm, mc_variance=mc_variance,
                risk_score=risk, corrected_risk=0.0,
            )

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────
    def _build_inputs(self, prompt: str) -> dict:
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": "Reply in 1-2 short sentences only. Be extremely concise. No bullet points, no elaboration."},
                {"role": "user",   "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return self.tokenizer(text, return_tensors="pt").to(self.device)
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def _compute_risk(self, confidence: float, entropy: float, grad_norm: float, mc_variance: float) -> float:
        norm_e = min(entropy / 10.0, 1.0)
        norm_g = min(grad_norm / 10.0, 1.0)
        norm_m = min(mc_variance / 1.0, 1.0)
        raw = 0.15 * norm_e + 0.30 * norm_g + 0.25 * norm_m
        x = 5.0 * (confidence * raw - 0.3)
        return float(1 / (1 + np.exp(-x)))

    def _forward_with_grads(self, input_ids: torch.Tensor, compute_grads: bool = True):
        self.model.eval()
        if not compute_grads:
            with torch.no_grad():
                outputs = self.model(input_ids)
            return outputs.logits, None

        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()

        outputs = self.model(inputs_embeds=embeddings)
        logits = outputs.logits
        logits[:, -1, :].max().backward(retain_graph=False)

        grads = embeddings.grad.detach().cpu() if embeddings.grad is not None else None
        detached = logits.detach().cpu()

        del outputs, embeddings, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return detached, grads

    def _mc_dropout_forward(self, input_ids: torch.Tensor, n_passes: int = 5):
        mc_logits = []
        self.model.train()
        with torch.no_grad():
            for _ in range(n_passes):
                outputs = self.model(input_ids)
                mc_logits.append(outputs.logits[:, -1, :].cpu())
                del outputs
        self.model.eval()
        return mc_logits

    def _sample_token(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95):
        logits = logits / temperature
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kvals, _ = torch.topk(logits, top_k)
            logits = torch.where(logits < kvals[:, -1].unsqueeze(-1), torch.full_like(logits, float("-inf")), logits)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_logits[cum - F.softmax(sorted_logits, dim=-1) >= top_p] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze()
