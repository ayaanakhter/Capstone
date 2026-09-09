"""
LieDetectorModel — Universal HF wrapper with hook instrumentation for uncertainty signal extraction.

Loads Causal LMs from HuggingFace and provides methods for:
- Text generation with full internal state capture
- Embedding-level gradient computation using inputs_embeds
- MC Dropout inference with active dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from typing import Optional
from dataclasses import dataclass, field
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
    """Token signal with UGD status — accepted / corrected / retracted."""
    token: str                  # The final emitted token (corrected version if applicable)
    original_token: str         # The first-sampled token (before correction)
    status: str                 # "accepted" | "corrected" | "retracted"
    confidence: float
    entropy: float
    gradient_norm: float
    mc_variance: float
    risk_score: float           # Pre-computed 0–1 hallucination risk
    corrected_risk: float       # Risk of the replacement token (0 if not corrected)

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

        # Load model and tokenizer (supports GPT-2, Llama, Qwen, etc)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model in default float32 for native CPU math speed
        # bfloat16 causes a 10x-100x slowdown on CPUs without native hardware support
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
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": "Answer in 1-3 sentences maximum. Be direct and to the point. No long explanations."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
        current_ids = inputs["input_ids"]

        for step in range(max_new_tokens):
            # 1. Forward Pass & Gradients
            step_logits, step_grads = self._forward_with_grads(current_ids, compute_grads=compute_grads)
            last_logits = step_logits[:, -1, :].float()  # convert to float32 for softmax stability
            last_probs = F.softmax(last_logits / temperature, dim=-1)
            
            # Confidence
            confidence = last_probs.max().item()

            # Entropy
            log_probs = torch.log(last_probs + 1e-10)
            entropy = -(last_probs * log_probs).sum(dim=-1).item()

            # Gradient Norm
            grad_norm = 0.0
            if compute_grads and step_grads is not None:
                grad_tensor = step_grads.squeeze(0) if step_grads.dim() == 3 else step_grads
                grad_norm = torch.norm(grad_tensor, p=2).item()

            # MC Dropout
            mc_variance = 0.0
            if mc_dropout_passes > 0:
                mc_logits_step = self._mc_dropout_forward(current_ids, n_passes=mc_dropout_passes)
                stacked = torch.stack(mc_logits_step, dim=0).float()
                stacked_probs = F.softmax(stacked, dim=-1)
                max_probs_per_pass = stacked_probs.max(dim=-1).values
                mc_variance = max_probs_per_pass.var(dim=0).item()

            # Sample Token
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
                mc_variance=mc_variance
            )

    def _forward_with_grads(self, input_ids: torch.Tensor, compute_grads: bool = True) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
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

        last_logits = logits[:, -1, :]
        max_logit = last_logits.max()
        max_logit.backward(retain_graph=False)

        grads = embeddings.grad.detach().cpu() if hasattr(embeddings, 'grad') and embeddings.grad is not None else None
        detached_logits = logits.detach().cpu()

        del outputs
        del embeddings
        del logits
        del max_logit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return detached_logits, grads

    def _mc_dropout_forward(self, input_ids: torch.Tensor, n_passes: int = 5) -> list[torch.Tensor]:
        mc_logits = []
        self.model.train()
        with torch.no_grad():
            for _ in range(n_passes):
                outputs = self.model(input_ids)
                last_logits = outputs.logits[:, -1, :]
                mc_logits.append(last_logits.cpu())
                del outputs
        self.model.eval()
        return mc_logits

    def _sample_token(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95) -> torch.Tensor:
        logits = logits / temperature
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_values, _ = torch.topk(logits, top_k)
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_top_k, torch.full_like(logits, float("-inf")), logits)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.squeeze()

    def _compute_risk(self, confidence: float, entropy: float, grad_norm: float, mc_variance: float) -> float:
        """Compute 0–1 hallucination risk score from raw signals."""
        norm_e = min(entropy / 10.0, 1.0)
        norm_g = min(grad_norm / 10.0, 1.0)
        norm_m = min(mc_variance / 1.0, 1.0)
        raw = 0.15 * norm_e + 0.30 * norm_g + 0.25 * norm_m
        x = 5.0 * (confidence * raw - 0.3)
        return float(1 / (1 + np.exp(-x)))

    def generate_stream_ugd(
        self,
        prompt: str,
        max_new_tokens: int = 40,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        mc_dropout_passes: int = 2,
        warn_threshold: float = 0.40,    # Re-sample if risk exceeds this
        retract_threshold: float = 0.65, # Retract entirely if risk still exceeds this after re-sample
    ):
        """
        Uncertainty-Gated Decoding (UGD):
        Instead of blindly emitting every token, we gate each token through its
        live hallucination risk score and take corrective action:
          - risk < warn_threshold   → ✅ Accept and emit normally
          - warn_threshold ≤ risk < retract_threshold → ⚠️ Re-sample with higher
              temperature. If replacement is safer, emit it as "corrected".
          - risk ≥ retract_threshold → 🛑 Retract. Emit a RETRACTED event and stop.
        """
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": "Answer in 1-3 sentences maximum. Be direct and to the point. No long explanations."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        current_ids = inputs["input_ids"]

        for step in range(max_new_tokens):
            # ── Step 1: Forward pass with gradients ──────────────────────────
            step_logits, step_grads = self._forward_with_grads(current_ids, compute_grads=True)
            last_logits = step_logits[:, -1, :].float()
            last_probs  = F.softmax(last_logits / temperature, dim=-1)

            confidence = last_probs.max().item()
            log_probs  = torch.log(last_probs + 1e-10)
            entropy    = -(last_probs * log_probs).sum(dim=-1).item()

            grad_norm = 0.0
            if step_grads is not None:
                gt = step_grads.squeeze(0) if step_grads.dim() == 3 else step_grads
                grad_norm = torch.norm(gt, p=2).item()

            mc_variance = 0.0
            if mc_dropout_passes > 0:
                mc_list = self._mc_dropout_forward(current_ids, n_passes=mc_dropout_passes)
                stacked = torch.stack(mc_list, dim=0).float()
                stacked_probs = F.softmax(stacked, dim=-1)
                mc_variance = stacked_probs.max(dim=-1).values.var(dim=0).item()

            # ── Step 2: Risk score ────────────────────────────────────────────
            risk = self._compute_risk(confidence, entropy, grad_norm, mc_variance)

            # ── Step 3: Sample the original token ────────────────────────────
            original_token_id = self._sample_token(last_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            if original_token_id.item() == self.tokenizer.eos_token_id:
                break
            original_token_str = self.tokenizer.decode([original_token_id.item()], skip_special_tokens=True)

            # ── Step 4: Gate ─────────────────────────────────────────────────
            if risk < warn_threshold:
                # ✅ Accept — low risk, emit as-is
                current_ids = torch.cat([current_ids, original_token_id.unsqueeze(0).unsqueeze(0).to(self.device)], dim=1)
                yield UGDTokenSignal(
                    token=original_token_str, original_token=original_token_str,
                    status="accepted",
                    confidence=confidence, entropy=entropy,
                    gradient_norm=grad_norm, mc_variance=mc_variance,
                    risk_score=risk, corrected_risk=0.0,
                )

            elif risk < retract_threshold:
                # ⚠️ Re-sample with higher temperature — explore safer alternative
                alt_logits_raw = step_logits[:, -1, :].float()
                alt_token_id = self._sample_token(alt_logits_raw, temperature=temperature * 1.8, top_k=top_k, top_p=top_p)
                alt_token_str = self.tokenizer.decode([alt_token_id.item()], skip_special_tokens=True)

                # Quick risk estimate for alt token (no grad recompute — use same signals, diff confidence)
                alt_probs = F.softmax(alt_logits_raw / (temperature * 1.8), dim=-1)
                alt_conf  = alt_probs.max().item()
                alt_risk  = self._compute_risk(alt_conf, entropy, grad_norm, mc_variance)

                # Pick whichever is safer
                if alt_risk < risk and alt_token_id.item() != self.tokenizer.eos_token_id:
                    chosen_id  = alt_token_id
                    chosen_str = alt_token_str
                    status     = "corrected"
                    final_risk = alt_risk
                else:
                    chosen_id  = original_token_id
                    chosen_str = original_token_str
                    status     = "accepted"
                    final_risk = risk

                current_ids = torch.cat([current_ids, chosen_id.unsqueeze(0).unsqueeze(0).to(self.device)], dim=1)
                yield UGDTokenSignal(
                    token=chosen_str, original_token=original_token_str,
                    status=status,
                    confidence=confidence, entropy=entropy,
                    gradient_norm=grad_norm, mc_variance=mc_variance,
                    risk_score=risk, corrected_risk=final_risk,
                )

            else:
                # 🛑 Risk too high — retract and stop generation
                yield UGDTokenSignal(
                    token="", original_token=original_token_str,
                    status="retracted",
                    confidence=confidence, entropy=entropy,
                    gradient_norm=grad_norm, mc_variance=mc_variance,
                    risk_score=risk, corrected_risk=0.0,
                )
                break  # Stop generation entirely — model is hallucinating
