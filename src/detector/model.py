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
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        compute_grads: bool = True,
        mc_dropout_passes: int = 0,
    ):
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
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
