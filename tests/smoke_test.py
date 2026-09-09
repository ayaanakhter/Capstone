"""
Smoke test: Run the full Lie Detector pipeline on 5 hand-picked prompts
with actual GPT-2 to verify everything works end-to-end.

Prompts are chosen to cover different scenarios:
1. Factual question (likely correct) — should have LOW lie score
2. Obscure factual question (likely hallucination) — should have HIGH lie score
3. Continuation of well-known text — should have LOW lie score
4. Nonsense/adversarial prompt — should have HIGH signals
5. Ambiguous question — should show moderate uncertainty
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detector.model import LieDetectorModel
from src.detector.signals import SignalExtractor
from src.detector.lie_score import LieScoreAggregator, LieScoreWeights


PROMPTS = [
    {
        "text": "The capital of France is",
        "label": "Easy factual — expect low lie score",
        "expected": "low",
    },
    {
        "text": "The 17th president of the small country of Tuvalu was",
        "label": "Obscure factual — expect high lie score (hallucination likely)",
        "expected": "high",
    },
    {
        "text": "To be, or not to be, that is the",
        "label": "Well-known text continuation — expect low lie score",
        "expected": "low",
    },
    {
        "text": "Flurbo gnarx the zibbit when splonk",
        "label": "Nonsense/adversarial — expect high uncertainty signals",
        "expected": "high",
    },
    {
        "text": "The best programming language is",
        "label": "Subjective/ambiguous — expect moderate uncertainty",
        "expected": "moderate",
    },
]


def run_smoke_test():
    print("=" * 70)
    print("  THE LIE DETECTOR — Smoke Test")
    print("=" * 70)

    # Load model
    print("\n[1/3] Loading GPT-2...")
    t0 = time.time()
    model = LieDetectorModel(model_name="./GPT_2")
    load_time = time.time() - t0
    info = model.get_model_info()
    print(f"  Model: {info['model_name']} ({info['total_params_human']} params)")
    print(f"  Device: {info['device']}")
    print(f"  Load time: {load_time:.1f}s")

    # Set up signal extraction (skip Jacobian for speed in smoke test)
    extractor = SignalExtractor(compute_jacobian=False, normalize=True)
    aggregator = LieScoreAggregator(threshold=0.5)

    print("\n[2/3] Running pipeline on 5 prompts...")
    print("-" * 70)

    results_table = []

    for i, prompt_info in enumerate(PROMPTS):
        prompt = prompt_info["text"]
        label = prompt_info["label"]
        expected = prompt_info["expected"]

        print(f"\n  Prompt {i+1}: \"{prompt}\"")
        print(f"  Label: {label}")

        t0 = time.time()
        output = model.generate_with_signals(
            prompt,
            max_new_tokens=20,
            temperature=0.7,
            compute_grads=True,
            mc_dropout_passes=5,
        )
        gen_time = time.time() - t0

        # Extract signals
        signals = extractor.extract_all(output, model=None)

        # Compute lie scores
        confidence = output.softmax_probs_sequence.numpy().max(axis=-1)
        lie_result = aggregator.compute(signals, confidence)

        # Print results
        print(f"  Generated: \"{output.generated_text[:80]}{'...' if len(output.generated_text) > 80 else ''}\"")
        print(f"  Tokens generated: {len(output.generated_tokens)}")
        print(f"  Generation time: {gen_time:.2f}s")
        print(f"  --- Signals ---")
        print(f"    Entropy (mean):     {signals.entropy.values.mean():.4f}")
        print(f"    Gradient Norm (mean):{signals.gradient_norm.values.mean():.4f}")
        print(f"    MC Variance (mean): {signals.mc_variance.values.mean():.4f}")
        print(f"  --- Lie Score ---")
        print(f"    Overall Lie Score:  {lie_result.overall_lie_score:.4f}")
        print(f"    Mean Confidence:    {confidence.mean():.4f}")
        print(f"    Tokens flagged:     {lie_result.num_flagged}/{len(lie_result.tokens)}")

        if lie_result.num_flagged > 0:
            flagged = lie_result.flagged_tokens[:5]  # Show up to 5
            print(f"    Flagged tokens:     {[(t, f'{s:.3f}') for _, t, s in flagged]}")

        # Token-level detail (first 10 tokens)
        print(f"  --- Per-Token Detail (first 10) ---")
        n_show = min(10, len(output.generated_tokens))
        print(f"    {'Token':<15} {'Conf':>6} {'Entropy':>8} {'GradN':>8} {'MCV':>8} {'Lie':>6} {'Flag':>5}")
        for j in range(n_show):
            tok = output.generated_tokens[j][:12]
            conf = confidence[j]
            ent = signals.entropy.values[j]
            gn = signals.gradient_norm.values[j]
            mc = signals.mc_variance.values[j]
            lie = lie_result.scores[j]
            flag = "FLAG" if lie_result.is_flagged[j] else "OK"
            print(f"    {tok:<15} {conf:>6.3f} {ent:>8.4f} {gn:>8.4f} {mc:>8.4f} {lie:>6.3f} {flag:>5}")

        results_table.append({
            "prompt": prompt[:30],
            "expected": expected,
            "lie_score": lie_result.overall_lie_score,
            "confidence": confidence.mean(),
            "flagged": lie_result.num_flagged,
            "total": len(lie_result.tokens),
        })

        print("-" * 70)

    # Summary table
    print("\n[3/3] Summary")
    print("=" * 70)
    print(f"  {'Prompt':<32} {'Expected':<10} {'LieScore':>9} {'Conf':>6} {'Flagged':>8}")
    print("-" * 70)
    for r in results_table:
        print(f"  {r['prompt']:<32} {r['expected']:<10} {r['lie_score']:>9.4f} {r['confidence']:>6.3f} {r['flagged']:>3}/{r['total']:<4}")

    print("\n✅ Smoke test complete — pipeline runs end-to-end on real GPT-2!")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
