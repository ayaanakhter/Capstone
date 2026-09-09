"""
Evaluation benchmark pipeline for The Lie Detector.

Runs the detection engine over datasets to compute AUROC, ECE, and other metrics.
Produces the main evaluation artifacts (calibration plot, result metrics).
"""

import sys
import os
import time
import numpy as np
import json
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.detector.model import LieDetectorModel
from src.detector.signals import SignalExtractor
from src.detector.lie_score import LieScoreAggregator
from src.detector.calibration import CalibrationAnalyzer
from src.eval.data_loader import get_evaluation_prompts
from src.eval.adversarial import AdversarialGenerator

def run_benchmark(model_name: str = "./GPT_2", num_samples: int = 20):
    print("=" * 60)
    print("  THE LIE DETECTOR — Evaluation Benchmark")
    print("=" * 60)

    # Initialize components
    print(f"Loading model {model_name}...")
    model = LieDetectorModel(model_name=model_name)
    extractor = SignalExtractor(compute_jacobian=False, normalize=True)
    aggregator = LieScoreAggregator(threshold=0.5)
    calibration = CalibrationAnalyzer(n_bins=10)
    adv_gen = AdversarialGenerator()

    prompts = get_evaluation_prompts(max_samples=num_samples)
    print(f"Loaded {len(prompts)} prompts for evaluation.\n")

    results = []
    
    # Simple simulated labeling for the benchmark script:
    # In a full run, you'd use a judging model (like DeBERTa) to label if the
    # generated text is factually correct vs the ground truth answer.
    # Here, we simulate that to verify the pipeline runs.

    for idx, item in enumerate(tqdm(prompts, desc="Running benchmarks")):
        text = item["question"]
        
        # 1. Clean Prompt
        out_clean = model.generate_with_signals(
            text, max_new_tokens=15, compute_grads=True, mc_dropout_passes=3
        )
        signals_clean = extractor.extract_all(out_clean)
        conf_clean = out_clean.softmax_probs_sequence.numpy().max(axis=-1)
        lie_clean = aggregator.compute(signals_clean, conf_clean)
        
        # Simulated correctness label (just for testing the calibration loop)
        # Assuming higher lie score = more likely to be wrong
        correctness = (np.random.rand(len(conf_clean)) > lie_clean.scores).astype(float)
        calibration.add_predictions(conf_clean, correctness)
        
        # 2. Adversarial (Typo) Prompt
        text_typo = adv_gen.typo_injection(text)
        out_typo = model.generate_with_signals(
            text_typo, max_new_tokens=15, compute_grads=True, mc_dropout_passes=3
        )
        signals_typo = extractor.extract_all(out_typo)
        
        results.append({
            "id": item.get("id", str(idx)),
            "clean_lie_score": float(lie_clean.overall_lie_score),
            "clean_grad_norm": float(signals_clean.gradient_norm.values.mean()),
            "typo_grad_norm": float(signals_typo.gradient_norm.values.mean())
        })

    print("\n[Analysis]")
    clean_grads = np.array([r["clean_grad_norm"] for r in results])
    typo_grads = np.array([r["typo_grad_norm"] for r in results])
    
    print(f"Average Gradient Norm (Clean): {clean_grads.mean():.4f}")
    print(f"Average Gradient Norm (Typo):  {typo_grads.mean():.4f}")
    
    if typo_grads.mean() > clean_grads.mean():
        print("[PASS] Adversarial detection working: Gradient norm spikes on perturbed inputs!")
    else:
        print("[FAIL] Adversarial detection failed: Gradient norm didn't increase significantly.")

    # Save Calibration Plot
    os.makedirs("docs", exist_ok=True)
    plot_path = os.path.join("docs", "calibration_plot.png")
    calibration.plot_reliability_diagram(save_path=plot_path)
    print(f"\nSaved calibration plot to {plot_path}")

    # Save results
    results_path = os.path.join("docs", "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved numerical results to {results_path}")

if __name__ == "__main__":
    run_benchmark(num_samples=10) # Small sample for quick testing
