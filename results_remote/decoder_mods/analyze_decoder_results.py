#!/usr/bin/env python3
"""
Analyze decoder modification experiment results.

Compares vanilla, heteroscedastic, wider, and MC-dropout decoder variants
on model-predicted RA filtering for Cancer (gamma=4).

Usage:
    python3 results_remote/decoder_mods/analyze_decoder_results.py
"""

import pickle
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent
VARIANTS = ['vanilla', 'heteroscedastic', 'wider', 'mcdropout']
TAUS = [2, 4, 6, 8]
GAMMA = 4


def load_variant(variant):
    path = RESULTS_DIR / f'{variant}_gamma{GAMMA}.pkl'
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def summarize_variant(name, data):
    """Print summary table for one variant."""
    if data is None:
        print(f"  {name}: NOT AVAILABLE")
        return {}

    seeds = list(data.keys())
    summary = {}

    for tau in TAUS:
        metrics = {
            'elbo_top1': [], 'orac_feas': [], 'modl_feas': [],
            'orac_top1': [], 'modl_top1': [],
            'modl_safe': [], 'modl_intgt': [],
        }
        for seed in seeds:
            if tau not in data[seed]:
                continue
            for c in data[seed][tau]:
                metrics['elbo_top1'].append(c['oracle_metrics']['elbo_top1'])
                metrics['orac_feas'].append(c['oracle_feas'])
                metrics['modl_feas'].append(c['model_feas'])
                metrics['orac_top1'].append(c['oracle_metrics']['cstr_top1'])
                metrics['modl_top1'].append(c['model_metrics']['cstr_top1'])
                metrics['modl_safe'].append(c['model_truly_safe'])
                metrics['modl_intgt'].append(c['model_truly_intarget'])

        summary[tau] = {k: np.mean(v) for k, v in metrics.items()}

    return summary


def main():
    print("=" * 90)
    print("DECODER MODIFICATION EXPERIMENT RESULTS")
    print(f"Cancer, gamma={GAMMA}, 5 seeds × 4 taus × 100 patients × 100 candidates")
    print("=" * 90)

    all_summaries = {}
    for variant in VARIANTS:
        data = load_variant(variant)
        all_summaries[variant] = summarize_variant(variant, data)

    # Print comparison table
    print("\n" + "=" * 90)
    print("COMPARISON TABLE: Model Feasibility (key metric)")
    print("=" * 90)
    print(f"{'tau':>4}", end="")
    for v in VARIANTS:
        print(f" | {v:>15}", end="")
    print(" | {'oracle':>15}")
    print("-" * 90)

    for tau in TAUS:
        print(f"{tau:>4}", end="")
        for v in VARIANTS:
            s = all_summaries[v]
            if s and tau in s:
                print(f" | {s[tau]['modl_feas']:>14.1%}", end="")
            else:
                print(f" | {'N/A':>15}", end="")
        # Oracle feasibility (from vanilla, same for all)
        if all_summaries['vanilla'] and tau in all_summaries['vanilla']:
            print(f" | {all_summaries['vanilla'][tau]['orac_feas']:>14.1%}")
        else:
            print(f" | {'N/A':>15}")

    print("\n" + "=" * 90)
    print("COMPARISON TABLE: Model-Predicted Truly Safe (oracle-verified)")
    print("=" * 90)
    print(f"{'tau':>4}", end="")
    for v in VARIANTS:
        print(f" | {v:>15}", end="")
    print()
    print("-" * 80)

    for tau in TAUS:
        print(f"{tau:>4}", end="")
        for v in VARIANTS:
            s = all_summaries[v]
            if s and tau in s:
                print(f" | {s[tau]['modl_safe']:>14.1%}", end="")
            else:
                print(f" | {'N/A':>15}", end="")
        print()

    print("\n" + "=" * 90)
    print("COMPARISON TABLE: Model-Predicted Constrained Top-1")
    print("=" * 90)
    print(f"{'tau':>4}", end="")
    for v in VARIANTS:
        print(f" | {v:>15}", end="")
    print(f" | {'ELBO (uncstr)':>15} | {'Oracle RA':>15}")
    print("-" * 110)

    for tau in TAUS:
        print(f"{tau:>4}", end="")
        for v in VARIANTS:
            s = all_summaries[v]
            if s and tau in s:
                print(f" | {s[tau]['modl_top1']:>14.1%}", end="")
            else:
                print(f" | {'N/A':>15}", end="")
        if all_summaries['vanilla'] and tau in all_summaries['vanilla']:
            sv = all_summaries['vanilla'][tau]
            print(f" | {sv['elbo_top1']:>14.1%} | {sv['orac_top1']:>14.1%}")
        else:
            print()

    # Diagnosis
    print("\n" + "=" * 90)
    print("DIAGNOSIS")
    print("=" * 90)

    # Check if any variant achieved >5% model feasibility
    any_success = False
    for v in VARIANTS:
        s = all_summaries[v]
        if not s:
            continue
        for tau in TAUS:
            if tau in s and s[tau]['modl_feas'] > 0.05:
                any_success = True
                print(f"  {v} at tau={tau}: model feasibility = {s[tau]['modl_feas']:.1%} (> 5%)")

    if not any_success:
        print("  NO variant achieves >5% model feasibility at any tau.")
        print()
        print("  Root cause: The VCIP decoder produces near-constant cancer volume")
        print("  predictions regardless of treatment sequence. The latent space is")
        print("  action-invariant (KL ≈ 10⁻⁵ from VCI diagnostic), so the decoder")
        print("  receives no discriminative signal. This is upstream of the decoder")
        print("  head — no loss function or capacity change can fix it.")
        print()
        print("  RECOMMENDATION: Reframe the paper to present the oracle/model gap")
        print("  as a first-class diagnostic finding. The RA filter works when")
        print("  trajectory predictions are reliable (oracle) but current variational")
        print("  counterfactual planners do not provide this reliability.")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == '__main__':
    main()
