#!/usr/bin/env python3
"""
Adaptive-λ (PID-Lagrangian) baseline analysis on Cancer.

Compares hard RA filter vs fixed-λ Lagrangian vs per-patient adaptive-λ.
Uses cached Phase 1 trajectory features — no GPU needed.
"""

import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent / 'phase1_ra_v2' / 'my_outputs' / 'cancer_sim_cont' / '22'
TARGET_UPPER = 0.6
SAFETY_VOL_UPPER = 5.0
SAFETY_CHEMO_UPPER = 8.5
SEEDS = [10, 101, 1010, 10101, 101010]
TAUS = [2, 4, 6, 8]


def load_data(gamma):
    all_cases = {}
    for seed in SEEDS:
        path = BASE_DIR / f'coeff_{gamma}' / 'VCIP' / 'train' / 'True' / 'case_infos' / str(seed) / 'False' / 'case_infos_VCIP.pkl'
        if not path.exists():
            continue
        with open(path, 'rb') as f:
            data = pickle.load(f)
        all_cases[seed] = data['VCIP']
    return all_cases


def is_feasible(cv_t, cv_m, cd_m):
    return (cv_t <= TARGET_UPPER) & (cv_m <= SAFETY_VOL_UPPER) & (cd_m <= SAFETY_CHEMO_UPPER)


def evaluate_method(cases_list, method_fn):
    """Evaluate a selection method across all patients. Returns (safety, intarget, top1)."""
    safety, intarget, top1 = [], [], []
    for c in cases_list:
        elbos = c['model_losses']
        true_losses = c['true_losses']
        tf = c['traj_features']
        feasible = is_feasible(tf['cv_terminal'], tf['cv_max'], tf['cd_max'])
        oracle_best = int(np.argmin(true_losses))

        idx = method_fn(elbos, feasible)
        safety.append(bool(feasible[idx]))
        intarget.append(bool(tf['cv_terminal'][idx] <= TARGET_UPPER))
        top1.append(1.0 if idx == oracle_best else 0.0)
    return np.mean(safety), np.mean(intarget), np.mean(top1)


def elbo_select(elbos, feasible):
    return int(np.argmin(elbos))


def hard_filter(elbos, feasible):
    if feasible.sum() == 0:
        return int(np.argmin(elbos))
    masked = elbos.copy()
    masked[~feasible] = np.inf
    return int(np.argmin(masked))


def make_soft_select(lam):
    def soft_select(elbos, feasible):
        penalty = lam * (1.0 - feasible.astype(float))
        return int(np.argmin(elbos + penalty))
    return soft_select


def pid_adaptive(elbos, feasible):
    """Per-patient adaptive-λ: binary search for min λ that selects feasible."""
    if feasible.sum() == 0:
        return int(np.argmin(elbos))
    # If ELBO-best is already feasible, use it (λ=0 suffices)
    elbo_best = int(np.argmin(elbos))
    if feasible[elbo_best]:
        return elbo_best
    # Binary search for min λ
    lo, hi = 0.0, 1000.0
    for _ in range(50):
        mid = (lo + hi) / 2
        penalty = mid * (1.0 - feasible.astype(float))
        idx = int(np.argmin(elbos + penalty))
        if feasible[idx]:
            hi = mid
        else:
            lo = mid
    penalty = hi * (1.0 - feasible.astype(float))
    return int(np.argmin(elbos + penalty))


def main():
    gamma = 4
    data = load_data(gamma)
    print(f"Loaded {len(data)} seeds for gamma={gamma}")

    lambdas_to_show = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

    print(f"\n{'='*95}")
    print(f"SAFETY–QUALITY COMPARISON: Hard Filter vs Soft λ vs PID-Adaptive (Cancer, γ={gamma})")
    print(f"{'='*95}")

    for tau in TAUS:
        # Pool all patients across seeds
        cases = []
        for seed in sorted(data.keys()):
            if tau in data[seed]:
                cases.extend(data[seed][tau])

        if not cases:
            continue

        print(f"\n--- tau={tau} (n={len(cases)}) ---")
        print(f"{'Method':<30} | {'Safety':>8} | {'In-Tgt':>8} | {'Top-1':>8}")
        print("-" * 65)

        # ELBO unconstrained
        sr, it, t1 = evaluate_method(cases, elbo_select)
        print(f"{'ELBO (unconstrained)':<30} | {sr:>7.1%} | {it:>7.1%} | {t1:>7.1%}")
        elbo_sr = sr

        # Soft constraint sweep
        for lam in lambdas_to_show:
            sr, it, t1 = evaluate_method(cases, make_soft_select(lam))
            print(f"{f'Soft λ={lam}':<30} | {sr:>7.1%} | {it:>7.1%} | {t1:>7.1%}")

        # PID-adaptive
        sr_pid, it_pid, t1_pid = evaluate_method(cases, pid_adaptive)
        print(f"{'PID-Lagrangian (adaptive)':<30} | {sr_pid:>7.1%} | {it_pid:>7.1%} | {t1_pid:>7.1%}")

        # Hard filter
        sr_hard, it_hard, t1_hard = evaluate_method(cases, hard_filter)
        print(f"{'Hard RA filter':<30} | {sr_hard:>7.1%} | {it_hard:>7.1%} | {t1_hard:>7.1%}")

        # Compare
        print(f"\n  PID vs Hard: Safety {sr_pid:.1%} vs {sr_hard:.1%}, "
              f"Top-1 {t1_pid:.1%} vs {t1_hard:.1%}")
        if sr_hard >= sr_pid and t1_hard >= t1_pid:
            print(f"  → Hard filter DOMINATES PID-Lagrangian")
        elif sr_hard >= sr_pid:
            print(f"  → Hard filter safer, PID has +{t1_pid - t1_hard:.1%} Top-1")
        else:
            print(f"  → PID safer (+{sr_pid - sr_hard:.1%}), hard has +{t1_hard - t1_pid:.1%} Top-1")

    print(f"\n{'='*95}")
    print("DONE")


if __name__ == '__main__':
    main()
