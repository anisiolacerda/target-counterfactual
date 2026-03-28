"""
B4: k-expansion analysis for gamma=4, Cancer simulator.

Analyzes how increasing the candidate pool size k affects:
1. Feasibility rate (% of candidates satisfying RA constraints)
2. RA-constrained Top-1 quality (True Loss of RA-selected plan)
3. Safety rate (% of RA-selected plans satisfying all constraints)
4. Comparison with baseline k=100 results from Table 1
"""

import pickle
import numpy as np
import os
from collections import defaultdict

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'b4_results')
SEEDS = [10, 101, 1010, 10101, 101010]
TAUS = [2, 4, 6, 8]
K_VALUES = [100, 250, 500, 1000]

# RA thresholds (Cancer simulator)
TARGET_UPPER = 3.0       # cm³
SAFETY_VOLUME_UPPER = 12.0  # cm³
SAFETY_CHEMO_UPPER = 5.0    # chemo dosage

GAMMA = 4


def load_data(k):
    """Load pickle files for a given k value."""
    data = {}
    for seed in SEEDS:
        pkl_path = os.path.join(
            BASE_DIR, f'k{k}/coeff_{GAMMA}/seed_{seed}/case_infos/{seed}/False/case_infos_VCIP.pkl')
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    raw = pickle.load(f)
                # Standard VCIP format: {'VCIP': {tau: [case_infos]}}
                if 'VCIP' in raw:
                    data[seed] = raw['VCIP']
                else:
                    data[seed] = raw
            except Exception as e:
                print(f"    WARNING: corrupt pickle for k={k}, seed={seed}: {e}")
    return data


def compute_ra_metrics(case_infos, k):
    """Compute RA metrics for a list of case_infos."""
    n = len(case_infos)
    if n == 0:
        return None

    safety_rates = []
    in_target_rates = []
    feasibility_rates = []
    elbo_top1_losses = []
    ra_top1_losses = []
    true_ranks = []

    for case in case_infos:
        model_losses = case['model_losses']
        true_losses = case['true_losses']
        traj_feat = case.get('traj_features', None)

        actual_k = len(model_losses)

        # ELBO-best selection
        elbo_idx = np.argmin(model_losses)
        elbo_top1_losses.append(true_losses[elbo_idx])

        # True sequence rank
        true_ranks.append(case.get('true_sequence_rank',
                                    int(np.sum(model_losses < model_losses[-1]) + 1)))

        if traj_feat is not None:
            cv_terminal = traj_feat['cv_terminal']
            cv_max = traj_feat['cv_max']
            cd_max = traj_feat.get('cd_max', np.zeros(actual_k))

            # Feasibility: terminal in target AND trajectory safe
            feasible = (cv_terminal <= TARGET_UPPER)
            safe = (cv_max <= SAFETY_VOLUME_UPPER)
            if cd_max is not None and len(cd_max) == actual_k:
                safe &= (cd_max <= SAFETY_CHEMO_UPPER)

            feasible_and_safe = feasible & safe
            n_feasible = feasible_and_safe.sum()
            feasibility_rates.append(n_feasible / actual_k)

            if n_feasible > 0:
                # RA-constrained selection: best ELBO among feasible
                ra_losses = model_losses.copy()
                ra_losses[~feasible_and_safe] = np.inf
                ra_idx = np.argmin(ra_losses)
                ra_top1_losses.append(true_losses[ra_idx])

                # Safety of RA-selected plan
                ra_safe = (cv_max[ra_idx] <= SAFETY_VOLUME_UPPER)
                if cd_max is not None and len(cd_max) > ra_idx:
                    ra_safe &= (cd_max[ra_idx] <= SAFETY_CHEMO_UPPER)
                safety_rates.append(1.0 if ra_safe else 0.0)

                # In-target
                in_target_rates.append(1.0 if cv_terminal[ra_idx] <= TARGET_UPPER else 0.0)
            else:
                # Infeasible: fall back to ELBO-best
                ra_top1_losses.append(true_losses[elbo_idx])
                safety_rates.append(
                    1.0 if (cv_max[elbo_idx] <= SAFETY_VOLUME_UPPER and
                            (cd_max is None or len(cd_max) == 0 or cd_max[elbo_idx] <= SAFETY_CHEMO_UPPER))
                    else 0.0)
                in_target_rates.append(1.0 if cv_terminal[elbo_idx] <= TARGET_UPPER else 0.0)
        else:
            # No trajectory features — can't compute RA metrics
            ra_top1_losses.append(true_losses[elbo_idx])
            feasibility_rates.append(0.0)
            safety_rates.append(0.0)
            in_target_rates.append(0.0)

    return {
        'n': n,
        'avg_rank': np.mean(true_ranks),
        'elbo_top1': np.mean(elbo_top1_losses) if elbo_top1_losses else None,
        'ra_top1': np.mean(ra_top1_losses) if ra_top1_losses else None,
        'feasibility': np.mean(feasibility_rates) if feasibility_rates else None,
        'safety': np.mean(safety_rates) if safety_rates else None,
        'in_target': np.mean(in_target_rates) if in_target_rates else None,
    }


def main():
    print("B4: k-Expansion Analysis (Cancer, gamma=4)")
    print("=" * 80)
    print(f"RA thresholds: target_upper={TARGET_UPPER} cm³, "
          f"safety_vol={SAFETY_VOLUME_UPPER} cm³, safety_chemo={SAFETY_CHEMO_UPPER}")
    print(f"Seeds: {SEEDS}")
    print()

    # Check available data
    available = {}
    for k in K_VALUES:
        data = load_data(k)
        if data:
            available[k] = data
            seeds_found = list(data.keys())
            print(f"  k={k}: {len(seeds_found)} seeds loaded ({seeds_found})")
        else:
            print(f"  k={k}: NO DATA FOUND")

    # Per-tau analysis
    for tau in TAUS:
        print(f"\n{'='*80}")
        print(f"tau={tau}")
        print(f"{'='*80}")
        print(f"{'k':>6s}  {'Seeds':>5s}  {'AvgRank':>8s}  {'ELBO Top1':>10s}  "
              f"{'RA Top1':>8s}  {'Feas%':>6s}  {'Safety%':>8s}  {'InTgt%':>7s}")
        print("-" * 80)

        for k in K_VALUES:
            if k not in available:
                print(f"{k:>6d}  {'N/A':>5s}")
                continue

            data = available[k]
            all_metrics = []
            for seed, seed_data in data.items():
                if tau in seed_data:
                    m = compute_ra_metrics(seed_data[tau], k)
                    if m:
                        all_metrics.append(m)

            if not all_metrics:
                print(f"{k:>6d}  {'0':>5s}")
                continue

            n_seeds = len(all_metrics)
            avg_rank = np.mean([m['avg_rank'] for m in all_metrics])
            elbo_top1 = np.mean([m['elbo_top1'] for m in all_metrics if m['elbo_top1'] is not None])
            ra_top1 = np.mean([m['ra_top1'] for m in all_metrics if m['ra_top1'] is not None])
            feas = np.mean([m['feasibility'] for m in all_metrics if m['feasibility'] is not None])
            safety = np.mean([m['safety'] for m in all_metrics if m['safety'] is not None])
            in_tgt = np.mean([m['in_target'] for m in all_metrics if m['in_target'] is not None])

            print(f"{k:>6d}  {n_seeds:>5d}  {avg_rank:>8.2f}  {elbo_top1:>10.6f}  "
                  f"{ra_top1:>8.6f}  {100*feas:>5.1f}%  {100*safety:>7.1f}%  {100*in_tgt:>6.1f}%")

    # Aggregate summary across all taus
    print(f"\n{'='*80}")
    print("AGGREGATE SUMMARY (mean across tau)")
    print(f"{'='*80}")
    print(f"{'k':>6s}  {'Feas%':>6s}  {'Safety%':>8s}  {'InTgt%':>7s}  "
          f"{'RA/ELBO':>8s}  {'n_obs':>6s}")
    print("-" * 60)

    for k in K_VALUES:
        if k not in available:
            continue
        data = available[k]
        feas_all = []
        safety_all = []
        in_tgt_all = []
        ratio_all = []

        for tau in TAUS:
            for seed, seed_data in data.items():
                if tau in seed_data:
                    m = compute_ra_metrics(seed_data[tau], k)
                    if m and m['feasibility'] is not None:
                        feas_all.append(m['feasibility'])
                        safety_all.append(m['safety'])
                        in_tgt_all.append(m['in_target'])
                        if m['elbo_top1'] and m['elbo_top1'] > 0:
                            ratio_all.append(m['ra_top1'] / m['elbo_top1'])

        if feas_all:
            print(f"{k:>6d}  {100*np.mean(feas_all):>5.1f}%  {100*np.mean(safety_all):>7.1f}%  "
                  f"{100*np.mean(in_tgt_all):>6.1f}%  {np.mean(ratio_all):>8.3f}  "
                  f"{len(feas_all):>6d}")


if __name__ == '__main__':
    main()
