"""
S6.2 Extension: Cross-Model Failure Taxonomy
Shows failure modes are structural (correlated across all 5 models), not model-specific.

Uses true_losses to define "danger" as high ELBO selection regret.
For VCIP specifically, also computes constraint-based danger from traj_features.
"""

import pickle
import numpy as np
import os
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr

BASE_R = 'results_remote/r/my_outputs/cancer_sim_cont/22/coeff_4'
BASE_RA = 'results_remote/phase1_ra_v2/my_outputs/cancer_sim_cont/22'
SEEDS = [10, 101, 1010, 10101, 101010]
TAUS = [2, 4, 6, 8]
GAMMA = 4

TARGET_UPPER = 3.0
SAFETY_VOL_UPPER = 12.0
SAFETY_CHEMO_UPPER = 5.0

MODEL_PATHS = {
    'VCIP': 'VCIP',
    'CRN': 'CRN',
    'CT': 'CT/0.01',
    'RMSN': 'RMSN',
    'ACTIN': 'ACTIN/0.01',
}
MODEL_NAMES = list(MODEL_PATHS.keys())


def load_model_data(model_name, seed, base=BASE_R):
    """Load pickle for a given model and seed."""
    path = os.path.join(base, MODEL_PATHS[model_name],
                        'train/True/case_infos', str(seed), 'False',
                        f'case_infos_{model_name}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data[list(data.keys())[0]]
    return None


def load_vcip_ra_data(gamma, seed):
    """Load VCIP data with traj_features from phase1_ra_v2."""
    path = os.path.join(BASE_RA,
                        f'coeff_{gamma}/VCIP/train/True/case_infos/{seed}/False/case_infos_VCIP.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)['VCIP']
    return None


def analysis_1_selection_quality():
    """Per-model selection quality: ELBO ranking vs oracle."""
    print("=" * 90)
    print("1. SELECTION QUALITY BY MODEL (gamma=4)")
    print("   Metrics: Spearman(ELBO, oracle), mean regret, % zero regret")
    print("=" * 90)

    header = f"{'Model':>6} | {'tau':>3} | {'Spearman':>8} | {'Mean regret':>12} | {'Med regret':>11} | {'% zero':>7} | {'ELBO true':>10}"
    print(header)
    print("-" * 90)

    for model in MODEL_NAMES:
        for tau in TAUS:
            rhos = []
            regrets = []
            elbo_true_losses = []

            for seed in SEEDS:
                data = load_model_data(model, seed)
                if data is None or tau not in data:
                    continue
                for case in data[tau]:
                    ml = case['model_losses']
                    tl = case['true_losses']

                    rho, _ = spearmanr(ml, tl)
                    if not np.isnan(rho):
                        rhos.append(rho)

                    elbo_idx = np.argmin(ml)
                    oracle_idx = np.argmin(tl)
                    regret = tl[elbo_idx] - tl[oracle_idx]
                    regrets.append(regret)
                    elbo_true_losses.append(tl[elbo_idx])

            rhos = np.array(rhos)
            regrets = np.array(regrets)

            print(f"{model:>6} | {tau:>3} | {rhos.mean():>8.4f} | {regrets.mean():>12.6f} | "
                  f"{np.median(regrets):>11.6f} | {(regrets == 0).mean()*100:>6.1f}% | "
                  f"{np.mean(elbo_true_losses):>10.6f}")
        print()


def analysis_2_cross_model_failure_correlation():
    """Cross-model correlation of per-patient regret."""
    print("\n" + "=" * 90)
    print("2. CROSS-MODEL FAILURE CORRELATION (gamma=4)")
    print("   Per-patient regret correlation across model pairs")
    print("=" * 90)

    for tau in TAUS:
        # Build per-patient regret vectors for each model
        patient_regrets = {m: {} for m in MODEL_NAMES}

        for model in MODEL_NAMES:
            for seed in SEEDS:
                data = load_model_data(model, seed)
                if data is None or tau not in data:
                    continue
                for case in data[tau]:
                    pid = case['individual_id']
                    ml = case['model_losses']
                    tl = case['true_losses']
                    elbo_idx = np.argmin(ml)
                    oracle_idx = np.argmin(tl)
                    regret = tl[elbo_idx] - tl[oracle_idx]

                    if pid not in patient_regrets[model]:
                        patient_regrets[model][pid] = []
                    patient_regrets[model][pid].append(regret)

        # Average regret per patient per model
        avg_regrets = {}
        for model in MODEL_NAMES:
            avg_regrets[model] = {}
            for pid, vals in patient_regrets[model].items():
                avg_regrets[model][pid] = np.mean(vals)

        # Common patients
        common_pids = set.intersection(*[set(avg_regrets[m].keys()) for m in MODEL_NAMES])
        common_pids = sorted(common_pids)

        print(f"\n  tau={tau} ({len(common_pids)} patients):")
        print(f"  {'':>6}", end="")
        for m2 in MODEL_NAMES:
            print(f" {m2:>6}", end="")
        print()

        for m1 in MODEL_NAMES:
            print(f"  {m1:>6}", end="")
            v1 = np.array([avg_regrets[m1][p] for p in common_pids])
            for m2 in MODEL_NAMES:
                v2 = np.array([avg_regrets[m2][p] for p in common_pids])
                if m1 == m2:
                    print(f"   1.00", end="")
                else:
                    rho, _ = spearmanr(v1, v2)
                    print(f"  {rho:>5.2f}", end="")
            print()


def analysis_3_universally_dangerous():
    """Identify patients that are dangerous under ALL models."""
    print("\n" + "=" * 90)
    print("3. UNIVERSALLY DANGEROUS PATIENTS (gamma=4)")
    print("   Patients where ALL 5 models' ELBO selections have above-median regret")
    print("=" * 90)

    for tau in TAUS:
        # Per-patient, per-model: is ELBO selection in worst 50%?
        patient_danger = {m: {} for m in MODEL_NAMES}

        for model in MODEL_NAMES:
            for seed in SEEDS:
                data = load_model_data(model, seed)
                if data is None or tau not in data:
                    continue
                for case in data[tau]:
                    pid = case['individual_id']
                    ml = case['model_losses']
                    tl = case['true_losses']

                    elbo_idx = np.argmin(ml)
                    # Is ELBO selection in worst half by true loss?
                    rank = np.argsort(np.argsort(tl))  # rank array
                    elbo_rank_pct = rank[elbo_idx] / len(tl)

                    if pid not in patient_danger[model]:
                        patient_danger[model][pid] = []
                    patient_danger[model][pid].append(elbo_rank_pct > 0.5)

        # Average danger rate per patient per model
        patient_danger_rate = {}
        for model in MODEL_NAMES:
            patient_danger_rate[model] = {}
            for pid, vals in patient_danger[model].items():
                patient_danger_rate[model][pid] = np.mean(vals)

        common_pids = sorted(set.intersection(*[set(patient_danger_rate[m].keys()) for m in MODEL_NAMES]))

        # Universally dangerous: danger rate > 0.5 under ALL models
        universal = []
        for pid in common_pids:
            rates = [patient_danger_rate[m][pid] for m in MODEL_NAMES]
            if all(r > 0.5 for r in rates):
                universal.append((pid, np.mean(rates)))

        # At least 3/5 models
        three_plus = []
        for pid in common_pids:
            rates = [patient_danger_rate[m][pid] for m in MODEL_NAMES]
            n_dangerous = sum(1 for r in rates if r > 0.5)
            if n_dangerous >= 3:
                three_plus.append((pid, n_dangerous, np.mean(rates)))

        print(f"\n  tau={tau} ({len(common_pids)} patients):")
        print(f"    Dangerous under ALL 5 models: {len(universal)} patients")
        print(f"    Dangerous under ≥3 models:    {len(three_plus)} patients")
        print(f"    Dangerous under 0 models:     {sum(1 for p in common_pids if all(patient_danger_rate[m][p] <= 0.5 for m in MODEL_NAMES))} patients")

        if universal:
            print(f"    Universal patients: {[u[0] for u in universal[:10]]}")


def analysis_4_vcip_constraint_danger():
    """VCIP-specific: constraint-based danger using traj_features (for reference)."""
    print("\n" + "=" * 90)
    print("4. VCIP CONSTRAINT-BASED DANGER (from RA data, all gammas)")
    print("   For comparison with regret-based cross-model analysis")
    print("=" * 90)

    for gamma in [1, 2, 3, 4]:
        for tau in TAUS:
            total = 0
            unsafe = 0
            for seed in SEEDS:
                data = load_vcip_ra_data(gamma, seed)
                if data is None or tau not in data:
                    continue
                for case in data[tau]:
                    tf = case['traj_features']
                    ml = case['model_losses']
                    elbo_idx = np.argmin(ml)

                    cv_t = tf['cv_terminal'][elbo_idx]
                    cv_m = tf['cv_max'][elbo_idx]
                    cd_m = tf['cd_max'][elbo_idx]

                    is_safe = (cv_t <= TARGET_UPPER) and (cv_m <= SAFETY_VOL_UPPER) and (cd_m <= SAFETY_CHEMO_UPPER)
                    total += 1
                    if not is_safe:
                        unsafe += 1

            if total > 0:
                print(f"  gamma={gamma}, tau={tau}: {unsafe}/{total} unsafe ({100*unsafe/total:.1f}%)")


def analysis_5_model_agreement_on_rankings():
    """Do models agree on which treatment plans are good?"""
    print("\n" + "=" * 90)
    print("5. MODEL AGREEMENT ON CANDIDATE RANKINGS (gamma=4)")
    print("   Pairwise Spearman of model_losses across k=100 candidates, averaged over patients")
    print("=" * 90)

    for tau in TAUS:
        pair_rhos = defaultdict(list)

        for seed in SEEDS:
            # Load all models for this seed
            model_data = {}
            for model in MODEL_NAMES:
                data = load_model_data(model, seed)
                if data is not None and tau in data:
                    model_data[model] = {c['individual_id']: c for c in data[tau]}

            if len(model_data) < 2:
                continue

            # For each patient, compute pairwise Spearman of model_losses
            common_pids = set.intersection(*[set(d.keys()) for d in model_data.values()])

            for pid in common_pids:
                for i, m1 in enumerate(MODEL_NAMES):
                    if m1 not in model_data:
                        continue
                    for m2 in MODEL_NAMES[i+1:]:
                        if m2 not in model_data:
                            continue
                        ml1 = model_data[m1][pid]['model_losses']
                        ml2 = model_data[m2][pid]['model_losses']
                        rho, _ = spearmanr(ml1, ml2)
                        if not np.isnan(rho):
                            pair_rhos[(m1, m2)].append(rho)

        print(f"\n  tau={tau}:")
        print(f"  {'':>6}", end="")
        for m2 in MODEL_NAMES:
            print(f" {m2:>6}", end="")
        print()

        for m1 in MODEL_NAMES:
            print(f"  {m1:>6}", end="")
            for m2 in MODEL_NAMES:
                if m1 == m2:
                    print(f"   1.00", end="")
                elif (m1, m2) in pair_rhos:
                    print(f"  {np.mean(pair_rhos[(m1, m2)]):>5.2f}", end="")
                elif (m2, m1) in pair_rhos:
                    print(f"  {np.mean(pair_rhos[(m2, m1)]):>5.2f}", end="")
                else:
                    print(f"    N/A", end="")
            print()


if __name__ == '__main__':
    print("S6.2 Extension: Cross-Model Failure Taxonomy")
    print("=" * 90)

    analysis_1_selection_quality()
    analysis_2_cross_model_failure_correlation()
    analysis_3_universally_dangerous()
    analysis_4_vcip_constraint_danger()
    analysis_5_model_agreement_on_rankings()

    print("\nDone.")
