#!/usr/bin/env python3
"""
Matched-Treatment Subgroup Analysis on MIMIC-III.

Addresses P0 issue: the MIMIC "in-target rate lift from 73-81% to 99-100%" is
largely definitional because both the filtering criterion and evaluation metric
rely on the same predicted DBP. This analysis provides outcome-grounded evidence
by examining patients where the selected candidate sequence matches (or closely
matches) the observed treatment — for those patients, we can compare the
*observed* DBP outcome against the target range.

Strategy:
1. For each patient, identify which candidate the RA-constrained and ELBO
   selections chose.
2. Check if that candidate's treatment sequence matches the observed treatment
   (exact match or L1 distance < threshold).
3. For matched patients, report the *observed* in-target DBP rate.
4. Compare RA-matched vs ELBO-matched subgroups.

This is the one route to an outcome-grounded claim on real data (since MIMIC
has no ground-truth counterfactuals).
"""

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

# Paths
DATA_DIR = Path(__file__).parent / 'mimic_ra' / 'VCIP' / 'train' / 'case_infos'
SEEDS = [10, 101, 1010, 10101, 101010]
TAUS = [2, 3, 4, 5]

# Clinical thresholds
DBP_TARGET_LOW = 60.0    # mmHg
DBP_TARGET_HIGH = 90.0   # mmHg
DBP_SAFETY_LOW = 40.0    # mmHg
DBP_SAFETY_HIGH = 120.0  # mmHg

# RA feasibility thresholds (same as E7)
RA_DBP_TARGET = (60.0, 90.0)
RA_DBP_SAFETY = (40.0, 120.0)

# Treatment matching thresholds
EXACT_MATCH_TOL = 0.01   # tolerance for exact match
L1_THRESHOLDS = [0.0, 0.1, 0.2, 0.5, 1.0]  # L1 distance thresholds to try


def load_all_data():
    """Load case info for all seeds and taus."""
    all_data = {}
    for seed in SEEDS:
        pkl_path = DATA_DIR / str(seed) / 'False' / 'case_infos_VCIP.pkl'
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        all_data[seed] = data['VCIP']
    return all_data


def is_feasible(traj_features, idx):
    """Check if candidate idx passes RA feasibility."""
    dbp_terminal = traj_features['dbp_terminal'][idx]
    dbp_min = traj_features['dbp_min'][idx]
    dbp_max = traj_features['dbp_max'][idx]
    in_target = RA_DBP_TARGET[0] <= dbp_terminal <= RA_DBP_TARGET[1]
    safe = RA_DBP_SAFETY[0] <= dbp_min and dbp_max <= RA_DBP_SAFETY[1]
    return in_target and safe


def get_selections(case):
    """Get ELBO-selected and RA-constrained-selected candidate indices."""
    elbos = case['model_losses']
    tf = case['traj_features']

    # ELBO selection: best ELBO
    elbo_idx = int(np.argmin(elbos))

    # RA-constrained: best ELBO among feasible
    feasible = np.zeros(len(elbos), dtype=bool)
    for i in range(len(elbos)):
        feasible[i] = is_feasible(tf, i)

    if feasible.sum() == 0:
        ra_idx = None
    else:
        masked_elbos = elbos.copy()
        masked_elbos[~feasible] = np.inf
        ra_idx = int(np.argmin(masked_elbos))

    return elbo_idx, ra_idx, feasible


def treatment_l1_distance(seq_a, seq_b):
    """Compute per-step mean L1 distance between two treatment sequences."""
    return np.abs(seq_a - seq_b).mean()


def analyze():
    print("=" * 80)
    print("MATCHED-TREATMENT SUBGROUP ANALYSIS ON MIMIC-III")
    print("=" * 80)
    print()

    all_data = load_all_data()

    # =====================================================================
    # Analysis 1: Treatment match rates
    # =====================================================================
    print("ANALYSIS 1: How often does each selection method pick the observed treatment?")
    print("-" * 80)

    for tau in TAUS:
        elbo_exact = 0
        ra_exact = 0
        total = 0
        ra_valid = 0
        elbo_l1s = []
        ra_l1s = []

        for seed in SEEDS:
            for case in all_data[seed][tau]:
                total += 1
                sequences = case['treatment_features']['sequences']
                obs_seq = case['true_sequence'][0, :tau, :]

                elbo_idx, ra_idx, feasible = get_selections(case)

                elbo_seq = sequences[elbo_idx]
                elbo_l1 = treatment_l1_distance(elbo_seq, obs_seq)
                elbo_l1s.append(elbo_l1)
                if elbo_l1 < EXACT_MATCH_TOL:
                    elbo_exact += 1

                if ra_idx is not None:
                    ra_valid += 1
                    ra_seq = sequences[ra_idx]
                    ra_l1 = treatment_l1_distance(ra_seq, obs_seq)
                    ra_l1s.append(ra_l1)
                    if ra_l1 < EXACT_MATCH_TOL:
                        ra_exact += 1

        print(f"  tau={tau}: n={total}, RA valid={ra_valid}")
        print(f"    ELBO exact match: {elbo_exact}/{total} ({100*elbo_exact/total:.1f}%)")
        print(f"    RA exact match:   {ra_exact}/{ra_valid} ({100*ra_exact/ra_valid:.1f}%)" if ra_valid > 0 else "    RA exact match: N/A")
        print(f"    ELBO L1 to obs:   {np.mean(elbo_l1s):.4f} ± {np.std(elbo_l1s):.4f}")
        if ra_l1s:
            print(f"    RA L1 to obs:     {np.mean(ra_l1s):.4f} ± {np.std(ra_l1s):.4f}")
        print()

    # =====================================================================
    # Analysis 2: Observed DBP outcomes for matched subgroups
    # =====================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Observed DBP outcomes for treatment-matched subgroups")
    print("=" * 80)
    print()
    print("For patients where the selected treatment matches the observed treatment,")
    print("we can check whether the *observed* DBP outcome falls in the target range.")
    print()

    for l1_thresh in L1_THRESHOLDS:
        print(f"--- L1 threshold = {l1_thresh} ---")
        print(f"{'tau':>4} | {'n_elbo':>7} | {'n_ra':>7} | "
              f"{'ELBO obs_InTgt':>14} | {'RA obs_InTgt':>14} | "
              f"{'ELBO obs_DBP':>12} | {'RA obs_DBP':>12}")
        print("-" * 90)

        for tau in TAUS:
            elbo_matched_obs_dbps = []
            ra_matched_obs_dbps = []
            elbo_matched_pred_dbps = []
            ra_matched_pred_dbps = []

            for seed in SEEDS:
                for case in all_data[seed][tau]:
                    sequences = case['treatment_features']['sequences']
                    obs_seq = case['true_sequence'][0, :tau, :]
                    obs_dbp = case['observed_dbp']
                    tf = case['traj_features']

                    elbo_idx, ra_idx, feasible = get_selections(case)

                    # ELBO match
                    elbo_seq = sequences[elbo_idx]
                    if treatment_l1_distance(elbo_seq, obs_seq) <= l1_thresh:
                        elbo_matched_obs_dbps.append(obs_dbp)
                        elbo_matched_pred_dbps.append(tf['dbp_terminal'][elbo_idx])

                    # RA match
                    if ra_idx is not None:
                        ra_seq = sequences[ra_idx]
                        if treatment_l1_distance(ra_seq, obs_seq) <= l1_thresh:
                            ra_matched_obs_dbps.append(obs_dbp)
                            ra_matched_pred_dbps.append(tf['dbp_terminal'][ra_idx])

            n_elbo = len(elbo_matched_obs_dbps)
            n_ra = len(ra_matched_obs_dbps)

            if n_elbo > 0:
                elbo_obs_arr = np.array(elbo_matched_obs_dbps)
                elbo_intarget = np.mean((elbo_obs_arr >= DBP_TARGET_LOW) &
                                        (elbo_obs_arr <= DBP_TARGET_HIGH))
                elbo_mean_dbp = np.mean(elbo_obs_arr)
            else:
                elbo_intarget = float('nan')
                elbo_mean_dbp = float('nan')

            if n_ra > 0:
                ra_obs_arr = np.array(ra_matched_obs_dbps)
                ra_intarget = np.mean((ra_obs_arr >= DBP_TARGET_LOW) &
                                      (ra_obs_arr <= DBP_TARGET_HIGH))
                ra_mean_dbp = np.mean(ra_obs_arr)
            else:
                ra_intarget = float('nan')
                ra_mean_dbp = float('nan')

            print(f"{tau:>4} | {n_elbo:>7} | {n_ra:>7} | "
                  f"{elbo_intarget:>13.1%} | {ra_intarget:>13.1%} | "
                  f"{elbo_mean_dbp:>11.1f} | {ra_mean_dbp:>11.1f}")

        print()

    # =====================================================================
    # Analysis 3: All patients — observed DBP by selection method
    # =====================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Observed DBP outcomes for ALL patients (not just matched)")
    print("=" * 80)
    print()
    print("Even without treatment matching, we can stratify patients by whether")
    print("the observed DBP is in-target and check if RA/ELBO selections differ.")
    print()

    print(f"{'tau':>4} | {'n':>5} | "
          f"{'Obs InTgt rate':>14} | {'Obs mean DBP':>12} | "
          f"{'ELBO pred InTgt':>15} | {'RA pred InTgt':>14} | "
          f"{'ELBO=obs idx':>12} | {'RA=obs idx':>12}")
    print("-" * 110)

    for tau in TAUS:
        obs_intarget_list = []
        obs_dbps = []
        elbo_pred_intarget = []
        ra_pred_intarget = []
        elbo_is_obs = []
        ra_is_obs = []

        for seed in SEEDS:
            for case in all_data[seed][tau]:
                obs_dbp = case['observed_dbp']
                tf = case['traj_features']
                obs_it = DBP_TARGET_LOW <= obs_dbp <= DBP_TARGET_HIGH
                obs_intarget_list.append(obs_it)
                obs_dbps.append(obs_dbp)

                elbo_idx, ra_idx, feasible = get_selections(case)

                # Predicted in-target
                elbo_pred_it = (DBP_TARGET_LOW <= tf['dbp_terminal'][elbo_idx] <= DBP_TARGET_HIGH)
                elbo_pred_intarget.append(elbo_pred_it)

                if ra_idx is not None:
                    ra_pred_it = (DBP_TARGET_LOW <= tf['dbp_terminal'][ra_idx] <= DBP_TARGET_HIGH)
                    ra_pred_intarget.append(ra_pred_it)
                else:
                    ra_pred_intarget.append(False)

                # Does selection match observed sequence index (last candidate = 99)?
                elbo_is_obs.append(elbo_idx == 99)
                ra_is_obs.append(ra_idx == 99 if ra_idx is not None else False)

        n = len(obs_intarget_list)
        print(f"{tau:>4} | {n:>5} | "
              f"{np.mean(obs_intarget_list):>13.1%} | {np.mean(obs_dbps):>11.1f} | "
              f"{np.mean(elbo_pred_intarget):>14.1%} | {np.mean(ra_pred_intarget):>13.1%} | "
              f"{np.mean(elbo_is_obs):>11.1%} | {np.mean(ra_is_obs):>11.1%}")

    # =====================================================================
    # Analysis 4: Conditional analysis — among patients with good observed outcomes,
    # does RA select treatments more similar to the observed ones?
    # =====================================================================
    print("\n\n" + "=" * 80)
    print("ANALYSIS 4: Treatment similarity to observed, stratified by observed outcome")
    print("=" * 80)
    print()
    print("If RA selects safer plans, its selections should be closer to the observed")
    print("treatments of patients who had good outcomes (observed DBP in [60,90]).")
    print()

    print(f"{'tau':>4} | {'Stratum':>20} | {'n':>5} | "
          f"{'ELBO L1':>8} | {'RA L1':>8} | "
          f"{'ELBO exact%':>11} | {'RA exact%':>11}")
    print("-" * 85)

    for tau in TAUS:
        strata = {
            'Good (DBP in [60,90])': lambda dbp: DBP_TARGET_LOW <= dbp <= DBP_TARGET_HIGH,
            'Low (DBP < 60)':        lambda dbp: dbp < DBP_TARGET_LOW,
            'High (DBP > 90)':       lambda dbp: dbp > DBP_TARGET_HIGH,
        }

        for stratum_name, stratum_fn in strata.items():
            elbo_l1s = []
            ra_l1s = []
            elbo_exact = 0
            ra_exact = 0
            n = 0

            for seed in SEEDS:
                for case in all_data[seed][tau]:
                    obs_dbp = case['observed_dbp']
                    if not stratum_fn(obs_dbp):
                        continue

                    n += 1
                    sequences = case['treatment_features']['sequences']
                    obs_seq = case['true_sequence'][0, :tau, :]

                    elbo_idx, ra_idx, feasible = get_selections(case)

                    elbo_l1 = treatment_l1_distance(sequences[elbo_idx], obs_seq)
                    elbo_l1s.append(elbo_l1)
                    if elbo_l1 < EXACT_MATCH_TOL:
                        elbo_exact += 1

                    if ra_idx is not None:
                        ra_l1 = treatment_l1_distance(sequences[ra_idx], obs_seq)
                        ra_l1s.append(ra_l1)
                        if ra_l1 < EXACT_MATCH_TOL:
                            ra_exact += 1

            if n == 0:
                continue

            print(f"{tau:>4} | {stratum_name:>20} | {n:>5} | "
                  f"{np.mean(elbo_l1s):>7.4f} | "
                  f"{np.mean(ra_l1s) if ra_l1s else float('nan'):>7.4f} | "
                  f"{100*elbo_exact/n:>10.1f} | "
                  f"{100*ra_exact/len(ra_l1s) if ra_l1s else float('nan'):>10.1f}")

    # =====================================================================
    # Analysis 5: Prediction calibration by observed outcome stratum
    # =====================================================================
    print("\n\n" + "=" * 80)
    print("ANALYSIS 5: Prediction calibration — predicted vs observed DBP")
    print("=" * 80)
    print()

    for tau in TAUS:
        pred_dbps_all = []
        obs_dbps_all = []

        for seed in SEEDS:
            for case in all_data[seed][tau]:
                obs_dbp = case['observed_dbp']
                # Use the observed treatment's predicted DBP (last candidate = 99)
                pred_dbp = case['traj_features']['dbp_terminal'][99]
                pred_dbps_all.append(pred_dbp)
                obs_dbps_all.append(obs_dbp)

        pred = np.array(pred_dbps_all)
        obs = np.array(obs_dbps_all)
        mae = np.mean(np.abs(pred - obs))
        bias = np.mean(pred - obs)
        corr = np.corrcoef(pred, obs)[0, 1]

        # Concordance: both in-target or both out-of-target
        pred_it = (pred >= DBP_TARGET_LOW) & (pred <= DBP_TARGET_HIGH)
        obs_it = (obs >= DBP_TARGET_LOW) & (obs <= DBP_TARGET_HIGH)
        concordance = np.mean(pred_it == obs_it)

        print(f"  tau={tau}: n={len(obs)}, MAE={mae:.1f} mmHg, bias={bias:+.1f}, "
              f"Pearson r={corr:.3f}, in-target concordance={concordance:.1%}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    analyze()
