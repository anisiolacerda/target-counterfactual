"""
B1: MIMIC Calibration + Correction Rate Analysis

Analyzes existing MIMIC-III RA pickle data to:
1. Report full predicted DBP distribution (mean, std, percentiles)
2. Compare predicted DBP against observed DBP (calibration)
3. Compute correction rate: fraction of ELBO-selected plans out-of-target corrected by RA
4. Analyze whether the filter discriminates meaningfully
"""

import pickle
import numpy as np
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [10, 101, 1010, 10101, 101010]
TAUS = [2, 3, 4, 5]

# RA thresholds for MIMIC (DBP in mmHg)
TARGET_LOWER = 60.0
TARGET_UPPER = 90.0
SAFETY_LOWER = 50.0
SAFETY_UPPER = 100.0


def load_all_data():
    """Load pickle files for all seeds."""
    all_data = {}
    for seed in SEEDS:
        pkl_path = os.path.join(
            BASE_DIR, 'VCIP/train/case_infos', str(seed), 'False', 'case_infos_VCIP.pkl'
        )
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            all_data[seed] = data['VCIP']
            print(f"  Seed {seed}: loaded, taus={list(data['VCIP'].keys())}")
        else:
            print(f"  Seed {seed}: NOT FOUND at {pkl_path}")
    return all_data


def analyze_dbp_distribution(all_data):
    """1. Report full predicted DBP distribution."""
    print("\n" + "=" * 70)
    print("1. PREDICTED DBP DISTRIBUTION (across all seeds, patients, candidates)")
    print("=" * 70)

    for tau in TAUS:
        all_terminal = []
        all_min = []
        all_max = []
        all_traj = []

        for seed, seed_data in all_data.items():
            if tau not in seed_data:
                continue
            for case in seed_data[tau]:
                tf = case['traj_features']
                all_terminal.extend(tf['dbp_terminal'].tolist())
                all_min.extend(tf['dbp_min'].tolist())
                all_max.extend(tf['dbp_max'].tolist())
                all_traj.append(tf['dbp_trajectory'])  # (k, tau)

        all_terminal = np.array(all_terminal)
        all_min = np.array(all_min)
        all_max = np.array(all_max)

        print(f"\n  tau={tau} ({len(all_terminal)} total candidate sequences):")
        print(f"    DBP terminal: mean={all_terminal.mean():.2f}, "
              f"std={all_terminal.std():.2f}, "
              f"median={np.median(all_terminal):.2f}")
        print(f"      percentiles: [1%={np.percentile(all_terminal, 1):.2f}, "
              f"5%={np.percentile(all_terminal, 5):.2f}, "
              f"25%={np.percentile(all_terminal, 25):.2f}, "
              f"75%={np.percentile(all_terminal, 75):.2f}, "
              f"95%={np.percentile(all_terminal, 95):.2f}, "
              f"99%={np.percentile(all_terminal, 99):.2f}]")
        print(f"    DBP min (trajectory): mean={all_min.mean():.2f}, "
              f"std={all_min.std():.2f}")
        print(f"    DBP max (trajectory): mean={all_max.mean():.2f}, "
              f"std={all_max.std():.2f}")
        print(f"    Range of terminal values: [{all_terminal.min():.2f}, {all_terminal.max():.2f}]")


def analyze_calibration(all_data):
    """2. Compare predicted DBP against observed DBP."""
    print("\n" + "=" * 70)
    print("2. CALIBRATION: PREDICTED vs OBSERVED DBP")
    print("=" * 70)

    for tau in TAUS:
        pred_terminals = []  # best ELBO candidate's terminal DBP
        obs_dbps = []
        all_pred_means = []  # mean across all candidates per patient

        for seed, seed_data in all_data.items():
            if tau not in seed_data:
                continue
            for case in seed_data[tau]:
                obs_dbp = case.get('observed_dbp', None)
                if obs_dbp is None:
                    continue
                obs_dbps.append(obs_dbp)

                # Best ELBO candidate
                model_losses = case['model_losses']
                best_idx = np.argmin(model_losses)
                tf = case['traj_features']
                pred_terminals.append(tf['dbp_terminal'][best_idx])
                all_pred_means.append(tf['dbp_terminal'].mean())

        if not obs_dbps:
            print(f"\n  tau={tau}: No observed DBP data available")
            continue

        obs_dbps = np.array(obs_dbps)
        pred_terminals = np.array(pred_terminals)
        all_pred_means = np.array(all_pred_means)

        # Correlation
        corr = np.corrcoef(obs_dbps, pred_terminals)[0, 1]
        mae = np.abs(obs_dbps - pred_terminals).mean()
        rmse = np.sqrt(((obs_dbps - pred_terminals) ** 2).mean())
        bias = (pred_terminals - obs_dbps).mean()

        corr_mean = np.corrcoef(obs_dbps, all_pred_means)[0, 1]

        print(f"\n  tau={tau} ({len(obs_dbps)} patients × 5 seeds = {len(obs_dbps)} observations):")
        print(f"    Observed DBP:  mean={obs_dbps.mean():.2f}, std={obs_dbps.std():.2f}, "
              f"range=[{obs_dbps.min():.2f}, {obs_dbps.max():.2f}]")
        print(f"    Predicted DBP (ELBO-best): mean={pred_terminals.mean():.2f}, "
              f"std={pred_terminals.std():.2f}, range=[{pred_terminals.min():.2f}, {pred_terminals.max():.2f}]")
        print(f"    Predicted DBP (candidate mean): mean={all_pred_means.mean():.2f}, "
              f"std={all_pred_means.std():.2f}")
        print(f"    Calibration metrics:")
        print(f"      Pearson r (ELBO-best vs obs): {corr:.4f}")
        print(f"      Pearson r (candidate mean vs obs): {corr_mean:.4f}")
        print(f"      MAE: {mae:.2f} mmHg")
        print(f"      RMSE: {rmse:.2f} mmHg")
        print(f"      Bias (pred - obs): {bias:+.2f} mmHg")

        # Distribution comparison
        print(f"    Observed in target [{TARGET_LOWER}, {TARGET_UPPER}]: "
              f"{np.sum((obs_dbps >= TARGET_LOWER) & (obs_dbps <= TARGET_UPPER))}/{len(obs_dbps)} "
              f"({100 * np.mean((obs_dbps >= TARGET_LOWER) & (obs_dbps <= TARGET_UPPER)):.1f}%)")
        print(f"    Predicted (ELBO-best) in target: "
              f"{np.sum((pred_terminals >= TARGET_LOWER) & (pred_terminals <= TARGET_UPPER))}/{len(pred_terminals)} "
              f"({100 * np.mean((pred_terminals >= TARGET_LOWER) & (pred_terminals <= TARGET_UPPER)):.1f}%)")


def analyze_correction_rate(all_data):
    """3. Compute correction rate: how many ELBO-selected plans get corrected by RA."""
    print("\n" + "=" * 70)
    print("3. CORRECTION RATE ANALYSIS")
    print("=" * 70)
    print("   (Does RA filtering change the selection? How often?)")

    for tau in TAUS:
        n_patients = 0
        n_elbo_in_target = 0
        n_elbo_out_target = 0
        n_corrected = 0  # ELBO out-of-target, RA selects in-target
        n_ra_changes_selection = 0  # RA picks different plan than ELBO
        n_feasible_total = 0  # candidates in target
        n_candidates_total = 0
        elbo_dbp_list = []
        ra_dbp_list = []

        for seed, seed_data in all_data.items():
            if tau not in seed_data:
                continue
            for case in seed_data[tau]:
                tf = case['traj_features']
                model_losses = case['model_losses']
                dbp_terminal = tf['dbp_terminal']
                dbp_trajectory = tf['dbp_trajectory']  # (k, tau)
                k = len(model_losses)
                n_candidates_total += k
                n_patients += 1

                # ELBO-best selection
                elbo_best_idx = np.argmin(model_losses)
                elbo_dbp = dbp_terminal[elbo_best_idx]
                elbo_dbp_list.append(elbo_dbp)

                elbo_in_target = TARGET_LOWER <= elbo_dbp <= TARGET_UPPER
                if elbo_in_target:
                    n_elbo_in_target += 1
                else:
                    n_elbo_out_target += 1

                # RA-feasible set: terminal in target AND trajectory in safety bounds
                feasible_mask = np.ones(k, dtype=bool)
                # Terminal target
                feasible_mask &= (dbp_terminal >= TARGET_LOWER) & (dbp_terminal <= TARGET_UPPER)
                # Trajectory safety (all steps)
                for step in range(dbp_trajectory.shape[1]):
                    feasible_mask &= (dbp_trajectory[:, step] >= SAFETY_LOWER) & \
                                     (dbp_trajectory[:, step] <= SAFETY_UPPER)

                n_feasible = feasible_mask.sum()
                n_feasible_total += n_feasible

                if n_feasible > 0:
                    # RA selection: best ELBO among feasible
                    feasible_losses = model_losses.copy()
                    feasible_losses[~feasible_mask] = np.inf
                    ra_best_idx = np.argmin(feasible_losses)
                    ra_dbp = dbp_terminal[ra_best_idx]
                    ra_dbp_list.append(ra_dbp)

                    if ra_best_idx != elbo_best_idx:
                        n_ra_changes_selection += 1

                    if not elbo_in_target and TARGET_LOWER <= ra_dbp <= TARGET_UPPER:
                        n_corrected += 1
                else:
                    # No feasible candidates — RA falls back to ELBO
                    ra_dbp_list.append(elbo_dbp)

        if n_patients == 0:
            print(f"\n  tau={tau}: No data")
            continue

        elbo_dbp_arr = np.array(elbo_dbp_list)
        ra_dbp_arr = np.array(ra_dbp_list)
        feasibility_rate = n_feasible_total / n_candidates_total if n_candidates_total > 0 else 0

        print(f"\n  tau={tau} ({n_patients} patient-seed observations):")
        print(f"    Candidate feasibility: {n_feasible_total}/{n_candidates_total} "
              f"({100 * feasibility_rate:.1f}%)")
        print(f"    ELBO-best in target: {n_elbo_in_target}/{n_patients} "
              f"({100 * n_elbo_in_target / n_patients:.1f}%)")
        print(f"    ELBO-best out-of-target: {n_elbo_out_target}/{n_patients} "
              f"({100 * n_elbo_out_target / n_patients:.1f}%)")
        print(f"    RA changes selection: {n_ra_changes_selection}/{n_patients} "
              f"({100 * n_ra_changes_selection / n_patients:.1f}%)")
        print(f"    Corrections (out→in): {n_corrected}/{n_elbo_out_target} "
              f"({100 * n_corrected / max(n_elbo_out_target, 1):.1f}%)")
        print(f"    DBP shift (ELBO→RA): {elbo_dbp_arr.mean():.2f} → {ra_dbp_arr.mean():.2f} mmHg "
              f"(Δ={ra_dbp_arr.mean() - elbo_dbp_arr.mean():+.2f})")


def analyze_filter_discrimination(all_data):
    """4. Does the filter discriminate meaningfully, or just pick boundary noise?"""
    print("\n" + "=" * 70)
    print("4. FILTER DISCRIMINATION ANALYSIS")
    print("=" * 70)
    print("   (Is the filter selecting meaningfully different plans?)")

    for tau in TAUS:
        elbo_losses_selected = []
        ra_losses_selected = []
        dbp_spreads = []
        treatment_diffs = []

        for seed, seed_data in all_data.items():
            if tau not in seed_data:
                continue
            for case in seed_data[tau]:
                tf = case['traj_features']
                model_losses = case['model_losses']
                dbp_terminal = tf['dbp_terminal']
                dbp_trajectory = tf['dbp_trajectory']
                k = len(model_losses)

                elbo_best_idx = np.argmin(model_losses)
                elbo_losses_selected.append(model_losses[elbo_best_idx])

                # RA feasible
                feasible_mask = np.ones(k, dtype=bool)
                feasible_mask &= (dbp_terminal >= TARGET_LOWER) & (dbp_terminal <= TARGET_UPPER)
                for step in range(dbp_trajectory.shape[1]):
                    feasible_mask &= (dbp_trajectory[:, step] >= SAFETY_LOWER) & \
                                     (dbp_trajectory[:, step] <= SAFETY_UPPER)

                if feasible_mask.sum() > 0:
                    feasible_losses = model_losses.copy()
                    feasible_losses[~feasible_mask] = np.inf
                    ra_best_idx = np.argmin(feasible_losses)
                    ra_losses_selected.append(model_losses[ra_best_idx])

                    # Treatment difference (if available)
                    treat_feat = case.get('treatment_features', None)
                    if treat_feat is not None:
                        vaso_elbo = treat_feat['vaso_total'][elbo_best_idx]
                        vaso_ra = treat_feat['vaso_total'][ra_best_idx]
                        vent_elbo = treat_feat['vent_total'][elbo_best_idx]
                        vent_ra = treat_feat['vent_total'][ra_best_idx]
                        treatment_diffs.append({
                            'vaso_diff': vaso_ra - vaso_elbo,
                            'vent_diff': vent_ra - vent_elbo,
                            'vaso_elbo': vaso_elbo,
                            'vaso_ra': vaso_ra,
                            'vent_elbo': vent_elbo,
                            'vent_ra': vent_ra,
                        })
                else:
                    ra_losses_selected.append(model_losses[elbo_best_idx])

                # Spread of DBP predictions for this patient
                dbp_spreads.append(dbp_terminal.max() - dbp_terminal.min())

        if not elbo_losses_selected:
            continue

        elbo_arr = np.array(elbo_losses_selected)
        ra_arr = np.array(ra_losses_selected)
        spread_arr = np.array(dbp_spreads)

        print(f"\n  tau={tau}:")
        print(f"    ELBO of selected plan:")
        print(f"      ELBO-best: mean={elbo_arr.mean():.4f}, std={elbo_arr.std():.4f}")
        print(f"      RA-best:   mean={ra_arr.mean():.4f}, std={ra_arr.std():.4f}")
        elbo_cost = ra_arr.mean() - elbo_arr.mean()
        print(f"      ELBO cost of safety: {elbo_cost:+.4f} "
              f"({100 * elbo_cost / abs(elbo_arr.mean()):.2f}% relative)")

        print(f"    DBP prediction spread per patient:")
        print(f"      mean spread: {spread_arr.mean():.2f} mmHg, "
              f"std: {spread_arr.std():.2f}")
        print(f"      min spread: {spread_arr.min():.2f}, max: {spread_arr.max():.2f}")

        if treatment_diffs:
            vaso_diffs = np.array([t['vaso_diff'] for t in treatment_diffs])
            vent_diffs = np.array([t['vent_diff'] for t in treatment_diffs])
            vaso_elbo = np.array([t['vaso_elbo'] for t in treatment_diffs])
            vaso_ra = np.array([t['vaso_ra'] for t in treatment_diffs])
            vent_elbo = np.array([t['vent_elbo'] for t in treatment_diffs])
            vent_ra = np.array([t['vent_ra'] for t in treatment_diffs])

            print(f"    Treatment changes (ELBO → RA) [{len(treatment_diffs)} cases w/ feasible RA]:")
            print(f"      Vasopressor total: {vaso_elbo.mean():.3f} → {vaso_ra.mean():.3f} "
                  f"(Δ={vaso_diffs.mean():+.3f})")
            print(f"      Ventilation total: {vent_elbo.mean():.3f} → {vent_ra.mean():.3f} "
                  f"(Δ={vent_diffs.mean():+.3f})")
            changed = np.sum(np.abs(vaso_diffs) > 0.01) + np.sum(np.abs(vent_diffs) > 0.01)
            print(f"      Treatment plans changed: {changed}/{2 * len(treatment_diffs)} "
                  f"dimensions ({100 * changed / (2 * len(treatment_diffs)):.1f}%)")


def per_seed_summary(all_data):
    """Summary table per seed for paper."""
    print("\n" + "=" * 70)
    print("5. PER-SEED SUMMARY (for paper table)")
    print("=" * 70)

    for tau in TAUS:
        print(f"\n  tau={tau}:")
        print(f"  {'Seed':>8s}  {'N':>4s}  {'Feas%':>6s}  {'ELBO_inT%':>10s}  "
              f"{'RA_inT%':>8s}  {'Corr%':>6s}  {'ELBO_DBP':>9s}  {'RA_DBP':>7s}")

        for seed in SEEDS:
            if seed not in all_data or tau not in all_data[seed]:
                continue
            cases = all_data[seed][tau]
            n = len(cases)
            n_feas = 0
            n_elbo_in = 0
            n_ra_in = 0
            n_corrected = 0
            elbo_dbps = []
            ra_dbps = []

            for case in cases:
                tf = case['traj_features']
                ml = case['model_losses']
                dbp_t = tf['dbp_terminal']
                dbp_traj = tf['dbp_trajectory']
                k = len(ml)

                elbo_idx = np.argmin(ml)
                elbo_dbp = dbp_t[elbo_idx]
                elbo_dbps.append(elbo_dbp)
                elbo_in = TARGET_LOWER <= elbo_dbp <= TARGET_UPPER
                if elbo_in:
                    n_elbo_in += 1

                feas = np.ones(k, dtype=bool)
                feas &= (dbp_t >= TARGET_LOWER) & (dbp_t <= TARGET_UPPER)
                for s in range(dbp_traj.shape[1]):
                    feas &= (dbp_traj[:, s] >= SAFETY_LOWER) & (dbp_traj[:, s] <= SAFETY_UPPER)
                n_feas += feas.sum()

                if feas.sum() > 0:
                    fl = ml.copy()
                    fl[~feas] = np.inf
                    ra_idx = np.argmin(fl)
                    ra_dbp = dbp_t[ra_idx]
                    ra_dbps.append(ra_dbp)
                    ra_in = TARGET_LOWER <= ra_dbp <= TARGET_UPPER
                    if ra_in:
                        n_ra_in += 1
                    if not elbo_in and ra_in:
                        n_corrected += 1
                else:
                    ra_dbps.append(elbo_dbp)
                    if elbo_in:
                        n_ra_in += 1

            feas_pct = 100 * n_feas / (n * 100)  # k=100 per patient
            elbo_in_pct = 100 * n_elbo_in / n
            ra_in_pct = 100 * n_ra_in / n
            n_out = n - n_elbo_in
            corr_pct = 100 * n_corrected / max(n_out, 1) if n_out > 0 else 0.0
            print(f"  {seed:>8d}  {n:>4d}  {feas_pct:>5.1f}%  {elbo_in_pct:>9.1f}%  "
                  f"{ra_in_pct:>7.1f}%  {corr_pct:>5.1f}%  "
                  f"{np.mean(elbo_dbps):>8.2f}  {np.mean(ra_dbps):>6.2f}")


if __name__ == '__main__':
    print("B1: MIMIC-III RA Calibration & Correction Rate Analysis")
    print("=" * 70)
    print(f"Target range: [{TARGET_LOWER}, {TARGET_UPPER}] mmHg")
    print(f"Safety range: [{SAFETY_LOWER}, {SAFETY_UPPER}] mmHg")
    print(f"Seeds: {SEEDS}")
    print(f"Taus: {TAUS}")
    print("\nLoading data...")
    all_data = load_all_data()

    analyze_dbp_distribution(all_data)
    analyze_calibration(all_data)
    analyze_correction_rate(all_data)
    analyze_filter_discrimination(all_data)
    per_seed_summary(all_data)

    print("\n" + "=" * 70)
    print("Analysis complete.")
