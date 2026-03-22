#!/usr/bin/env python3
"""E7: MIMIC-III RA-Constrained Selection Analysis.

Executes the E7 analysis cells from analysis.ipynb as a standalone script.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', font_scale=1.2)

PROJECT_ROOT = Path('/Users/anisiomlacerda/code/target-counterfactual')
MIMIC_RESULTS = PROJECT_ROOT / 'results_remote' / 'mimic_ra'
FIG_DIR = str(PROJECT_ROOT / 'lightning-hydra-template-main' / 'src' / 'reach_avoid' / 'figures')
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

MIMIC_SEEDS = [10, 101, 1010, 10101, 101010]
MIMIC_TAUS = [2, 3, 4, 5]

# Clinical thresholds for diastolic BP
DBP_TARGET_LOW = 60.0   # mmHg
DBP_TARGET_HIGH = 90.0  # mmHg
DBP_SAFETY_LOW = 40.0   # avoid severe hypotension
DBP_SAFETY_HIGH = 120.0  # avoid hypertensive crisis

# ── Cell 38: Load MIMIC-III RA evaluation results ──
print('=' * 70)
print('E7: MIMIC-III RA-Constrained Selection (Clinical Plausibility)')
print('=' * 70)

mimic_data = {}
for seed in MIMIC_SEEDS:
    pkl_path = (MIMIC_RESULTS / 'VCIP' / 'train' / 'case_infos' /
                str(seed) / 'False' / 'case_infos_VCIP.pkl')
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        mimic_data[seed] = data['VCIP']
    else:
        print(f'Missing: seed={seed} ({pkl_path})')

if not mimic_data:
    print('No MIMIC RA results found.')
    print(f'Expected path: {MIMIC_RESULTS}')
    exit(1)

sample_seed = list(mimic_data.keys())[0]
sample_tau = list(mimic_data[sample_seed].keys())[0]
sample_ci = mimic_data[sample_seed][sample_tau][0]
print(f'Loaded {len(mimic_data)} seeds')
print(f'Taus: {list(mimic_data[sample_seed].keys())}')
print(f'Case info keys: {list(sample_ci.keys())}')
print(f'traj_features keys: {list(sample_ci["traj_features"].keys())}')

# DBP distribution summary
dbp_all = []
for ci in mimic_data[sample_seed][sample_tau]:
    dbp_all.extend(ci['traj_features']['dbp_terminal'].tolist())
dbp_all = np.array(dbp_all)
print(f'\nDBP terminal distribution (seed={sample_seed}, tau={sample_tau}):')
print(f'  mean={dbp_all.mean():.1f}, std={dbp_all.std():.1f}')
print(f'  min={dbp_all.min():.1f}, p25={np.percentile(dbp_all,25):.1f}, '
      f'p50={np.percentile(dbp_all,50):.1f}, p75={np.percentile(dbp_all,75):.1f}, '
      f'max={dbp_all.max():.1f}')


# ── Cell 39: RA-constrained selection ──
def evaluate_mimic_constrained(case_infos, target_low=60, target_high=90,
                                 safety_low=40, safety_high=120):
    results = []
    for ci in case_infos:
        if 'traj_features' not in ci:
            continue
        tf = ci['traj_features']
        elbos = ci['model_losses']
        k = len(elbos)

        dbp_terminal = tf['dbp_terminal']
        dbp_min = tf['dbp_min']
        dbp_max = tf['dbp_max']

        # Feasibility: terminal in target range AND trajectory in safety range
        feasible = ((dbp_terminal >= target_low) & (dbp_terminal <= target_high) &
                     (dbp_min >= safety_low) & (dbp_max <= safety_high))

        n_feasible = feasible.sum()
        feas_rate = n_feasible / k

        # Unconstrained ELBO pick
        best_elbo_idx = np.argmin(elbos)
        elbo_dbp = float(dbp_terminal[best_elbo_idx])
        elbo_feasible = bool(feasible[best_elbo_idx])

        # Constrained pick
        if n_feasible > 0:
            feasible_elbos = np.where(feasible, elbos, np.inf)
            constrained_idx = np.argmin(feasible_elbos)
            cstr_dbp = float(dbp_terminal[constrained_idx])
            cstr_feasible = True
        else:
            constrained_idx = best_elbo_idx
            cstr_dbp = elbo_dbp
            cstr_feasible = False

        # Treatment features
        tf_treat = ci.get('treatment_features', {})

        results.append({
            'individual_id': ci['individual_id'],
            'observed_dbp': ci.get('observed_dbp', np.nan),
            'feasibility_rate': feas_rate,
            'n_feasible': n_feasible,
            'elbo_dbp': elbo_dbp,
            'elbo_feasible': elbo_feasible,
            'cstr_dbp': cstr_dbp,
            'cstr_feasible': cstr_feasible,
            'elbo_in_target': target_low <= elbo_dbp <= target_high,
            'cstr_in_target': target_low <= cstr_dbp <= target_high,
            'elbo_vaso': float(tf_treat.get('vaso_total', np.zeros(k))[best_elbo_idx]) if 'vaso_total' in tf_treat else np.nan,
            'cstr_vaso': float(tf_treat.get('vaso_total', np.zeros(k))[constrained_idx]) if 'vaso_total' in tf_treat else np.nan,
            'elbo_vent': float(tf_treat.get('vent_total', np.zeros(k))[best_elbo_idx]) if 'vent_total' in tf_treat else np.nan,
            'cstr_vent': float(tf_treat.get('vent_total', np.zeros(k))[constrained_idx]) if 'vent_total' in tf_treat else np.nan,
        })
    return results

print(f'\n=== E7: MIMIC-III Constrained Selection ===')
print(f'Target: DBP ∈ [{DBP_TARGET_LOW}, {DBP_TARGET_HIGH}] mmHg')
print(f'Safety: DBP ∈ [{DBP_SAFETY_LOW}, {DBP_SAFETY_HIGH}] mmHg at all steps')
print()

print(f'{"Tau":>4} | {"Feas%":>6} {"ELBO in-target":>14} {"Cstr in-target":>14} | '
      f'{"ELBO safe":>9} {"Cstr safe":>9} | {"ELBO DBP":>9} {"Cstr DBP":>9}')
print('-' * 100)

mimic_e7_records = []
for tau in MIMIC_TAUS:
    all_results = []
    for seed in mimic_data:
        if tau not in mimic_data[seed]:
            continue
        all_results.extend(
            evaluate_mimic_constrained(mimic_data[seed][tau],
                DBP_TARGET_LOW, DBP_TARGET_HIGH, DBP_SAFETY_LOW, DBP_SAFETY_HIGH))

    if not all_results:
        continue

    df_tmp = pd.DataFrame(all_results)
    feas = df_tmp['feasibility_rate'].mean()
    e_target = df_tmp['elbo_in_target'].mean()
    c_target = df_tmp['cstr_in_target'].mean()
    e_safe = df_tmp['elbo_feasible'].mean()
    c_safe = df_tmp['cstr_feasible'].mean()
    e_dbp = df_tmp['elbo_dbp'].mean()
    c_dbp = df_tmp['cstr_dbp'].mean()

    print(f'{tau:4d} | {feas:6.1%} {e_target:14.1%} {c_target:14.1%} | '
          f'{e_safe:9.1%} {c_safe:9.1%} | {e_dbp:9.1f} {c_dbp:9.1f}')

    mimic_e7_records.append({
        'tau': tau, 'feas_rate': feas,
        'elbo_in_target': e_target, 'cstr_in_target': c_target,
        'elbo_safe': e_safe, 'cstr_safe': c_safe,
        'elbo_dbp_mean': e_dbp, 'cstr_dbp_mean': c_dbp,
    })

df_mimic_e7 = pd.DataFrame(mimic_e7_records) if mimic_e7_records else pd.DataFrame()


# ── Cell 40: Visualization ──
if mimic_e7_records:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # Panel 1: DBP distributions — ELBO vs constrained picks
    ax = axes[0]
    for tau in MIMIC_TAUS:
        elbo_dbps = []
        cstr_dbps = []
        for seed in mimic_data:
            if tau not in mimic_data[seed]:
                continue
            results = evaluate_mimic_constrained(mimic_data[seed][tau],
                DBP_TARGET_LOW, DBP_TARGET_HIGH, DBP_SAFETY_LOW, DBP_SAFETY_HIGH)
            elbo_dbps.extend([r['elbo_dbp'] for r in results])
            cstr_dbps.extend([r['cstr_dbp'] for r in results])

        if tau == 4:  # show one representative tau
            ax.hist(elbo_dbps, bins=30, alpha=0.5, label='ELBO', color='#1f77b4', density=True)
            ax.hist(cstr_dbps, bins=30, alpha=0.5, label='Constrained', color='#ff7f0e', density=True)

    ax.axvline(DBP_TARGET_LOW, color='g', linestyle='--', alpha=0.7, label=f'Target [{DBP_TARGET_LOW},{DBP_TARGET_HIGH}]')
    ax.axvline(DBP_TARGET_HIGH, color='g', linestyle='--', alpha=0.7)
    ax.axvline(DBP_SAFETY_LOW, color='r', linestyle=':', alpha=0.5)
    ax.axvline(DBP_SAFETY_HIGH, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Predicted Diastolic BP (mmHg)')
    ax.set_ylabel('Density')
    ax.set_title('Selected DBP Distribution (τ=4)')
    ax.legend(fontsize=9)

    # Panel 2: In-target rate by tau
    ax = axes[1]
    taus_arr = [r['tau'] for r in mimic_e7_records]
    e_targets = [r['elbo_in_target'] * 100 for r in mimic_e7_records]
    c_targets = [r['cstr_in_target'] * 100 for r in mimic_e7_records]
    ax.plot(taus_arr, e_targets, '--o', label='ELBO', markersize=8)
    ax.plot(taus_arr, c_targets, '-s', label='Constrained', markersize=8)
    ax.set_xlabel('τ (time horizon)')
    ax.set_ylabel('In-target rate (%)')
    ax.set_title('Clinical Target Attainment')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)

    # Panel 3: Feasibility by tau
    ax = axes[2]
    feas_rates = [r['feas_rate'] * 100 for r in mimic_e7_records]
    ax.bar(taus_arr, feas_rates, width=0.6, color='#2ca02c', alpha=0.7)
    ax.set_xlabel('τ (time horizon)')
    ax.set_ylabel('Feasibility rate (%)')
    ax.set_title(f'Feasibility (DBP ∈ [{DBP_TARGET_LOW},{DBP_TARGET_HIGH}])')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    fig_path = f'{FIG_DIR}/e7_mimic_constrained_selection.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {fig_path}')


# ── Cell 41: Cross-seed stability ──
if len(mimic_data) >= 3:
    print('\n=== E7: Cross-Seed Ranking Stability (MIMIC) ===')
    print()

    for tau in MIMIC_TAUS:
        elbo_ranks_by_indiv = {}

        for seed in mimic_data:
            if tau not in mimic_data[seed]:
                continue
            for ci in mimic_data[seed][tau]:
                iid = ci['individual_id']
                elbo_rank = ci['true_sequence_rank']
                elbo_ranks_by_indiv.setdefault(iid, []).append(elbo_rank)

        elbo_stds = [np.std(ranks) for ranks in elbo_ranks_by_indiv.values() if len(ranks) >= 3]

        if elbo_stds:
            print(f'  tau={tau}: ELBO rank std = {np.mean(elbo_stds):.1f} '
                  f'(median={np.median(elbo_stds):.1f}, max={np.max(elbo_stds):.1f})')

    print()
    print('Note: Constrained ranking stability requires comparing selected treatment')
    print('plans across seeds, which depends on feasibility consistency.')


# ── Cell 42: Summary ──
if mimic_e7_records:
    print()
    print('=' * 70)
    print('E7 MIMIC-III CONSTRAINED SELECTION: KEY FINDINGS')
    print('=' * 70)
    print()
    print(f'Clinical thresholds: target DBP ∈ [{DBP_TARGET_LOW}, {DBP_TARGET_HIGH}] mmHg, '
          f'safety DBP ∈ [{DBP_SAFETY_LOW}, {DBP_SAFETY_HIGH}] mmHg')
    print()

    for r in mimic_e7_records:
        improvement = r['cstr_in_target'] - r['elbo_in_target']
        print(f'  tau={r["tau"]}: feasibility={r["feas_rate"]:.1%}, '
              f'in-target: ELBO={r["elbo_in_target"]:.1%} -> constrained={r["cstr_in_target"]:.1%} '
              f'({improvement:+.1%}), DBP: {r["elbo_dbp_mean"]:.1f} -> {r["cstr_dbp_mean"]:.1f} mmHg')

    print()
    print('INTERPRETATION:')
    print('  Without ground truth, we evaluate clinical plausibility:')
    print('  1. Does constrained selection push predicted DBP into clinically safe range?')
    print('  2. Is the feasibility rate reasonable (not too restrictive)?')
    print('  3. Do selected treatment plans differ in clinically meaningful ways?')

    # Additional: Treatment pattern comparison
    print()
    print('=== Treatment Pattern Comparison (all seeds pooled) ===')
    for tau in MIMIC_TAUS:
        all_results = []
        for seed in mimic_data:
            if tau not in mimic_data[seed]:
                continue
            all_results.extend(
                evaluate_mimic_constrained(mimic_data[seed][tau],
                    DBP_TARGET_LOW, DBP_TARGET_HIGH, DBP_SAFETY_LOW, DBP_SAFETY_HIGH))
        if all_results:
            df_t = pd.DataFrame(all_results)
            print(f'  tau={tau}: vaso (ELBO={df_t["elbo_vaso"].mean():.2f}, '
                  f'Cstr={df_t["cstr_vaso"].mean():.2f}), '
                  f'vent (ELBO={df_t["elbo_vent"].mean():.2f}, '
                  f'Cstr={df_t["cstr_vent"].mean():.2f})')
