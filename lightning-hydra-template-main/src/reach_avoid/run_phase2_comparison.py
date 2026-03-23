#!/usr/bin/env python3
"""Phase 2 Comparison: VCIP_RA vs Vanilla VCIP under RA-constrained selection.

Compares:
1. GRP/RCS (ranking quality) — already similar, this confirms
2. RA feasibility rate — fraction of candidates in target+safety set
3. Constrained Top-1 accuracy — best feasible sequence vs ground truth
4. Safety violation rate — fraction of Top-1 picks violating safety
5. Treatment pattern differences
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', font_scale=1.2)

PROJECT_ROOT = Path('/Users/anisiomlacerda/code/target-counterfactual')
FIG_DIR = PROJECT_ROOT / 'lightning-hydra-template-main' / 'src' / 'reach_avoid' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [10, 101, 1010, 10101, 101010]
TAUS = [2, 4, 6, 8]
GAMMAS = [1, 4]

# RA thresholds (from Phase 1 analysis — calibrated for cancer simulator)
TARGET_UPPER = 3.0
SAFETY_VOL_UPPER = 12.0
SAFETY_CHEMO_UPPER = 5.0

# ── Load data ──
def load_results(base_path, model_key, gamma, seeds, taus):
    """Load case_infos for all seeds, return dict[seed][tau] = list of case_infos."""
    results = {}
    for seed in seeds:
        pkl_path = base_path / str(seed) / 'False' / f'case_infos_{model_key}.pkl'
        if not pkl_path.exists():
            print(f'  Missing: {pkl_path}')
            continue
        with open(pkl_path, 'rb') as f:
            raw = pickle.load(f)
        data = raw[model_key]
        results[seed] = {tau: data[tau] for tau in taus if tau in data}
    return results


def evaluate_ra_constrained(case_infos, target_upper, safety_vol_upper,
                             safety_chemo_upper=None):
    """Evaluate RA-constrained selection on cancer case_infos with ground truth."""
    results = []
    for ci in case_infos:
        tf = ci.get('traj_features', {})
        if not tf:
            continue
        elbos = ci['model_losses']
        true_losses = ci['true_losses']
        k = len(elbos)

        cv_terminal = tf['cv_terminal']
        cv_max = tf['cv_max']
        cd_max = tf.get('cd_max', np.zeros(k))

        # Feasibility: terminal < target AND max_vol < safety AND max_chemo < safety
        feasible = (cv_terminal <= target_upper) & (cv_max <= safety_vol_upper)
        if safety_chemo_upper is not None and len(cd_max) == k:
            feasible = feasible & (cd_max <= safety_chemo_upper)

        n_feasible = int(feasible.sum())
        feas_rate = n_feasible / k

        # Ground truth best
        best_true_idx = int(np.argmin(true_losses))

        # Unconstrained ELBO pick
        best_elbo_idx = int(np.argmin(elbos))
        elbo_top1 = int(best_elbo_idx == best_true_idx)
        elbo_true_loss = float(true_losses[best_elbo_idx])
        elbo_feasible = bool(feasible[best_elbo_idx])
        elbo_cv_terminal = float(cv_terminal[best_elbo_idx])

        # Constrained ELBO pick
        if n_feasible > 0:
            feasible_elbos = np.where(feasible, elbos, np.inf)
            cstr_idx = int(np.argmin(feasible_elbos))
            cstr_top1 = int(cstr_idx == best_true_idx)
            cstr_true_loss = float(true_losses[cstr_idx])
            cstr_feasible = True
            cstr_cv_terminal = float(cv_terminal[cstr_idx])
        else:
            # Fallback to unconstrained
            cstr_idx = best_elbo_idx
            cstr_top1 = elbo_top1
            cstr_true_loss = elbo_true_loss
            cstr_feasible = False
            cstr_cv_terminal = elbo_cv_terminal

        # True best's feasibility
        true_best_feasible = bool(feasible[best_true_idx])

        # Safety violation of unconstrained pick
        elbo_safety_violation = (float(cv_max[best_elbo_idx]) > safety_vol_upper)

        results.append({
            'individual_id': ci['individual_id'],
            'feasibility_rate': feas_rate,
            'n_feasible': n_feasible,
            'elbo_top1': elbo_top1,
            'cstr_top1': cstr_top1,
            'elbo_true_loss': elbo_true_loss,
            'cstr_true_loss': cstr_true_loss,
            'elbo_feasible': elbo_feasible,
            'cstr_feasible': cstr_feasible,
            'elbo_cv_terminal': elbo_cv_terminal,
            'cstr_cv_terminal': cstr_cv_terminal,
            'elbo_safety_violation': elbo_safety_violation,
            'true_best_feasible': true_best_feasible,
        })
    return results


print('=' * 70)
print('Phase 2 Comparison: VCIP_RA vs Vanilla VCIP')
print('=' * 70)
print(f'RA thresholds: target_upper={TARGET_UPPER}, safety_vol={SAFETY_VOL_UPPER}, '
      f'safety_chemo={SAFETY_CHEMO_UPPER}')
print()

# Load vanilla VCIP results
vanilla_base = PROJECT_ROOT / 'results_remote' / 'phase1_ra_v2' / 'my_outputs' / 'cancer_sim_cont' / '22'
ra_base = PROJECT_ROOT / 'results_remote' / 'phase2'

all_records = []

for gamma in GAMMAS:
    print(f'\n{"="*60}')
    print(f'  GAMMA = {gamma}')
    print(f'{"="*60}')

    vanilla_data = load_results(
        vanilla_base / f'coeff_{gamma}' / 'VCIP' / 'train' / 'True' / 'case_infos',
        'VCIP', gamma, SEEDS, TAUS)
    ra_data = load_results(
        ra_base / f'coeff_{gamma}' / 'train' / 'True' / 'case_infos',
        'VCIP_RA', gamma, SEEDS, TAUS)

    print(f'  Loaded: vanilla={len(vanilla_data)} seeds, RA={len(ra_data)} seeds')

    print(f'\n  {"tau":>3} | {"Method":>10} | {"GRP":>6} {"Top1":>5} {"Feas%":>6} '
          f'{"Cstr-Top1":>9} {"ELBO-safe":>9} {"Cstr-safe":>9} '
          f'{"ELBO-cv_t":>9} {"Cstr-cv_t":>9}')
    print(f'  ' + '-' * 95)

    for tau in TAUS:
        for label, data, model_key in [('Vanilla', vanilla_data, 'VCIP'),
                                        ('VCIP_RA', ra_data, 'VCIP_RA')]:
            grps, top1s, feas_rates = [], [], []
            cstr_top1s, elbo_safes, cstr_safes = [], [], []
            elbo_cvts, cstr_cvts = [], []

            for seed in SEEDS:
                if seed not in data or tau not in data[seed]:
                    continue
                case_infos = data[seed][tau]

                # GRP
                ranks = [c['true_sequence_rank'] for c in case_infos]
                grps.append(1.0 - np.mean(ranks) / 100.0)

                # RA evaluation
                ra_results = evaluate_ra_constrained(
                    case_infos, TARGET_UPPER, SAFETY_VOL_UPPER, SAFETY_CHEMO_UPPER)

                if ra_results:
                    df_r = pd.DataFrame(ra_results)
                    top1s.append(df_r['elbo_top1'].mean())
                    feas_rates.append(df_r['feasibility_rate'].mean())
                    cstr_top1s.append(df_r['cstr_top1'].mean())
                    elbo_safes.append(df_r['elbo_feasible'].mean())
                    cstr_safes.append(df_r['cstr_feasible'].mean())
                    elbo_cvts.append(df_r['elbo_cv_terminal'].mean())
                    cstr_cvts.append(df_r['cstr_cv_terminal'].mean())

            if grps:
                rec = {
                    'gamma': gamma, 'tau': tau, 'method': label,
                    'grp': np.mean(grps), 'grp_std': np.std(grps),
                    'top1': np.mean(top1s) if top1s else 0,
                    'feas_rate': np.mean(feas_rates) if feas_rates else 0,
                    'cstr_top1': np.mean(cstr_top1s) if cstr_top1s else 0,
                    'elbo_safe': np.mean(elbo_safes) if elbo_safes else 0,
                    'cstr_safe': np.mean(cstr_safes) if cstr_safes else 0,
                    'elbo_cv_t': np.mean(elbo_cvts) if elbo_cvts else 0,
                    'cstr_cv_t': np.mean(cstr_cvts) if cstr_cvts else 0,
                }
                all_records.append(rec)
                print(f'  {tau:3d} | {label:>10} | {rec["grp"]:.3f} '
                      f'{rec["top1"]:5.1%} {rec["feas_rate"]:6.1%} '
                      f'{rec["cstr_top1"]:9.1%} {rec["elbo_safe"]:9.1%} {rec["cstr_safe"]:9.1%} '
                      f'{rec["elbo_cv_t"]:9.2f} {rec["cstr_cv_t"]:9.2f}')

df = pd.DataFrame(all_records)

# ── Visualization ──
fig, axes = plt.subplots(2, 3, figsize=(20, 11))

for row, gamma in enumerate(GAMMAS):
    df_g = df[df.gamma == gamma]

    # Panel 1: GRP comparison
    ax = axes[row, 0]
    for method, marker, ls in [('Vanilla', 'o', '--'), ('VCIP_RA', 's', '-')]:
        d = df_g[df_g.method == method]
        ax.plot(d.tau, d.grp, f'{ls}{marker}', label=method, markersize=8)
    ax.set_xlabel('τ')
    ax.set_ylabel('GRP')
    ax.set_title(f'Ranking Quality (γ={gamma})')
    ax.legend()
    ax.set_ylim(0.5 if gamma == 1 else 0.85, 1.02)

    # Panel 2: Feasibility + Constrained safety
    ax = axes[row, 1]
    for method, marker, ls in [('Vanilla', 'o', '--'), ('VCIP_RA', 's', '-')]:
        d = df_g[df_g.method == method]
        ax.plot(d.tau, d.feas_rate * 100, f'{ls}{marker}', label=f'{method} feas%', markersize=8)
        ax.plot(d.tau, d.cstr_safe * 100, f'{ls}{marker}', label=f'{method} cstr-safe%',
                markersize=6, alpha=0.6)
    ax.set_xlabel('τ')
    ax.set_ylabel('%')
    ax.set_title(f'Feasibility & Safety (γ={gamma})')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)

    # Panel 3: Top-1 accuracy — ELBO vs constrained
    ax = axes[row, 2]
    x = np.arange(len(TAUS))
    w = 0.2
    for i, (method, color) in enumerate([('Vanilla', '#1f77b4'), ('VCIP_RA', '#ff7f0e')]):
        d = df_g[df_g.method == method]
        ax.bar(x + (i - 0.5) * w * 2 - w/2, d.top1.values * 100, w,
               label=f'{method} ELBO', color=color, alpha=0.5)
        ax.bar(x + (i - 0.5) * w * 2 + w/2, d.cstr_top1.values * 100, w,
               label=f'{method} Cstr', color=color, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f'τ={t}' for t in TAUS])
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title(f'Top-1: ELBO vs Constrained (γ={gamma})')
    ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
fig_path = FIG_DIR / 'phase2_vcip_ra_comparison.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: {fig_path}')

# ── Summary ──
print()
print('=' * 70)
print('PHASE 2 COMPARISON SUMMARY')
print('=' * 70)

for gamma in GAMMAS:
    print(f'\n  gamma={gamma}:')
    df_g = df[df.gamma == gamma]
    for tau in TAUS:
        van = df_g[(df_g.method == 'Vanilla') & (df_g.tau == tau)]
        ra = df_g[(df_g.method == 'VCIP_RA') & (df_g.tau == tau)]
        if van.empty or ra.empty:
            continue
        v, r = van.iloc[0], ra.iloc[0]
        grp_diff = r.grp - v.grp
        feas_diff = r.feas_rate - v.feas_rate
        cstr_safe_diff = r.cstr_safe - v.cstr_safe
        cstr_top1_diff = r.cstr_top1 - v.cstr_top1
        print(f'    tau={tau}: GRP {v.grp:.3f}→{r.grp:.3f} ({grp_diff:+.3f}), '
              f'Feas {v.feas_rate:.1%}→{r.feas_rate:.1%} ({feas_diff:+.1%}), '
              f'Cstr-safe {v.cstr_safe:.1%}→{r.cstr_safe:.1%} ({cstr_safe_diff:+.1%}), '
              f'Cstr-Top1 {v.cstr_top1:.1%}→{r.cstr_top1:.1%} ({cstr_top1_diff:+.1%})')

print()
print('INTERPRETATION:')
print('  - GRP: Does RA-aware retraining maintain ranking quality?')
print('  - Feasibility: Does RA-aware training produce more feasible candidates?')
print('  - Cstr-safe: Does constrained selection achieve higher safety?')
print('  - Cstr-Top1: Does constrained selection maintain accuracy?')
