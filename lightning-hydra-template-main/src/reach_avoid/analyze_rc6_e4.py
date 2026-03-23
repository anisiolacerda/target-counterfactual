#!/usr/bin/env python3
"""Analyze RC6 (intermediate prediction quality) + E4 (VCI diagnostic) results.

Reads pickle files from results_remote/rc6_e4/ and produces:
- Aggregate statistics across seeds with confidence intervals
- Figures for RC6 per-step prediction quality and E4 KL trends
- Markdown summary for RESULTS.md
"""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('/Users/anisiomlacerda/code/target-counterfactual')
RESULTS_DIR = PROJECT_ROOT / 'results_remote' / 'rc6_e4'
FIG_DIR = PROJECT_ROOT / 'lightning-hydra-template-main' / 'src' / 'reach_avoid' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [10, 101, 1010, 10101, 101010]
GAMMA = 4
TAUS = [2, 4, 6, 8]


def load_all_results():
    """Load all seed results into {seed: {tau: [case_results]}}."""
    all_data = {}
    for seed in SEEDS:
        path = RESULTS_DIR / f'rc6_e4_gamma{GAMMA}_seed{seed}.pkl'
        with open(path, 'rb') as f:
            all_data[seed] = pickle.load(f)
    return all_data


def analyze_rc6(all_data):
    """RC6: Intermediate prediction quality analysis."""
    print("\n" + "="*70)
    print("  RC6: INTERMEDIATE PREDICTION QUALITY")
    print("="*70)

    # Per-tau, per-step MSE aggregated across seeds
    rc6_stats = {}
    for tau in TAUS:
        seed_mses = []  # per-seed average MSE at each step
        seed_corrs = []  # per-seed Spearman correlation at each step

        for seed in SEEDS:
            data = all_data[seed][tau]
            n_individuals = len(data)

            # Per-step MSE across individuals (averaged over sequences)
            step_mses = np.zeros(tau)
            step_counts = np.zeros(tau)
            for c in data:
                for r in c['rc6']:
                    losses = r['per_step_losses']
                    for s in range(min(len(losses), tau)):
                        step_mses[s] += losses[s]
                        step_counts[s] += 1
            step_mses /= np.maximum(step_counts, 1)
            seed_mses.append(step_mses)

            # Per-step Spearman rank correlation (scale-invariant)
            step_corrs = []
            for s in range(tau):
                preds_s = []
                gts_s = []
                for c in data:
                    r = c['rc6'][0]  # first sequence
                    if len(r['predictions']) > s and r['gt_trajectory'] is not None and len(r['gt_trajectory']) > s:
                        preds_s.append(float(r['predictions'][s]))
                        gts_s.append(float(r['gt_trajectory'][s]))
                if len(preds_s) > 5:
                    try:
                        rho, _ = spearmanr(preds_s, gts_s)
                        step_corrs.append(rho if not np.isnan(rho) else 0.0)
                    except Exception:
                        step_corrs.append(0.0)
                else:
                    step_corrs.append(0.0)
            seed_corrs.append(step_corrs)

        seed_mses = np.array(seed_mses)  # (5, tau)
        seed_corrs = np.array(seed_corrs)  # (5, tau)

        rc6_stats[tau] = {
            'mse_mean': seed_mses.mean(axis=0),
            'mse_std': seed_mses.std(axis=0),
            'corr_mean': seed_corrs.mean(axis=0),
            'corr_std': seed_corrs.std(axis=0),
            'overall_mse_mean': seed_mses.mean(),
            'overall_mse_std': seed_mses.mean(axis=1).std(),
            'overall_corr_mean': seed_corrs.mean(),
        }

        print(f"\n  tau={tau}:")
        print(f"    Per-step MSE (mean±std across seeds):")
        for s in range(tau):
            print(f"      s={s+1}: {rc6_stats[tau]['mse_mean'][s]:.6f} ± {rc6_stats[tau]['mse_std'][s]:.6f}")
        print(f"    Overall MSE: {rc6_stats[tau]['overall_mse_mean']:.6f} ± {rc6_stats[tau]['overall_mse_std']:.6f}")
        print(f"    Per-step Spearman ρ (pred vs GT, across individuals):")
        for s in range(tau):
            print(f"      s={s+1}: {rc6_stats[tau]['corr_mean'][s]:.4f} ± {rc6_stats[tau]['corr_std'][s]:.4f}")

    # Check prediction variance across sequences (do different action sequences yield different predictions?)
    print("\n  Prediction variance across action sequences (per individual):")
    for tau in TAUS:
        variances = []
        for seed in SEEDS:
            for c in all_data[seed][tau]:
                # Terminal predictions across the 10 sequences
                terminal_preds = [r['predictions'][-1] for r in c['rc6']]
                variances.append(np.var(terminal_preds))
        print(f"    tau={tau}: mean var={np.mean(variances):.8f}, max var={np.max(variances):.8f}")

    return rc6_stats


def analyze_e4(all_data):
    """E4: VCI consistency diagnostic — DKL[q(Z_s|a_obs) || q(Z_s|a_alt)]."""
    print("\n" + "="*70)
    print("  E4: VCI CONSISTENCY DIAGNOSTIC (LATENT ACTION-SENSITIVITY)")
    print("="*70)

    e4_stats = {}
    for tau in TAUS:
        seed_mean_kls = []
        seed_per_step_kls = []

        for seed in SEEDS:
            data = all_data[seed][tau]
            mean_kls = [c['e4']['mean_kl'] for c in data]
            seed_mean_kls.append(np.mean(mean_kls))

            # Per-step KL
            per_step = np.array([c['e4']['per_step_kl'] for c in data])  # (n_individuals, tau)
            seed_per_step_kls.append(per_step.mean(axis=0))

        seed_mean_kls = np.array(seed_mean_kls)
        seed_per_step_kls = np.array(seed_per_step_kls)  # (5, tau)

        e4_stats[tau] = {
            'kl_mean': seed_mean_kls.mean(),
            'kl_std': seed_mean_kls.std(),
            'per_step_kl_mean': seed_per_step_kls.mean(axis=0),
            'per_step_kl_std': seed_per_step_kls.std(axis=0),
        }

        print(f"\n  tau={tau}:")
        print(f"    Mean KL: {e4_stats[tau]['kl_mean']:.6f} ± {e4_stats[tau]['kl_std']:.6f}")
        print(f"    Per-step KL:")
        for s in range(tau):
            print(f"      s={s+1}: {e4_stats[tau]['per_step_kl_mean'][s]:.6f} ± {e4_stats[tau]['per_step_kl_std'][s]:.6f}")

    # Interpretation
    print("\n  Interpretation:")
    print("    KL ≈ 0 means the posterior q(Z_s|history, actions) barely changes")
    print("    when the action sequence is swapped — the latent space is nearly")
    print("    action-invariant. This means:")
    print("    (1) The model encodes outcome information primarily in Z, not in a")
    print("    (2) ELBO differences between action sequences come from the decoder,")
    print("        not from latent-space separation")
    print("    (3) RA filtering works because the *decoder* output (predicted Y)")
    print("        does depend on actions, even though the *latent* doesn't")

    return e4_stats


def create_figures(rc6_stats, e4_stats):
    """Create publication-quality figures."""

    # Figure 1: RC6 per-step MSE by tau
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: MSE by step
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(TAUS)))
    for i, tau in enumerate(TAUS):
        steps = np.arange(1, tau + 1)
        means = rc6_stats[tau]['mse_mean']
        stds = rc6_stats[tau]['mse_std']
        ax.plot(steps, means, 'o-', color=colors[i], label=f'τ={tau}', linewidth=2, markersize=5)
        ax.fill_between(steps, means - stds, means + stds, alpha=0.15, color=colors[i])
    ax.set_xlabel('Step s (within horizon)', fontsize=12)
    ax.set_ylabel('MSE (model space)', fontsize=12)
    ax.set_title('RC6: Per-Step Prediction MSE', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(range(1, max(TAUS) + 1))
    ax.grid(alpha=0.3)

    # Right: Spearman correlation by step
    ax = axes[1]
    for i, tau in enumerate(TAUS):
        steps = np.arange(1, tau + 1)
        means = rc6_stats[tau]['corr_mean']
        stds = rc6_stats[tau]['corr_std']
        ax.plot(steps, means, 'o-', color=colors[i], label=f'τ={tau}', linewidth=2, markersize=5)
        ax.fill_between(steps, means - stds, means + stds, alpha=0.15, color=colors[i])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step s (within horizon)', fontsize=12)
    ax.set_ylabel('Spearman ρ (pred vs GT)', fontsize=12)
    ax.set_title('RC6: Pred-GT Rank Correlation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(range(1, max(TAUS) + 1))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / 'rc6_intermediate_predictions.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {fig_path}')

    # Figure 2: E4 per-step KL by tau
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Per-step KL
    ax = axes[0]
    for i, tau in enumerate(TAUS):
        steps = np.arange(1, tau + 1)
        means = e4_stats[tau]['per_step_kl_mean']
        stds = e4_stats[tau]['per_step_kl_std']
        ax.plot(steps, means, 'o-', color=colors[i], label=f'τ={tau}', linewidth=2, markersize=5)
        ax.fill_between(steps, means - stds, means + stds, alpha=0.15, color=colors[i])
    ax.set_xlabel('Step s (within horizon)', fontsize=12)
    ax.set_ylabel('D_KL [q(Z|a_obs) || q(Z|a_alt)]', fontsize=12)
    ax.set_title('E4: Per-Step Latent Action-Sensitivity', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(range(1, max(TAUS) + 1))
    ax.grid(alpha=0.3)

    # Right: Mean KL by tau (bar plot)
    ax = axes[1]
    tau_labels = [str(t) for t in TAUS]
    kl_means = [e4_stats[t]['kl_mean'] for t in TAUS]
    kl_stds = [e4_stats[t]['kl_std'] for t in TAUS]
    bars = ax.bar(tau_labels, kl_means, yerr=kl_stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Horizon τ', fontsize=12)
    ax.set_ylabel('Mean D_KL', fontsize=12)
    ax.set_title('E4: Mean Latent KL by Horizon', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, m in zip(bars, kl_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{m:.2e}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig_path = FIG_DIR / 'e4_vci_diagnostic.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fig_path}')


def print_summary_table(rc6_stats, e4_stats):
    """Print markdown summary table."""
    print("\n" + "="*70)
    print("  SUMMARY TABLE (for RESULTS.md)")
    print("="*70)

    print("\n### RC6: Per-Step MSE (×10⁻³, model space)")
    print("| τ | s=1 | s=2 | s=3 | s=4 | s=5 | s=6 | s=7 | s=8 | Overall |")
    print("|---|-----|-----|-----|-----|-----|-----|-----|-----|---------|")
    for tau in TAUS:
        row = f"| {tau} |"
        for s in range(8):
            if s < tau:
                row += f" {rc6_stats[tau]['mse_mean'][s]*1000:.2f} |"
            else:
                row += " — |"
        row += f" {rc6_stats[tau]['overall_mse_mean']*1000:.2f} |"
        print(row)

    print("\n### RC6: Spearman ρ (pred rank vs GT rank, across individuals)")
    print("| τ | s=1 | s=2 | s=3 | s=4 | s=5 | s=6 | s=7 | s=8 |")
    print("|---|-----|-----|-----|-----|-----|-----|-----|-----|")
    for tau in TAUS:
        row = f"| {tau} |"
        for s in range(8):
            if s < tau:
                row += f" {rc6_stats[tau]['corr_mean'][s]:.3f} |"
            else:
                row += " — |"
        print(row)

    print("\n### E4: Latent Action-Sensitivity (D_KL)")
    print("| τ | Mean KL | s=1 | s=2 | s=3 | s=4 | s=5 | s=6 | s=7 | s=8 |")
    print("|---|---------|-----|-----|-----|-----|-----|-----|-----|-----|")
    for tau in TAUS:
        row = f"| {tau} | {e4_stats[tau]['kl_mean']:.2e} |"
        for s in range(8):
            if s < tau:
                row += f" {e4_stats[tau]['per_step_kl_mean'][s]:.2e} |"
            else:
                row += " — |"
        print(row)


def main():
    all_data = load_all_results()
    print(f"Loaded results for {len(SEEDS)} seeds, gamma={GAMMA}")

    rc6_stats = analyze_rc6(all_data)
    e4_stats = analyze_e4(all_data)
    create_figures(rc6_stats, e4_stats)
    print_summary_table(rc6_stats, e4_stats)


if __name__ == '__main__':
    main()
