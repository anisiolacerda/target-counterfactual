#!/usr/bin/env python3
"""RC7: Create Figure 1 — Reach-Avoid Concept Visualization.

Shows patient trajectories entering target set T while staying in safety set S.
Illustrates the difference between point-target ELBO and RA-constrained selection.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from pathlib import Path

PROJECT_ROOT = Path('/Users/anisiomlacerda/code/target-counterfactual')
FIG_DIR = PROJECT_ROOT / 'lightning-hydra-template-main' / 'src' / 'reach_avoid' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ── Parameters ──
tau = 8  # horizon
T_lower, T_upper = 2.0, 5.0  # target set
S_lower, S_upper = 0.0, 10.0  # safety set
y_target = (T_lower + T_upper) / 2  # midpoint target for ELBO

timesteps = np.arange(tau + 1)

def generate_trajectory(start, trend, noise_std=0.3, n_steps=tau):
    """Generate a smooth trajectory with noise."""
    t = np.linspace(0, 1, n_steps + 1)
    base = start + trend * t
    noise = np.cumsum(np.random.randn(n_steps + 1) * noise_std) * 0.3
    return base + noise


# ── Generate example trajectories ──
# Trajectory A: reaches target, stays safe — GOOD (RA selects this)
traj_a = generate_trajectory(7.5, -4.5, noise_std=0.15)
traj_a = np.clip(traj_a, 1.0, 9.5)
traj_a[-1] = 3.5  # ends in target

# Trajectory B: reaches target but violates safety mid-way — BAD
traj_b = generate_trajectory(8.0, -5.0, noise_std=0.4)
traj_b[3] = 10.5  # safety violation at step 3
traj_b[4] = 11.0  # safety violation at step 4
traj_b[-1] = 3.2  # ends in target (better terminal than A!)

# Trajectory C: misses target entirely — BAD
traj_c = generate_trajectory(7.0, -1.0, noise_std=0.2)
traj_c[-1] = 6.2  # ends outside target

# Trajectory D: best ELBO (closest to midpoint at terminal) but unsafe path
traj_d = generate_trajectory(9.0, -6.0, noise_std=0.3)
traj_d[2] = 10.8  # brief safety violation
traj_d[-1] = 3.0  # closest to midpoint — ELBO would pick this

# Smooth trajectories
from scipy.ndimage import gaussian_filter1d
traj_a = gaussian_filter1d(traj_a, sigma=0.5)
traj_b_smooth = traj_b.copy()
traj_b_smooth = gaussian_filter1d(traj_b_smooth, sigma=0.3)
# Keep the safety violations visible
traj_b_smooth[3] = 10.5
traj_b_smooth[4] = 11.0
traj_b_smooth[-1] = 3.2

traj_c = gaussian_filter1d(traj_c, sigma=0.5)

traj_d_smooth = traj_d.copy()
traj_d_smooth = gaussian_filter1d(traj_d_smooth, sigma=0.3)
traj_d_smooth[2] = 10.8
traj_d_smooth[-1] = 3.0

# ── Figure ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

for ax_idx, (ax, title) in enumerate(zip(axes, [
    'Point-Target ELBO Selection',
    'Reach-Avoid Constrained Selection'
])):
    # Safety region (background)
    ax.axhspan(S_lower, S_upper, alpha=0.06, color='blue', label='Safety set S')
    ax.axhspan(S_upper, 13, alpha=0.08, color='red')
    ax.axhspan(-1, S_lower, alpha=0.08, color='red')

    # Target region
    ax.axhspan(T_lower, T_upper, alpha=0.15, color='green', label='Target set T')

    # Safety boundaries
    ax.axhline(y=S_upper, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axhline(y=S_lower, color='red', linestyle='--', alpha=0.6, linewidth=1.5)

    # Target boundaries
    ax.axhline(y=T_upper, color='green', linestyle='-', alpha=0.4, linewidth=1.5)
    ax.axhline(y=T_lower, color='green', linestyle='-', alpha=0.4, linewidth=1.5)

    # Midpoint target line
    ax.axhline(y=y_target, color='darkgreen', linestyle=':', alpha=0.5, linewidth=1)

    # Plot trajectories
    trajectories = [
        (traj_a, 'Seq A: Safe path to T', '#2ca02c', '-', 2.5),
        (traj_b_smooth, 'Seq B: Reaches T, violates S', '#d62728', '-', 2.0),
        (traj_c, 'Seq C: Misses T', '#7f7f7f', '--', 1.5),
        (traj_d_smooth, 'Seq D: Best ELBO, violates S', '#ff7f0e', '-', 2.0),
    ]

    for traj, label, color, ls, lw in trajectories:
        ax.plot(timesteps, traj, ls, color=color, linewidth=lw, label=label,
                marker='o', markersize=4, alpha=0.85)

    # Mark safety violations
    for traj, _, color, _, _ in trajectories:
        for t_idx in range(len(traj)):
            if traj[t_idx] > S_upper or traj[t_idx] < S_lower:
                ax.plot(t_idx, traj[t_idx], 'x', color='red', markersize=12,
                        markeredgewidth=2.5, zorder=5)

    # Highlight selected sequence
    if ax_idx == 0:  # ELBO panel — selects D (closest to midpoint)
        ax.plot(tau, traj_d_smooth[-1], '*', color='#ff7f0e', markersize=20,
                zorder=6, markeredgecolor='black', markeredgewidth=1)
        ax.annotate('ELBO picks D\n(best loss, unsafe path)',
                    xy=(tau, traj_d_smooth[-1]), xytext=(tau - 2.5, 1.2),
                    fontsize=9, fontweight='bold', color='#ff7f0e',
                    arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5))
    else:  # RA panel — selects A (feasible with best ELBO)
        ax.plot(tau, traj_a[-1], '*', color='#2ca02c', markersize=20,
                zorder=6, markeredgecolor='black', markeredgewidth=1)
        ax.annotate('RA picks A\n(best ELBO in feasible set)',
                    xy=(tau, traj_a[-1]), xytext=(tau - 3.0, 1.2),
                    fontsize=9, fontweight='bold', color='#2ca02c',
                    arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5))

        # Cross out infeasible sequences
        ax.plot(tau, traj_b_smooth[-1], 'x', color='red', markersize=15,
                markeredgewidth=2, zorder=6)
        ax.plot(tau, traj_d_smooth[-1], 'x', color='red', markersize=15,
                markeredgewidth=2, zorder=6)

    # Labels
    ax.set_xlabel('Timestep s', fontsize=12)
    ax.set_ylabel('Outcome Y_s', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(-0.3, tau + 0.3)
    ax.set_ylim(-0.5, 12.5)
    ax.set_xticks(timesteps)
    ax.set_xticklabels([f't' if i == 0 else f't+{i}' for i in range(tau + 1)],
                        fontsize=8)

    # Add text labels for regions
    ax.text(0.3, S_upper + 0.4, 'Unsafe (Y > S_upper)', fontsize=8,
            color='red', alpha=0.7)
    ax.text(0.3, (T_lower + T_upper) / 2, 'Target T', fontsize=9,
            color='green', fontweight='bold', alpha=0.7,
            ha='left', va='center')
    ax.text(tau + 0.1, y_target, 'midpoint', fontsize=7, color='darkgreen',
            alpha=0.6, va='center')

    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

plt.tight_layout()
fig_path = FIG_DIR / 'rc7_figure1_concept.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {fig_path}')

# Also save as PDF for paper
fig_path_pdf = FIG_DIR / 'rc7_figure1_concept.pdf'
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

for ax_idx, (ax, title) in enumerate(zip(axes, [
    'Point-Target ELBO Selection',
    'Reach-Avoid Constrained Selection'
])):
    ax.axhspan(S_lower, S_upper, alpha=0.06, color='blue')
    ax.axhspan(S_upper, 13, alpha=0.08, color='red')
    ax.axhspan(-1, S_lower, alpha=0.08, color='red')
    ax.axhspan(T_lower, T_upper, alpha=0.15, color='green')
    ax.axhline(y=S_upper, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axhline(y=S_lower, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axhline(y=T_upper, color='green', linestyle='-', alpha=0.4, linewidth=1.5)
    ax.axhline(y=T_lower, color='green', linestyle='-', alpha=0.4, linewidth=1.5)
    ax.axhline(y=y_target, color='darkgreen', linestyle=':', alpha=0.5, linewidth=1)

    trajectories = [
        (traj_a, 'Seq A: Safe path to T', '#2ca02c', '-', 2.5),
        (traj_b_smooth, 'Seq B: Reaches T, violates S', '#d62728', '-', 2.0),
        (traj_c, 'Seq C: Misses T', '#7f7f7f', '--', 1.5),
        (traj_d_smooth, 'Seq D: Best ELBO, violates S', '#ff7f0e', '-', 2.0),
    ]

    for traj, label, color, ls, lw in trajectories:
        ax.plot(timesteps, traj, ls, color=color, linewidth=lw, label=label,
                marker='o', markersize=4, alpha=0.85)

    for traj, _, color, _, _ in trajectories:
        for t_idx in range(len(traj)):
            if traj[t_idx] > S_upper or traj[t_idx] < S_lower:
                ax.plot(t_idx, traj[t_idx], 'x', color='red', markersize=12,
                        markeredgewidth=2.5, zorder=5)

    if ax_idx == 0:
        ax.plot(tau, traj_d_smooth[-1], '*', color='#ff7f0e', markersize=20,
                zorder=6, markeredgecolor='black', markeredgewidth=1)
        ax.annotate('ELBO picks D\n(best loss, unsafe path)',
                    xy=(tau, traj_d_smooth[-1]), xytext=(tau - 2.5, 1.2),
                    fontsize=9, fontweight='bold', color='#ff7f0e',
                    arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5))
    else:
        ax.plot(tau, traj_a[-1], '*', color='#2ca02c', markersize=20,
                zorder=6, markeredgecolor='black', markeredgewidth=1)
        ax.annotate('RA picks A\n(best ELBO in feasible set)',
                    xy=(tau, traj_a[-1]), xytext=(tau - 3.0, 1.2),
                    fontsize=9, fontweight='bold', color='#2ca02c',
                    arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5))
        ax.plot(tau, traj_b_smooth[-1], 'x', color='red', markersize=15,
                markeredgewidth=2, zorder=6)
        ax.plot(tau, traj_d_smooth[-1], 'x', color='red', markersize=15,
                markeredgewidth=2, zorder=6)

    ax.set_xlabel('Timestep s', fontsize=12)
    ax.set_ylabel('Outcome Y_s', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(-0.3, tau + 0.3)
    ax.set_ylim(-0.5, 12.5)
    ax.set_xticks(timesteps)
    ax.set_xticklabels([f't' if i == 0 else f't+{i}' for i in range(tau + 1)],
                        fontsize=8)
    ax.text(0.3, S_upper + 0.4, 'Unsafe (Y > S_upper)', fontsize=8,
            color='red', alpha=0.7)
    ax.text(0.3, (T_lower + T_upper) / 2, 'Target T', fontsize=9,
            color='green', fontweight='bold', alpha=0.7, ha='left', va='center')
    ax.text(tau + 0.1, y_target, 'midpoint', fontsize=7, color='darkgreen',
            alpha=0.6, va='center')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig(fig_path_pdf, bbox_inches='tight')
plt.close()
print(f'Saved: {fig_path_pdf}')
