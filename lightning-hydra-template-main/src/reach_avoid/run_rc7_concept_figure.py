#!/usr/bin/env python3
"""RC7: Create Figure 1 — Reach-Avoid Concept Visualization.

Illustrates the difference between point-target ELBO selection and
reach-avoid constrained selection using four candidate treatment sequences.

Design principles for NeurIPS paper figure (will be rendered at \textwidth):
- Large fonts (everything ≥11pt at figure size → readable after scaling)
- Minimal annotations — story told by color/shape, not text
- Clean whitespace, no clutter
- Trajectories carefully spaced to avoid overlap
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path('/Users/anisiomlacerda/code/target-counterfactual')
FIG_DIR = PROJECT_ROOT / 'lightning-hydra-template-main' / 'src' / 'reach_avoid' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8,
})

# ── Layout ──
tau = 5
ts = np.arange(tau + 1)
T_lo, T_hi = 2.0, 4.5
S_lo, S_hi = 0.5, 10.0
Y_STAR = (T_lo + T_hi) / 2

# ── Trajectories (well-separated terminals) ──
traj = {
    'A': np.array([8.2, 7.4, 6.2, 5.0, 4.2, 3.8]),    # safe → target (RA pick)
    'B': np.array([7.5, 8.8, 11.2, 6.8, 4.2, 2.4]),    # violates S at s=2
    'C': np.array([6.8, 7.0, 6.6, 6.3, 5.8, 5.5]),     # misses target
    'D': np.array([9.2, 8.0, 6.8, 10.8, 4.8, 3.2]),    # best terminal, violates S at s=3
}

# ── Visual style ──
STYLE = {
    'A': dict(color='#1a7832', ls='-',  lw=3.0, marker='o', ms=8, zorder=4),
    'B': dict(color='#c4197d', ls='-',  lw=2.2, marker='s', ms=6, zorder=3),
    'C': dict(color='#888888', ls='--', lw=2.0, marker='^', ms=6, zorder=2),
    'D': dict(color='#e66a00', ls='-',  lw=2.5, marker='D', ms=6, zorder=3),
}
LABELS = {
    'A': 'A: safe, reaches target',
    'B': 'B: violates safety',
    'C': 'C: misses target',
    'D': 'D: violates safety',
}


def draw_panel(ax, selected, rejected, annot_label, annot_color,
               annot_xytext=(1.8, 0.5), show_ystar=False):

    # ── Regions ──
    ax.axhspan(S_hi, 12.5, color='#fce4e4', zorder=0)     # unsafe top
    ax.axhspan(-0.5, S_lo,  color='#fce4e4', zorder=0)     # unsafe bottom
    ax.axhspan(T_lo, T_hi,  color='#d4edda', zorder=0)     # target

    # Boundary lines
    ax.axhline(S_hi, color='#cc0000', ls='--', lw=1.0, alpha=0.5, zorder=1)
    ax.axhline(S_lo, color='#cc0000', ls='--', lw=1.0, alpha=0.5, zorder=1)
    ax.axhline(T_hi, color='#28a745', ls='-',  lw=0.8, alpha=0.4, zorder=1)
    ax.axhline(T_lo, color='#28a745', ls='-',  lw=0.8, alpha=0.4, zorder=1)

    if show_ystar:
        ax.axhline(Y_STAR, color='#1a5e28', ls=':', lw=1.0, alpha=0.45, zorder=1)
        ax.text(tau + 0.15, Y_STAR + 0.25, '$Y^*$', fontsize=18,
                color='#1a5e28', va='bottom', ha='left')

    # ── Region labels ──
    ax.text(0.15, (T_lo + T_hi) / 2, r'$\mathcal{T}$',
            fontsize=26, fontweight='bold', color='#1a7832',
            va='center', ha='left', alpha=0.6)
    ax.text(tau + 0.3, (T_hi + S_hi) / 2, r'$\mathcal{S}$',
            fontsize=26, fontweight='bold', color='#555555',
            va='center', ha='left', alpha=0.4)
    ax.text(0.15, 11.5, 'Unsafe',
            fontsize=16, color='#cc0000', va='center', ha='left',
            alpha=0.55, fontweight='bold')

    # ── Plot trajectories ──
    for key in ['C', 'B', 'D', 'A']:  # A on top
        s = STYLE[key]
        ax.plot(ts, traj[key], s['ls'], color=s['color'], linewidth=s['lw'],
                marker=s['marker'], markersize=s['ms'],
                markeredgecolor='white', markeredgewidth=0.8,
                alpha=0.9, zorder=s['zorder'], label=LABELS[key])

    # ── Safety violation markers ──
    for key in traj:
        for i, y in enumerate(traj[key]):
            if y > S_hi or y < S_lo:
                ax.plot(i, y, 'X', color='#cc0000', markersize=12,
                        markeredgewidth=2.5, zorder=5)

    # ── Selected: gold star ──
    y_sel = traj[selected][-1]
    ax.plot(tau, y_sel, '*', color='#FFD700', markersize=26,
            markeredgecolor='black', markeredgewidth=1.2, zorder=7)

    # ── Annotation: place BELOW the target region for clarity ──
    ax.annotate(
        annot_label,
        xy=(tau - 0.15, y_sel),
        xytext=annot_xytext,
        fontsize=16, fontweight='bold', color=annot_color,
        arrowprops=dict(arrowstyle='->', color=annot_color,
                        lw=2.0, connectionstyle='arc3,rad=0.2'),
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=annot_color,
                  lw=1.5, alpha=0.95),
        zorder=8,
    )

    # ── Rejection: large X on terminal points ──
    for key in rejected:
        y_rej = traj[key][-1]
        ax.plot(tau, y_rej, 'o', color='white', markersize=15, zorder=5.5)
        ax.plot(tau, y_rej, 'X', color='#cc0000', markersize=15,
                markeredgewidth=3.0, zorder=6)

    # ── Axes ──
    ax.set_xlim(-0.4, tau + 0.7)
    ax.set_ylim(-0.5, 12.5)
    ax.set_xticks(ts)
    xlabels = ['$t$'] + ['$t{\\!+\\!}' + str(s) + '$' for s in range(1, tau + 1)]
    ax.set_xticklabels(xlabels, fontsize=16)
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xlabel('Time step', fontsize=19)
    ax.grid(axis='y', alpha=0.1, lw=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(16, 6.0), sharey=True,
        gridspec_kw={'wspace': 0.12})

    # (a) ELBO picks D
    draw_panel(ax_l, selected='D', rejected=[],
               annot_label='Selects D\n(best ELBO, unsafe path)',
               annot_color='#e66a00', annot_xytext=(1.5, 0.3),
               show_ystar=True)
    ax_l.set_ylabel('Outcome  $Y_s$', fontsize=19)
    ax_l.set_title('(a)  Point-Target ELBO Selection',
                    fontsize=18, fontweight='bold', pad=10)

    # (b) RA picks A
    draw_panel(ax_r, selected='A', rejected=['B', 'D'],
               annot_label='Selects A\n(best ELBO in feasible set)',
               annot_color='#1a7832', annot_xytext=(1.2, 0.3),
               show_ystar=False)
    ax_r.set_title('(b)  Reach-Avoid Constrained Selection',
                    fontsize=18, fontweight='bold', pad=10)

    # Shared legend
    handles, labels = ax_l.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=15,
               frameon=True, fancybox=True, framealpha=0.95, edgecolor='#cccccc',
               bbox_to_anchor=(0.5, -0.01), columnspacing=1.5, handletextpad=0.5)

    fig.subplots_adjust(bottom=0.18)

    for ext in ['pdf', 'png']:
        path = FIG_DIR / ('rc7_figure1_concept.' + ext)
        fig.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'Saved: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
