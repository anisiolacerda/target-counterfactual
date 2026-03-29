"""
S5.1a Power Analysis: Can 20 paired cases detect a clinically meaningful difference?

Simulates 10,000 clinician evaluations under different effect sizes (RA advantage)
and noise levels to estimate statistical power for:
  1. Paired Wilcoxon signed-rank test (primary)
  2. Binomial preference test (secondary)

Usage:
    python s51a_power_analysis.py
"""

import numpy as np
from scipy.stats import wilcoxon, binomtest

N_CASES = 20
N_SIMULATIONS = 10000
ALPHA = 0.05
SEED = 42


def simulate_one_trial(n, ra_bias, noise_std, rng):
    """Simulate one clinician evaluation.

    Args:
        n: number of paired cases
        ra_bias: mean Likert advantage of RA over ELBO (e.g., 0.5 = half a point)
        noise_std: std of per-case rating noise
        rng: numpy RandomState

    Returns:
        wilcoxon_p: p-value from paired Wilcoxon test
        pref_ra: number of cases where clinician prefers RA
        mean_diff: mean RA - ELBO Likert difference
    """
    # Generate paired Likert ratings (continuous, then clip to 1-5)
    base = 3.0  # center of scale
    elbo_ratings = np.clip(np.round(base + rng.normal(0, noise_std, n)), 1, 5).astype(int)
    ra_ratings = np.clip(np.round(base + ra_bias + rng.normal(0, noise_std, n)), 1, 5).astype(int)

    diff = ra_ratings - elbo_ratings
    nonzero = diff[diff != 0]

    # Wilcoxon test
    if len(nonzero) >= 5:
        _, wilcoxon_p = wilcoxon(nonzero)
    else:
        wilcoxon_p = 1.0  # not enough non-zero differences

    # Preference: RA preferred when ra_rating > elbo_rating
    pref_ra = np.sum(ra_ratings > elbo_ratings)
    pref_elbo = np.sum(elbo_ratings > ra_ratings)

    return wilcoxon_p, pref_ra, pref_elbo, np.mean(diff)


def run_power_analysis():
    rng = np.random.RandomState(SEED)

    # Parameters to sweep
    biases = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    noise_stds = [0.6, 0.8, 1.0]

    print("=" * 85)
    print(f"POWER ANALYSIS: {N_CASES} paired cases, {N_SIMULATIONS} simulations, α={ALPHA}")
    print("=" * 85)
    print()
    print("RA bias = mean Likert advantage of RA over ELBO (on 1-5 scale)")
    print("Noise σ = per-case rating variability")
    print("Power   = P(reject H0 | H1 true) — probability of detecting the effect")
    print()

    for noise_std in noise_stds:
        print(f"--- Noise σ = {noise_std} ---")
        print(f"{'Bias':>6s} | {'Wilcoxon Power':>14s} | {'Pref Power':>10s} | "
              f"{'Mean Δ':>7s} | {'Mean pref RA':>12s} | {'Mean pref ELBO':>14s} | {'Ties':>5s}")
        print("-" * 85)

        for bias in biases:
            w_reject = 0
            pref_reject = 0
            diffs = []
            pref_ras = []
            pref_elbos = []

            for _ in range(N_SIMULATIONS):
                w_p, pref_ra, pref_elbo, mean_diff = simulate_one_trial(
                    N_CASES, bias, noise_std, rng
                )

                if w_p < ALPHA:
                    w_reject += 1

                # Binomial test: is RA preferred more than 50%?
                n_decisive = pref_ra + pref_elbo
                if n_decisive > 0:
                    binom_p = binomtest(pref_ra, n_decisive, 0.5, alternative='greater').pvalue
                    if binom_p < ALPHA:
                        pref_reject += 1

                diffs.append(mean_diff)
                pref_ras.append(pref_ra)
                pref_elbos.append(pref_elbo)

            w_power = w_reject / N_SIMULATIONS
            p_power = pref_reject / N_SIMULATIONS
            mean_d = np.mean(diffs)
            mean_pra = np.mean(pref_ras)
            mean_pelbo = np.mean(pref_elbos)
            mean_ties = N_CASES - mean_pra - mean_pelbo

            print(f"{bias:6.2f} | {w_power:14.1%} | {p_power:10.1%} | "
                  f"{mean_d:+7.2f} | {mean_pra:12.1f} | {mean_pelbo:14.1f} | {mean_ties:5.1f}")

        print()

    # Detailed analysis at the most realistic scenario
    print("=" * 85)
    print("DETAILED ANALYSIS: bias=0.5, noise=0.8 (most realistic scenario)")
    print("=" * 85)

    bias, noise_std = 0.5, 0.8
    w_pvals = []
    mean_diffs = []
    pref_rates = []

    for _ in range(N_SIMULATIONS):
        w_p, pref_ra, pref_elbo, mean_diff = simulate_one_trial(
            N_CASES, bias, noise_std, rng
        )
        w_pvals.append(w_p)
        mean_diffs.append(mean_diff)
        n_dec = pref_ra + pref_elbo
        pref_rates.append(pref_ra / n_dec if n_dec > 0 else 0.5)

    w_power = np.mean(np.array(w_pvals) < ALPHA)
    print(f"\nWilcoxon power: {w_power:.1%}")
    print(f"Median p-value: {np.median(w_pvals):.3f}")
    print(f"Mean Likert difference: {np.mean(mean_diffs):+.2f} ± {np.std(mean_diffs):.2f}")
    print(f"Mean RA preference rate: {np.mean(pref_rates):.1%} ± {np.std(pref_rates):.1%}")

    # What sample size would we need for 80% power?
    print("\n" + "=" * 85)
    print("SAMPLE SIZE SWEEP: bias=0.5, noise=0.8 — how many cases for 80% power?")
    print("=" * 85)
    print(f"{'N cases':>8s} | {'Wilcoxon Power':>14s} | {'Pref Power':>10s}")
    print("-" * 40)

    for n in [10, 15, 20, 25, 30, 40, 50]:
        w_rej = 0
        p_rej = 0
        for _ in range(N_SIMULATIONS):
            w_p, pref_ra, pref_elbo, _ = simulate_one_trial(n, 0.5, 0.8, rng)
            if w_p < ALPHA:
                w_rej += 1
            n_dec = pref_ra + pref_elbo
            if n_dec > 0 and binomtest(pref_ra, n_dec, 0.5, alternative='greater').pvalue < ALPHA:
                p_rej += 1
        print(f"{n:8d} | {w_rej/N_SIMULATIONS:14.1%} | {p_rej/N_SIMULATIONS:10.1%}")

    print()
    print("INTERPRETATION:")
    print("- bias=0.5 means RA plans are rated half a Likert point higher on average")
    print("- This is a clinically meaningful difference (e.g., 'Acceptable' vs 'Good')")
    print("- If the true effect is ≥0.5 Likert points, 20 cases gives ~X% power (see above)")
    print("- If the true effect is ≥1.0 Likert point, 20 cases is more than sufficient")


if __name__ == '__main__':
    run_power_analysis()
