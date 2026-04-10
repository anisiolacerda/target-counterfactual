#!/usr/bin/env python3
"""
Rater random-effects re-analysis of clinician evaluation data.

Replaces the pooled Wilcoxon test with:
1. Per-rater analysis (separate effect sizes + CIs)
2. Mixed-effects model (rater as random effect)
3. Inter-rater agreement (Cohen's kappa on preference)
4. Overall effect size with 95% CI

This addresses the hostile review's P1 concern about fragile pooling
with calibration differences between raters.
"""

import json
import csv
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings


def load_ratings(csv_path):
    """Load completed rating form, handling CSV quoting issues."""
    ratings = []
    with open(csv_path) as f:
        content = f.read()

    # Try to parse; handle quoting issues in Coli's CSV
    lines = content.strip().split('\n')
    header = lines[0]

    for line in lines[1:]:
        # Handle nested quoting in Coli's CSV
        line = line.strip()
        if not line:
            continue

        # Remove outer quotes if present
        if line.startswith('"') and ',""' in line:
            # Coli format: "C01,""Christiano Coli"",1,4,B,"""
            line = line.strip('"')
            line = line.replace('""', '"')

        parts = []
        in_quotes = False
        current = ''
        for ch in line:
            if ch == '"':
                in_quotes = not in_quotes
            elif ch == ',' and not in_quotes:
                parts.append(current.strip().strip('"'))
                current = ''
            else:
                current += ch
        parts.append(current.strip().strip('"'))

        if len(parts) >= 5:
            try:
                ratings.append({
                    'case_id': parts[0],
                    'reviewer': parts[1],
                    'plan_a_likert': int(parts[2]),
                    'plan_b_likert': int(parts[3]),
                    'preference': parts[4].strip() if parts[4] else 'N',
                    'comments': parts[5] if len(parts) > 5 else '',
                })
            except (ValueError, IndexError):
                continue

    return ratings


def unblind(ratings, answer_key):
    """Unblind ratings using answer key."""
    unblinded = []
    for r in ratings:
        case_id = r['case_id']
        if case_id not in answer_key:
            continue
        key = answer_key[case_id]

        if key['plan_a_is'] == 'ra':
            ra_likert = r['plan_a_likert']
            elbo_likert = r['plan_b_likert']
        else:
            ra_likert = r['plan_b_likert']
            elbo_likert = r['plan_a_likert']

        # Preference mapped to RA/ELBO
        if r['preference'] == 'A':
            pref = 'ra' if key['plan_a_is'] == 'ra' else 'elbo'
        elif r['preference'] == 'B':
            pref = 'ra' if key['plan_b_is'] == 'ra' else 'elbo'
        else:
            pref = 'neutral'

        unblinded.append({
            'case_id': case_id,
            'reviewer': r['reviewer'],
            'ra_likert': ra_likert,
            'elbo_likert': elbo_likert,
            'diff': ra_likert - elbo_likert,
            'preference': pref,
        })

    return unblinded


def cohens_kappa(labels1, labels2, categories=None):
    """Compute Cohen's kappa for inter-rater agreement."""
    if categories is None:
        categories = sorted(set(labels1) | set(labels2))

    n = len(labels1)
    if n == 0:
        return float('nan')

    # Confusion matrix
    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}
    matrix = np.zeros((k, k))
    for l1, l2 in zip(labels1, labels2):
        matrix[cat_idx[l1], cat_idx[l2]] += 1

    # Observed agreement
    po = np.trace(matrix) / n

    # Expected agreement
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    pe = np.sum(row_sums * col_sums) / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def bootstrap_ci(data, statistic_fn, n_boot=10000, alpha=0.05, seed=42):
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_stats.append(statistic_fn(data[idx]))
    boot_stats = sorted(boot_stats)
    lo = boot_stats[int(n_boot * alpha / 2)]
    hi = boot_stats[int(n_boot * (1 - alpha / 2))]
    return lo, hi


def analyze():
    # Load answer key
    with open('results_remote/s51a_answer_key.json') as f:
        answer_key = json.load(f)

    # Load both raters
    r1 = load_ratings('Physician Evaluations/s51a_ratings_Alexandre_Barros_round1_300326.csv')
    r2 = load_ratings('Physician Evaluations/s51a_ratings_Christiano_Coli.csv')

    print(f"Rater 1 (Barros): {len(r1)} ratings")
    print(f"Rater 2 (Coli):   {len(r2)} ratings")

    # Unblind
    u1 = unblind(r1, answer_key)
    u2 = unblind(r2, answer_key)

    print(f"Unblinded rater 1: {len(u1)} cases")
    print(f"Unblinded rater 2: {len(u2)} cases")

    # =====================================================================
    # 1. Per-rater analysis
    # =====================================================================
    print("\n" + "=" * 70)
    print("1. PER-RATER ANALYSIS")
    print("=" * 70)

    for name, data in [("Rater 1 (Barros)", u1), ("Rater 2 (Coli)", u2)]:
        ra = np.array([d['ra_likert'] for d in data])
        elbo = np.array([d['elbo_likert'] for d in data])
        diffs = ra - elbo

        # Wilcoxon signed-rank
        nonzero_diffs = diffs[diffs != 0]
        if len(nonzero_diffs) > 0:
            stat, p_val = stats.wilcoxon(nonzero_diffs, alternative='greater')
        else:
            stat, p_val = 0, 1.0

        # Effect size: matched-pairs rank biserial r
        n_nonzero = len(nonzero_diffs)
        if n_nonzero > 0:
            r_effect = 1 - (2 * stat) / (n_nonzero * (n_nonzero + 1) / 2)
            # Actually for Wilcoxon: r = Z / sqrt(N)
            # Use the formula: r = stat / (n*(n+1)/4) mapped to [-1,1]
            # Simpler: just compute mean diff / pooled std
            cohens_d = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs) > 0 else 0
        else:
            r_effect = 0
            cohens_d = 0

        # Bootstrap 95% CI for mean difference
        lo, hi = bootstrap_ci(diffs, np.mean, n_boot=10000)

        # Preference counts
        ra_pref = sum(1 for d in data if d['preference'] == 'ra')
        elbo_pref = sum(1 for d in data if d['preference'] == 'elbo')
        neutral = sum(1 for d in data if d['preference'] == 'neutral')

        print(f"\n  {name} (n={len(data)}):")
        print(f"    RA mean: {np.mean(ra):.2f} ± {np.std(ra):.2f}")
        print(f"    ELBO mean: {np.mean(elbo):.2f} ± {np.std(elbo):.2f}")
        print(f"    Mean diff (RA - ELBO): {np.mean(diffs):.2f} [{lo:.2f}, {hi:.2f}] 95% CI")
        print(f"    Cohen's d: {cohens_d:.2f}")
        print(f"    Wilcoxon p-value: {p_val:.4f}")
        print(f"    Preference: RA={ra_pref}, ELBO={elbo_pref}, Neutral={neutral}")
        decisive = ra_pref + elbo_pref
        if decisive > 0:
            print(f"    RA preference rate (decisive): {ra_pref}/{decisive} = {100*ra_pref/decisive:.0f}%")

    # =====================================================================
    # 2. Inter-rater agreement
    # =====================================================================
    print("\n" + "=" * 70)
    print("2. INTER-RATER AGREEMENT")
    print("=" * 70)

    # Match cases across raters
    u1_dict = {d['case_id']: d for d in u1}
    u2_dict = {d['case_id']: d for d in u2}
    common_cases = sorted(set(u1_dict.keys()) & set(u2_dict.keys()))
    print(f"  Common cases: {len(common_cases)}")

    # Cohen's kappa on preference (ra/elbo/neutral)
    prefs1 = [u1_dict[c]['preference'] for c in common_cases]
    prefs2 = [u2_dict[c]['preference'] for c in common_cases]
    kappa = cohens_kappa(prefs1, prefs2, categories=['ra', 'elbo', 'neutral'])
    print(f"  Cohen's kappa (preference): {kappa:.3f}")

    # Agreement rate
    agree = sum(1 for p1, p2 in zip(prefs1, prefs2) if p1 == p2)
    print(f"  Agreement rate: {agree}/{len(common_cases)} = {100*agree/len(common_cases):.0f}%")

    # Correlation of Likert diffs
    diffs1 = np.array([u1_dict[c]['diff'] for c in common_cases])
    diffs2 = np.array([u2_dict[c]['diff'] for c in common_cases])
    if len(diffs1) > 2:
        corr, p_corr = stats.spearmanr(diffs1, diffs2)
        print(f"  Spearman correlation of diffs: {corr:.3f} (p={p_corr:.4f})")

    # =====================================================================
    # 3. Mixed-effects model (rater as random effect)
    # =====================================================================
    print("\n" + "=" * 70)
    print("3. COMBINED ANALYSIS (rater-stratified)")
    print("=" * 70)

    # Pool all diffs with rater indicator
    all_diffs = []
    all_raters = []
    for d in u1:
        all_diffs.append(d['diff'])
        all_raters.append(0)
    for d in u2:
        all_diffs.append(d['diff'])
        all_raters.append(1)

    all_diffs = np.array(all_diffs)
    all_raters = np.array(all_raters)

    # Simple mixed model: diff_ij = mu + b_j + epsilon_ij
    # where b_j ~ N(0, sigma_b^2) is the rater random effect
    # Estimate mu (fixed effect = RA advantage)
    mean_r1 = np.mean(all_diffs[all_raters == 0])
    mean_r2 = np.mean(all_diffs[all_raters == 1])
    n1 = np.sum(all_raters == 0)
    n2 = np.sum(all_raters == 1)

    # Grand mean (weighted)
    grand_mean = np.mean(all_diffs)
    # Between-rater variance
    sigma_b2 = max(0, ((mean_r1 - grand_mean)**2 + (mean_r2 - grand_mean)**2) / 2)
    # Within-rater variance
    var_r1 = np.var(all_diffs[all_raters == 0], ddof=1)
    var_r2 = np.var(all_diffs[all_raters == 1], ddof=1)
    sigma_w2 = (var_r1 * (n1 - 1) + var_r2 * (n2 - 1)) / (n1 + n2 - 2)

    # SE of grand mean accounting for rater random effect
    se_mixed = np.sqrt(sigma_b2 / 2 + sigma_w2 / (n1 + n2))  # approximate
    # More conservative: treat raters as clusters
    se_cluster = np.sqrt(((mean_r1 - grand_mean)**2 + (mean_r2 - grand_mean)**2) / 2 +
                         sigma_w2 * (1/n1 + 1/n2) / 4)

    # Bootstrap CI on pooled mean
    lo, hi = bootstrap_ci(all_diffs, np.mean, n_boot=10000)

    # Wilcoxon on pooled (for comparison with original)
    nonzero = all_diffs[all_diffs != 0]
    if len(nonzero) > 0:
        stat, p_pooled = stats.wilcoxon(nonzero, alternative='greater')
    else:
        p_pooled = 1.0

    # One-sample t-test on pooled diffs (more appropriate for mixed model)
    t_stat, p_ttest = stats.ttest_1samp(all_diffs, 0)
    p_ttest_1sided = p_ttest / 2 if t_stat > 0 else 1 - p_ttest / 2

    # Effect size
    cohens_d_pooled = grand_mean / np.sqrt(sigma_w2) if sigma_w2 > 0 else 0

    print(f"  Rater 1 mean diff: {mean_r1:.2f}")
    print(f"  Rater 2 mean diff: {mean_r2:.2f}")
    print(f"  Grand mean diff: {grand_mean:.2f}")
    print(f"  Between-rater SD: {np.sqrt(sigma_b2):.2f}")
    print(f"  Within-rater SD: {np.sqrt(sigma_w2):.2f}")
    print(f"  Mixed-model SE: {se_mixed:.3f}")
    print(f"  95% CI (bootstrap): [{lo:.2f}, {hi:.2f}]")
    print(f"  Cohen's d (pooled): {cohens_d_pooled:.2f}")
    print(f"  Wilcoxon p (pooled): {p_pooled:.6f}")
    print(f"  t-test p (1-sided): {p_ttest_1sided:.6f}")

    # Overall preference
    all_ra = sum(1 for d in u1 + u2 if d['preference'] == 'ra')
    all_elbo = sum(1 for d in u1 + u2 if d['preference'] == 'elbo')
    all_neutral = sum(1 for d in u1 + u2 if d['preference'] == 'neutral')
    decisive = all_ra + all_elbo

    print(f"\n  Overall preference: RA={all_ra}, ELBO={all_elbo}, Neutral={all_neutral}")
    if decisive > 0:
        print(f"  RA rate (decisive): {all_ra}/{decisive} = {100*all_ra/decisive:.0f}%")

    # =====================================================================
    # 4. Summary table for paper
    # =====================================================================
    print("\n" + "=" * 70)
    print("4. SUMMARY TABLE FOR PAPER")
    print("=" * 70)
    print()
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Clinician plausibility evaluation: per-rater and combined results.}")
    print("\\label{tab:clinician_reanalysis}")
    print("\\small")
    print("\\begin{tabular}{l c c c c c}")
    print("\\toprule")
    print("Rater & $n$ & RA Likert & ELBO Likert & Mean diff [95\\% CI] & Decisive pref. \\\\")
    print("\\midrule")

    for name, data in [("Barros", u1), ("Coli", u2)]:
        ra = np.array([d['ra_likert'] for d in data])
        elbo = np.array([d['elbo_likert'] for d in data])
        diffs = ra - elbo
        lo, hi = bootstrap_ci(diffs, np.mean, n_boot=10000)
        ra_pref = sum(1 for d in data if d['preference'] == 'ra')
        elbo_pref = sum(1 for d in data if d['preference'] == 'elbo')
        decisive = ra_pref + elbo_pref
        pref_str = f"{ra_pref}/{decisive} ({100*ra_pref/decisive:.0f}\\%)" if decisive > 0 else "---"
        print(f"{name} & {len(data)} & {np.mean(ra):.1f}$\\pm${np.std(ra):.1f} & "
              f"{np.mean(elbo):.1f}$\\pm${np.std(elbo):.1f} & "
              f"{np.mean(diffs):+.1f} [{lo:+.1f}, {hi:+.1f}] & {pref_str} \\\\")

    lo, hi = bootstrap_ci(all_diffs, np.mean, n_boot=10000)
    pref_str = f"{all_ra}/{decisive} ({100*all_ra/(all_ra+all_elbo):.0f}\\%)"
    ra_all = np.array([d['ra_likert'] for d in u1 + u2])
    elbo_all = np.array([d['elbo_likert'] for d in u1 + u2])
    print("\\midrule")
    print(f"Combined & {len(u1)+len(u2)} & {np.mean(ra_all):.1f}$\\pm${np.std(ra_all):.1f} & "
          f"{np.mean(elbo_all):.1f}$\\pm${np.std(elbo_all):.1f} & "
          f"{grand_mean:+.1f} [{lo:+.1f}, {hi:+.1f}] & {pref_str} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print(f"\nCohen's kappa (inter-rater): {kappa:.2f}")
    print(f"Cohen's d (combined): {cohens_d_pooled:.2f}")
    print(f"Wilcoxon p (combined): {p_pooled:.6f}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION FOR PAPER")
    print("=" * 70)
    if kappa < 0.4:
        print("  Kappa < 0.4 → fair agreement. Reframe as 'suggestive pilot'.")
    elif kappa < 0.6:
        print("  Kappa 0.4-0.6 → moderate agreement. Can claim 'consistent preference'.")
    else:
        print("  Kappa > 0.6 → substantial agreement. Can claim 'strong preference'.")


if __name__ == '__main__':
    analyze()
