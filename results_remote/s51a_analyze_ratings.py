"""
S5.1a Analysis: Unblind and analyze clinician ratings.

Input: s51a_rating_form.csv (completed by clinicians), s51a_answer_key.json
Output: Statistical analysis + results table for paper

Usage:
    python s51a_analyze_ratings.py                    # analyze real ratings
    python s51a_analyze_ratings.py --simulate          # test with simulated data
"""

import json
import csv
import argparse
import numpy as np
from collections import defaultdict


def load_ratings(csv_path):
    """Load completed rating form."""
    ratings = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['plan_a_likert'] and row['plan_b_likert']:
                ratings.append({
                    'case_id': row['case_id'],
                    'plan_a_likert': int(row['plan_a_likert']),
                    'plan_b_likert': int(row['plan_b_likert']),
                    'preference': row['preference'].strip() if row['preference'] else 'N',
                    'comments': row.get('comments', ''),
                })
    return ratings


def simulate_ratings(answer_key, bias=0.5, noise=0.8):
    """Simulate ratings for testing the analysis pipeline.

    bias: mean advantage for RA over ELBO (positive = RA rated higher)
    noise: std of rating noise
    """
    rng = np.random.RandomState(123)
    ratings = []

    for case_id, key in answer_key.items():
        if case_id.startswith('_'):
            continue

        # Base rating around 3 (acceptable)
        base = 3.0
        case_type = answer_key['_metadata']['case_types'][case_id]

        # RA gets a bias advantage in discordant cases
        if case_type == 'discordant':
            ra_bonus = bias
        else:
            ra_bonus = 0.0

        elbo_rating = int(np.clip(round(base + rng.normal(0, noise)), 1, 5))
        ra_rating = int(np.clip(round(base + ra_bonus + rng.normal(0, noise)), 1, 5))

        # Map to blinded A/B
        if key['plan_a_is'] == 'elbo':
            a_rating, b_rating = elbo_rating, ra_rating
        else:
            a_rating, b_rating = ra_rating, elbo_rating

        # Preference follows higher rating
        if a_rating > b_rating:
            pref = 'A'
        elif b_rating > a_rating:
            pref = 'B'
        else:
            pref = 'N'

        ratings.append({
            'case_id': case_id,
            'plan_a_likert': a_rating,
            'plan_b_likert': b_rating,
            'preference': pref,
            'comments': '',
        })

    return ratings


def unblind(ratings, answer_key):
    """Map blinded A/B ratings to ELBO/RA."""
    unblinded = []
    for r in ratings:
        key = answer_key[r['case_id']]
        case_type = answer_key['_metadata']['case_types'][r['case_id']]

        if key['plan_a_is'] == 'elbo':
            elbo_rating = r['plan_a_likert']
            ra_rating = r['plan_b_likert']
            pref_method = {'A': 'elbo', 'B': 'ra', 'N': 'none'}[r['preference']]
        else:
            elbo_rating = r['plan_b_likert']
            ra_rating = r['plan_a_likert']
            pref_method = {'A': 'ra', 'B': 'elbo', 'N': 'none'}[r['preference']]

        unblinded.append({
            'case_id': r['case_id'],
            'case_type': case_type,
            'elbo_rating': elbo_rating,
            'ra_rating': ra_rating,
            'preference': pref_method,
            'comments': r['comments'],
        })

    return unblinded


def wilcoxon_signed_rank(x, y):
    """Paired Wilcoxon signed-rank test (two-sided).

    Returns (statistic, p-value, n_nonzero).
    """
    from scipy.stats import wilcoxon
    d = np.array(x) - np.array(y)
    nonzero = d[d != 0]
    if len(nonzero) < 5:
        return None, None, len(nonzero)
    stat, p = wilcoxon(nonzero)
    return stat, p, len(nonzero)


def cohens_kappa(rater1, rater2):
    """Cohen's kappa for ordinal agreement (linear weights)."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(rater1, rater2, weights='linear')


def analyze(unblinded):
    """Run full analysis and print results."""
    disc = [u for u in unblinded if u['case_type'] == 'discordant']
    conc = [u for u in unblinded if u['case_type'] == 'concordant']

    print("=" * 70)
    print("S5.1a CLINICIAN PLAUSIBILITY ASSESSMENT — RESULTS")
    print("=" * 70)

    # --- Overall ---
    all_elbo = [u['elbo_rating'] for u in unblinded]
    all_ra = [u['ra_rating'] for u in unblinded]
    print(f"\nOverall ({len(unblinded)} cases):")
    print(f"  ELBO mean Likert: {np.mean(all_elbo):.2f} ± {np.std(all_elbo):.2f}")
    print(f"  RA   mean Likert: {np.mean(all_ra):.2f} ± {np.std(all_ra):.2f}")
    print(f"  Difference (RA - ELBO): {np.mean(np.array(all_ra) - np.array(all_elbo)):.2f}")

    # --- Discordant (primary endpoint) ---
    disc_elbo = [u['elbo_rating'] for u in disc]
    disc_ra = [u['ra_rating'] for u in disc]

    print(f"\nDiscordant cases ({len(disc)} cases) — PRIMARY ENDPOINT:")
    print(f"  ELBO mean Likert: {np.mean(disc_elbo):.2f} ± {np.std(disc_elbo):.2f}")
    print(f"  RA   mean Likert: {np.mean(disc_ra):.2f} ± {np.std(disc_ra):.2f}")
    diff = np.mean(np.array(disc_ra) - np.array(disc_elbo))
    print(f"  Mean difference (RA - ELBO): {diff:+.2f}")

    try:
        stat, p, n_nz = wilcoxon_signed_rank(disc_ra, disc_elbo)
        if p is not None:
            print(f"  Wilcoxon signed-rank: W={stat:.0f}, p={p:.4f}, n_nonzero={n_nz}")
            print(f"  {'*** Significant at α=0.05 ***' if p < 0.05 else '  Not significant at α=0.05'}")
        else:
            print(f"  Wilcoxon: insufficient non-zero differences (n={n_nz})")
    except ImportError:
        print("  [scipy not available — skipping Wilcoxon test]")

    # Preference
    pref_ra = sum(1 for u in disc if u['preference'] == 'ra')
    pref_elbo = sum(1 for u in disc if u['preference'] == 'elbo')
    pref_none = sum(1 for u in disc if u['preference'] == 'none')
    print(f"\n  Preference (discordant): RA={pref_ra}, ELBO={pref_elbo}, None={pref_none}")
    if len(disc) > 0:
        print(f"  RA preference rate: {pref_ra/len(disc)*100:.1f}%")

    # --- Concordant (control) ---
    conc_elbo = [u['elbo_rating'] for u in conc]
    conc_ra = [u['ra_rating'] for u in conc]

    print(f"\nConcordant cases ({len(conc)} cases) — CONTROL:")
    print(f"  ELBO mean Likert: {np.mean(conc_elbo):.2f} ± {np.std(conc_elbo):.2f}")
    print(f"  RA   mean Likert: {np.mean(conc_ra):.2f} ± {np.std(conc_ra):.2f}")
    print(f"  Mean difference: {np.mean(np.array(conc_ra) - np.array(conc_elbo)):+.2f}")
    print(f"  (Expected: ~0 since ELBO=RA for concordant cases)")

    # --- Paper table ---
    print("\n" + "=" * 70)
    print("TABLE FOR PAPER (LaTeX):")
    print("=" * 70)
    print(r"\begin{tabular}{l cc cc}")
    print(r"\toprule")
    print(r" & \multicolumn{2}{c}{Likert (mean $\pm$ std)} & \multicolumn{2}{c}{Preference} \\")
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    print(r"Cases & ELBO & RA & \% RA pref. & $p$ (Wilcoxon) \\")
    print(r"\midrule")

    try:
        _, p_disc, _ = wilcoxon_signed_rank(disc_ra, disc_elbo)
        p_str = f"{p_disc:.3f}" if p_disc is not None else "---"
    except ImportError:
        p_str = "---"

    print(f"Discordant ($n={len(disc)}$) & "
          f"${np.mean(disc_elbo):.1f} \\pm {np.std(disc_elbo):.1f}$ & "
          f"${np.mean(disc_ra):.1f} \\pm {np.std(disc_ra):.1f}$ & "
          f"{pref_ra/max(len(disc),1)*100:.0f}\\% & {p_str} \\\\")

    pref_ra_c = sum(1 for u in conc if u['preference'] == 'ra')
    print(f"Concordant ($n={len(conc)}$) & "
          f"${np.mean(conc_elbo):.1f} \\pm {np.std(conc_elbo):.1f}$ & "
          f"${np.mean(conc_ra):.1f} \\pm {np.std(conc_ra):.1f}$ & "
          f"{pref_ra_c/max(len(conc),1)*100:.0f}\\% & --- \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate', action='store_true', help='Run with simulated ratings')
    parser.add_argument('--ratings', default='s51a_rating_form.csv', help='Path to completed rating CSV')
    parser.add_argument('--key', default='s51a_answer_key.json', help='Path to answer key JSON')
    args = parser.parse_args()

    with open(args.key) as f:
        answer_key = json.load(f)

    if args.simulate:
        print("*** RUNNING WITH SIMULATED RATINGS (bias=0.5, noise=0.8) ***\n")
        ratings = simulate_ratings(answer_key)
    else:
        ratings = load_ratings(args.ratings)
        if not ratings:
            print("No completed ratings found. Use --simulate to test the pipeline.")
            return

    unblinded = unblind(ratings, answer_key)
    analyze(unblinded)


if __name__ == '__main__':
    main()
