"""
B1 Deep Analysis: Per-patient breakdown and reviewer-ready figures.

Focuses on:
- How many patients have ALL predictions below target? (addressing W4 concern)
- Per-patient: is the spread enough to meaningfully discriminate?
- Relationship between observed DBP and filter activity
"""

import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [10, 101, 1010, 10101, 101010]
TAUS = [2, 3, 4, 5]
TARGET_LOWER = 60.0
TARGET_UPPER = 90.0
SAFETY_LOWER = 50.0
SAFETY_UPPER = 100.0


def load_all_data():
    all_data = {}
    for seed in SEEDS:
        pkl_path = os.path.join(
            BASE_DIR, 'VCIP/train/case_infos', str(seed), 'False', 'case_infos_VCIP.pkl')
        with open(pkl_path, 'rb') as f:
            all_data[seed] = pickle.load(f)['VCIP']
    return all_data


def analyze_prediction_range_vs_target(all_data):
    """How many patients have ALL candidates below target? Above? Spanning?"""
    print("=" * 70)
    print("PREDICTION RANGE vs TARGET ANALYSIS")
    print("=" * 70)
    print("(Addresses W4: does the filter find genuinely different plans,")
    print(" or are all predictions in a narrow sub-target range?)")

    for tau in TAUS:
        n_all_below = 0  # All k predictions below target lower
        n_all_above = 0
        n_all_in = 0     # All within target
        n_spanning = 0   # Some below, some in target (or above)
        n_total = 0
        frac_in_target_list = []

        for seed, seed_data in all_data.items():
            for case in seed_data[tau]:
                dbp_t = case['traj_features']['dbp_terminal']
                n_total += 1

                frac_in = np.mean((dbp_t >= TARGET_LOWER) & (dbp_t <= TARGET_UPPER))
                frac_in_target_list.append(frac_in)

                if dbp_t.max() < TARGET_LOWER:
                    n_all_below += 1
                elif dbp_t.min() > TARGET_UPPER:
                    n_all_above += 1
                elif dbp_t.min() >= TARGET_LOWER and dbp_t.max() <= TARGET_UPPER:
                    n_all_in += 1
                else:
                    n_spanning += 1

        frac_arr = np.array(frac_in_target_list)
        print(f"\n  tau={tau} ({n_total} patient-seed obs):")
        print(f"    All predictions below target (<{TARGET_LOWER}): {n_all_below} ({100*n_all_below/n_total:.1f}%)")
        print(f"    All predictions above target (>{TARGET_UPPER}): {n_all_above} ({100*n_all_above/n_total:.1f}%)")
        print(f"    All predictions in target: {n_all_in} ({100*n_all_in/n_total:.1f}%)")
        print(f"    Spanning target boundary: {n_spanning} ({100*n_spanning/n_total:.1f}%)")
        print(f"    Fraction of candidates in target per patient:")
        print(f"      mean={frac_arr.mean():.3f}, median={np.median(frac_arr):.3f}")
        print(f"      0%: {np.sum(frac_arr == 0)}, <10%: {np.sum(frac_arr < 0.1)}, "
              f"<50%: {np.sum(frac_arr < 0.5)}, >50%: {np.sum(frac_arr >= 0.5)}")


def analyze_observed_vs_filter_activity(all_data):
    """Do patients with observed DBP in target need less RA correction?"""
    print("\n" + "=" * 70)
    print("OBSERVED DBP vs FILTER ACTIVITY")
    print("=" * 70)
    print("(Do sicker patients — low observed DBP — benefit more from filtering?)")

    for tau in TAUS:
        obs_in_target = {'changed': 0, 'total': 0, 'dbp_shifts': []}
        obs_below = {'changed': 0, 'total': 0, 'dbp_shifts': []}
        obs_above = {'changed': 0, 'total': 0, 'dbp_shifts': []}

        for seed, seed_data in all_data.items():
            for case in seed_data[tau]:
                obs_dbp = case.get('observed_dbp', None)
                if obs_dbp is None:
                    continue

                ml = case['model_losses']
                tf = case['traj_features']
                dbp_t = tf['dbp_terminal']
                dbp_traj = tf['dbp_trajectory']
                k = len(ml)

                elbo_idx = np.argmin(ml)
                elbo_dbp = dbp_t[elbo_idx]

                feas = np.ones(k, dtype=bool)
                feas &= (dbp_t >= TARGET_LOWER) & (dbp_t <= TARGET_UPPER)
                for s in range(dbp_traj.shape[1]):
                    feas &= (dbp_traj[:, s] >= SAFETY_LOWER) & (dbp_traj[:, s] <= SAFETY_UPPER)

                if feas.sum() > 0:
                    fl = ml.copy()
                    fl[~feas] = np.inf
                    ra_idx = np.argmin(fl)
                    ra_dbp = dbp_t[ra_idx]
                    changed = ra_idx != elbo_idx
                else:
                    ra_dbp = elbo_dbp
                    changed = False

                dbp_shift = ra_dbp - elbo_dbp

                if TARGET_LOWER <= obs_dbp <= TARGET_UPPER:
                    bucket = obs_in_target
                elif obs_dbp < TARGET_LOWER:
                    bucket = obs_below
                else:
                    bucket = obs_above

                bucket['total'] += 1
                if changed:
                    bucket['changed'] += 1
                bucket['dbp_shifts'].append(dbp_shift)

        print(f"\n  tau={tau}:")
        for label, bucket in [('Obs DBP < 60 (low)', obs_below),
                               ('Obs DBP in [60,90]', obs_in_target),
                               ('Obs DBP > 90 (high)', obs_above)]:
            n = bucket['total']
            if n == 0:
                continue
            ch = bucket['changed']
            shifts = np.array(bucket['dbp_shifts'])
            print(f"    {label}: n={n}, RA changed {ch}/{n} ({100*ch/n:.1f}%), "
                  f"mean DBP shift={shifts.mean():+.2f} mmHg")


def analyze_elbo_ranking_quality(all_data):
    """How does ELBO ranking relate to DBP — does better ELBO mean closer to target?"""
    print("\n" + "=" * 70)
    print("ELBO vs DBP CORRELATION (within-patient)")
    print("=" * 70)
    print("(Does a better ELBO score predict DBP closer to target?)")

    from scipy.stats import spearmanr

    for tau in TAUS:
        corrs = []
        for seed, seed_data in all_data.items():
            for case in seed_data[tau]:
                ml = case['model_losses']
                dbp_t = case['traj_features']['dbp_terminal']
                # Distance from target center (75 mmHg)
                target_center = (TARGET_LOWER + TARGET_UPPER) / 2
                dist_from_center = np.abs(dbp_t - target_center)
                # Correlation: does lower ELBO → closer to target center?
                r, _ = spearmanr(ml, dist_from_center)
                if not np.isnan(r):
                    corrs.append(r)

        corrs = np.array(corrs)
        print(f"\n  tau={tau} ({len(corrs)} patients):")
        print(f"    Spearman(ELBO, |DBP - 75|): mean={corrs.mean():.4f}, "
              f"std={corrs.std():.4f}")
        print(f"    (negative = better ELBO → closer to target center)")
        print(f"    Fraction with r < 0: {np.mean(corrs < 0):.3f}")


def analyze_vasopressor_effect(all_data):
    """Does vasopressor dose predict DBP? (clinical face validity)"""
    print("\n" + "=" * 70)
    print("VASOPRESSOR DOSE vs PREDICTED DBP")
    print("=" * 70)
    print("(Clinical validity: do higher vasopressor doses predict higher DBP?)")

    from scipy.stats import spearmanr

    for tau in TAUS:
        corrs = []
        for seed, seed_data in all_data.items():
            for case in seed_data[tau]:
                tf = case.get('treatment_features', {})
                if 'vaso_total' not in tf:
                    continue
                vaso = tf['vaso_total']
                dbp_t = case['traj_features']['dbp_terminal']
                r, _ = spearmanr(vaso, dbp_t)
                if not np.isnan(r):
                    corrs.append(r)

        if not corrs:
            continue
        corrs = np.array(corrs)
        print(f"\n  tau={tau} ({len(corrs)} patients):")
        print(f"    Spearman(vaso_total, DBP_terminal): mean={corrs.mean():.4f}, "
              f"std={corrs.std():.4f}")
        print(f"    (positive = more vasopressors → higher DBP, clinically expected)")
        print(f"    Fraction with r > 0: {np.mean(corrs > 0):.3f}")


if __name__ == '__main__':
    print("B1 Deep Analysis: MIMIC-III RA Filter Discrimination\n")
    all_data = load_all_data()
    analyze_prediction_range_vs_target(all_data)
    analyze_observed_vs_filter_activity(all_data)
    analyze_elbo_ranking_quality(all_data)
    analyze_vasopressor_effect(all_data)
    print("\n" + "=" * 70)
    print("Deep analysis complete.")
