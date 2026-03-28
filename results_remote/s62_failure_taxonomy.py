"""
S6.2: Failure Taxonomy — When do counterfactual planners produce dangerous recommendations?

Analyzes existing Cancer RA data to identify:
1. Danger rate: P(ELBO-optimal is unsafe | patient)
2. Failure modes: aggressive overshoot, conservative undershoot, model-confused
3. Per-patient heterogeneity: which patients are most at risk?
4. Cross-gamma comparison: how danger scales with confounding
"""

import pickle
import numpy as np
import os
from collections import defaultdict

BASE_DIR = 'results_remote/phase1_ra_v2/my_outputs/cancer_sim_cont/22'
SEEDS = [10, 101, 1010, 10101, 101010]
GAMMAS = [1, 2, 3, 4]
TAUS = [2, 4, 6, 8]

# Paper thresholds (from paper.tex line 179)
TARGET_UPPER = 3.0  # tumor volume ≤ 3.0
SAFETY_VOL_UPPER = 12.0
SAFETY_CHEMO_UPPER = 5.0


def load_all_data():
    """Load all pickle files."""
    all_data = {}
    for gamma in GAMMAS:
        all_data[gamma] = {}
        for seed in SEEDS:
            pkl_path = os.path.join(
                BASE_DIR, f'coeff_{gamma}/VCIP/train/True/case_infos/{seed}/False/case_infos_VCIP.pkl')
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    all_data[gamma][seed] = pickle.load(f)['VCIP']
    return all_data


def analyze_danger_rate(all_data):
    """1. For each gamma × tau, what fraction of ELBO-selected plans are unsafe?"""
    print("=" * 80)
    print("1. DANGER RATE: P(ELBO-optimal is unsafe)")
    print("=" * 80)
    print(f"   Thresholds: T_upper={TARGET_UPPER}, S_vol={SAFETY_VOL_UPPER}, S_chemo={SAFETY_CHEMO_UPPER}")
    print()

    header = f"{'gamma':>5} | {'tau':>3} |" + \
             f" {'ELBO unsafe':>12} | {'Target viol':>12} | {'Safety viol':>12} | {'Both':>8}"
    print(header)
    print("-" * 75)

    for gamma in GAMMAS:
        for tau in TAUS:
            n_patients = 0
            n_elbo_unsafe = 0
            n_target_viol = 0
            n_safety_viol = 0
            n_both = 0

            for seed in SEEDS:
                if seed not in all_data[gamma]:
                    continue
                if tau not in all_data[gamma][seed]:
                    continue
                for case in all_data[gamma][seed][tau]:
                    tf = case['traj_features']
                    model_losses = case['model_losses']

                    # ELBO-best selection
                    elbo_idx = np.argmin(model_losses)
                    cv_t = tf['cv_terminal'][elbo_idx]
                    cv_m = tf['cv_max'][elbo_idx]
                    cd_m = tf['cd_max'][elbo_idx]

                    target_ok = cv_t <= TARGET_UPPER
                    safety_ok = (cv_m <= SAFETY_VOL_UPPER) and (cd_m <= SAFETY_CHEMO_UPPER)

                    n_patients += 1
                    if not target_ok and not safety_ok:
                        n_both += 1
                        n_elbo_unsafe += 1
                    elif not target_ok:
                        n_target_viol += 1
                        n_elbo_unsafe += 1
                    elif not safety_ok:
                        n_safety_viol += 1
                        n_elbo_unsafe += 1

            if n_patients > 0:
                print(f"{gamma:>5} | {tau:>3} | "
                      f"{n_elbo_unsafe:>4}/{n_patients:<4} ({100*n_elbo_unsafe/n_patients:5.1f}%) | "
                      f"{n_target_viol:>4}/{n_patients:<4} ({100*n_target_viol/n_patients:5.1f}%) | "
                      f"{n_safety_viol:>4}/{n_patients:<4} ({100*n_safety_viol/n_patients:5.1f}%) | "
                      f"{n_both:>4} ({100*n_both/n_patients:4.1f}%)")
        print()


def analyze_failure_modes(all_data):
    """2. Classify dangerous ELBO recommendations into failure modes."""
    print("\n" + "=" * 80)
    print("2. FAILURE MODE CLASSIFICATION (gamma=4)")
    print("=" * 80)
    print("   Mode A: 'Aggressive overshoot' — terminal CV > T, high treatment activity")
    print("   Mode B: 'Conservative undershoot' — terminal CV > T, low treatment activity")
    print("   Mode C: 'Toxic path' — terminal CV ≤ T but safety violated along trajectory")
    print()

    gamma = 4
    for tau in TAUS:
        mode_a = 0  # aggressive overshoot
        mode_b = 0  # conservative undershoot
        mode_c = 0  # toxic path
        total_unsafe = 0

        # Collect treatment info
        chemo_unsafe = []
        radio_unsafe = []
        chemo_safe = []
        radio_safe = []
        cv_terminal_unsafe = []
        cv_terminal_safe = []

        for seed in SEEDS:
            if seed not in all_data[gamma] or tau not in all_data[gamma][seed]:
                continue
            for case in all_data[gamma][seed][tau]:
                tf = case['traj_features']
                model_losses = case['model_losses']
                all_seqs = case['all_sequences']  # (k, tau, 2) treatment sequences

                elbo_idx = np.argmin(model_losses)
                cv_t = tf['cv_terminal'][elbo_idx]
                cv_m = tf['cv_max'][elbo_idx]
                cd_m = tf['cd_max'][elbo_idx]

                target_ok = cv_t <= TARGET_UPPER
                safety_ok = (cv_m <= SAFETY_VOL_UPPER) and (cd_m <= SAFETY_CHEMO_UPPER)

                # Get treatment intensity for ELBO-best
                seq = all_seqs[elbo_idx]  # (tau, 2) — [chemo, radio]
                chemo_total = seq[:, 0].sum()
                radio_total = seq[:, 1].sum()

                if target_ok and safety_ok:
                    chemo_safe.append(chemo_total)
                    radio_safe.append(radio_total)
                    cv_terminal_safe.append(cv_t)
                else:
                    total_unsafe += 1
                    chemo_unsafe.append(chemo_total)
                    radio_unsafe.append(radio_total)
                    cv_terminal_unsafe.append(cv_t)

                    if not target_ok:
                        # Classify based on treatment intensity
                        median_chemo = np.median([s[:, 0].sum() for s in all_seqs])
                        if chemo_total > median_chemo:
                            mode_a += 1  # high treatment but still missed target
                        else:
                            mode_b += 1  # low treatment, missed target
                    elif not safety_ok:
                        mode_c += 1  # hit target but unsafe path

        chemo_safe = np.array(chemo_safe) if chemo_safe else np.array([0])
        chemo_unsafe = np.array(chemo_unsafe) if chemo_unsafe else np.array([0])
        radio_safe = np.array(radio_safe) if radio_safe else np.array([0])
        radio_unsafe = np.array(radio_unsafe) if radio_unsafe else np.array([0])
        cv_terminal_safe = np.array(cv_terminal_safe) if cv_terminal_safe else np.array([0])
        cv_terminal_unsafe = np.array(cv_terminal_unsafe) if cv_terminal_unsafe else np.array([0])

        print(f"  tau={tau}: {total_unsafe} unsafe ELBO selections")
        print(f"    Mode A (aggressive overshoot): {mode_a} ({100*mode_a/max(total_unsafe,1):.0f}%)")
        print(f"    Mode B (conservative undershoot): {mode_b} ({100*mode_b/max(total_unsafe,1):.0f}%)")
        print(f"    Mode C (toxic path only): {mode_c} ({100*mode_c/max(total_unsafe,1):.0f}%)")
        print(f"    Treatment intensity (unsafe vs safe):")
        print(f"      Chemo: {chemo_unsafe.mean():.3f} vs {chemo_safe.mean():.3f}")
        print(f"      Radio: {radio_unsafe.mean():.3f} vs {radio_safe.mean():.3f}")
        print(f"    Terminal CV (unsafe): mean={cv_terminal_unsafe.mean():.2f}, max={cv_terminal_unsafe.max():.2f}")
        print(f"    Terminal CV (safe):   mean={cv_terminal_safe.mean():.2f}, max={cv_terminal_safe.max():.2f}")
        print()


def analyze_per_patient_heterogeneity(all_data):
    """3. Per-patient danger rate — which patients are chronically at risk?"""
    print("\n" + "=" * 80)
    print("3. PER-PATIENT HETEROGENEITY (gamma=4, aggregated across seeds)")
    print("=" * 80)

    gamma = 4
    for tau in TAUS:
        patient_danger = defaultdict(lambda: {'unsafe': 0, 'total': 0, 'cv_values': []})

        for seed in SEEDS:
            if seed not in all_data[gamma] or tau not in all_data[gamma][seed]:
                continue
            for case in all_data[gamma][seed][tau]:
                pid = case['individual_id']
                tf = case['traj_features']
                model_losses = case['model_losses']

                elbo_idx = np.argmin(model_losses)
                cv_t = tf['cv_terminal'][elbo_idx]
                cv_m = tf['cv_max'][elbo_idx]
                cd_m = tf['cd_max'][elbo_idx]

                target_ok = cv_t <= TARGET_UPPER
                safety_ok = (cv_m <= SAFETY_VOL_UPPER) and (cd_m <= SAFETY_CHEMO_UPPER)

                patient_danger[pid]['total'] += 1
                patient_danger[pid]['cv_values'].append(cv_t)
                if not (target_ok and safety_ok):
                    patient_danger[pid]['unsafe'] += 1

        # Compute per-patient danger rate
        danger_rates = []
        for pid, info in patient_danger.items():
            rate = info['unsafe'] / info['total']
            danger_rates.append((pid, rate, info['unsafe'], info['total'], np.mean(info['cv_values'])))

        danger_rates.sort(key=lambda x: -x[1])
        rates = [r[1] for r in danger_rates]

        always_unsafe = sum(1 for r in rates if r == 1.0)
        usually_unsafe = sum(1 for r in rates if r >= 0.6)
        sometimes_unsafe = sum(1 for r in rates if 0 < r < 0.6)
        always_safe = sum(1 for r in rates if r == 0.0)

        print(f"\n  tau={tau} ({len(danger_rates)} patients):")
        print(f"    Always unsafe (5/5 seeds): {always_unsafe} patients")
        print(f"    Usually unsafe (≥3/5):     {usually_unsafe} patients")
        print(f"    Sometimes unsafe (1-2/5):  {sometimes_unsafe} patients")
        print(f"    Always safe (0/5):         {always_safe} patients")
        print(f"    Mean danger rate: {np.mean(rates):.3f}, Std: {np.std(rates):.3f}")

        # Show top-10 most dangerous patients
        print(f"    Top-5 most dangerous patients:")
        for pid, rate, n_unsafe, n_total, mean_cv in danger_rates[:5]:
            print(f"      Patient {pid:>3}: {n_unsafe}/{n_total} unsafe ({100*rate:.0f}%), "
                  f"mean terminal CV={mean_cv:.2f}")


def analyze_ra_correction_by_patient(all_data):
    """4. Does RA filtering fix the most dangerous patients?"""
    print("\n" + "=" * 80)
    print("4. RA CORRECTION EFFECTIVENESS BY PATIENT RISK LEVEL")
    print("=" * 80)

    gamma = 4
    for tau in [4, 8]:
        corrected_dangerous = 0
        total_dangerous = 0
        corrected_moderate = 0
        total_moderate = 0
        corrected_safe = 0
        total_safe = 0

        for seed in SEEDS:
            if seed not in all_data[gamma] or tau not in all_data[gamma][seed]:
                continue
            for case in all_data[gamma][seed][tau]:
                tf = case['traj_features']
                model_losses = case['model_losses']
                true_losses = case['true_losses']

                elbo_idx = np.argmin(model_losses)
                cv_t_elbo = tf['cv_terminal'][elbo_idx]
                cv_m_elbo = tf['cv_max'][elbo_idx]
                cd_m_elbo = tf['cd_max'][elbo_idx]

                elbo_safe = (cv_t_elbo <= TARGET_UPPER) and (cv_m_elbo <= SAFETY_VOL_UPPER) and (cd_m_elbo <= SAFETY_CHEMO_UPPER)

                # RA selection
                feasible = (tf['cv_terminal'] <= TARGET_UPPER) & \
                          (tf['cv_max'] <= SAFETY_VOL_UPPER) & \
                          (tf['cd_max'] <= SAFETY_CHEMO_UPPER)

                if feasible.sum() > 0:
                    ra_losses = model_losses.copy()
                    ra_losses[~feasible] = np.inf
                    ra_idx = np.argmin(ra_losses)
                    ra_safe = True
                else:
                    ra_safe = elbo_safe

                # Oracle risk level (based on true loss of ELBO selection)
                oracle_rank = np.argsort(true_losses)
                elbo_rank_in_true = np.where(oracle_rank == elbo_idx)[0][0]

                if elbo_rank_in_true > 50:  # ELBO picks a bad plan (bottom half)
                    total_dangerous += 1
                    if not elbo_safe and ra_safe:
                        corrected_dangerous += 1
                elif elbo_rank_in_true > 20:  # moderate
                    total_moderate += 1
                    if not elbo_safe and ra_safe:
                        corrected_moderate += 1
                else:  # good ELBO selection
                    total_safe += 1
                    if not elbo_safe and ra_safe:
                        corrected_safe += 1

        print(f"\n  tau={tau} (gamma=4):")
        print(f"    ELBO rank > 50 (dangerous): {corrected_dangerous}/{total_dangerous} corrected "
              f"({100*corrected_dangerous/max(total_dangerous,1):.1f}%)")
        print(f"    ELBO rank 20-50 (moderate): {corrected_moderate}/{total_moderate} corrected "
              f"({100*corrected_moderate/max(total_moderate,1):.1f}%)")
        print(f"    ELBO rank < 20 (good):      {corrected_safe}/{total_safe} corrected "
              f"({100*corrected_safe/max(total_safe,1):.1f}%)")


def main():
    print("S6.2: Failure Taxonomy — When Do Counterfactual Planners Fail?")
    print("=" * 80)

    all_data = load_all_data()

    # Count loaded
    for gamma in GAMMAS:
        n_seeds = len(all_data[gamma])
        print(f"  gamma={gamma}: {n_seeds} seeds loaded")

    analyze_danger_rate(all_data)
    analyze_failure_modes(all_data)
    analyze_per_patient_heterogeneity(all_data)
    analyze_ra_correction_by_patient(all_data)

    print("\n" + "=" * 80)
    print("Failure taxonomy complete.")


if __name__ == '__main__':
    main()
