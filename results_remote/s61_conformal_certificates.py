"""
S6.1: Conformal Safety Certificates for Counterfactual Treatment Planning

Implements distribution-free safety certification using split conformal prediction.

Theory:
  Given a trained model μ̂ and calibration data {(X_i, A_i, Y_i^obs)}, we compute
  nonconformity scores s_i = |Y_i^obs - μ̂(X_i, A_i)| on factual outcomes.
  The conformal quantile q̂_α is the ⌈(1-α)(n+1)⌉/n empirical quantile.

  For a new patient with candidate treatment ā, the prediction band is:
    C_α(ā, X) = [μ̂(X, ā, t) ± q̂_α] for each step t

  Safety certificate: ā is "certified safe at level α" if C_α(ā, X) ⊆ S ∩ T,
  i.e., the entire prediction band lies within the safety and target regions.

  Guarantee: P(Y_true(ā) ∈ S ∩ T) ≥ 1-α (under model exchangeability assumption)

Implementation uses:
  - Cancer: A3 data (oracle + model trajectories) for calibration scores
  - MIMIC: B1 data (model trajectories + observed DBP) for calibration scores
"""

import pickle
import numpy as np
import os
from scipy.stats import spearmanr

BASE_DIR = '/Users/anisiomlacerda/code/target-counterfactual/results_remote'
SEEDS = [10, 101, 1010, 10101, 101010]


# ============================================================
# CANCER SIMULATOR
# ============================================================

# Thresholds (cm³)
CANCER_TARGET_UPPER = 3.0
CANCER_SAFETY_UPPER = 12.0
CANCER_SAFETY_CHEMO = 5.0

GAMMAS = [1, 2, 3, 4]
TAUS = [2, 4, 6, 8]


def load_a3_data():
    """Load A3 oracle-vs-model data for Cancer."""
    data = {}
    for gamma in GAMMAS:
        data[gamma] = {}
        for seed in SEEDS:
            p = os.path.join(BASE_DIR, f'a3_results/coeff_{gamma}/seed_{seed}/a3_oracle_vs_model.pkl')
            if os.path.exists(p):
                try:
                    with open(p, 'rb') as f:
                        data[gamma][seed] = pickle.load(f)
                except Exception:
                    pass
    return data


def compute_cancer_conformal_scores(a3_data, gamma, seed, tau, use_terminal_only=True):
    """Compute nonconformity scores on Cancer calibration data.

    Score = |oracle_cv_terminal - model_cv_terminal| for each candidate.
    We use the ELBO-selected candidate's predictions as the "factual" prediction.

    Actually, for calibration we need factual outcomes. In the A3 data,
    each patient has 100 candidates. The true_sequence (last candidate)
    represents the actual observed treatment. We compute the score on that.

    Better approach: use ALL candidates' oracle vs model discrepancies
    to build the score distribution. This gives us a score per candidate,
    and we calibrate on a held-out subset.
    """
    raw = a3_data[gamma][seed]
    vcip = raw['VCIP']
    if tau not in vcip:
        return None

    cases = vcip[tau]
    scores = []

    for case in cases:
        otf = case.get('oracle_traj_features', {})
        mtf = case.get('model_traj_features', {})

        if 'cv_terminal' not in otf or 'cv_terminal' not in mtf:
            continue

        o_cvt = otf['cv_terminal']  # (k,)
        m_cvt = mtf['cv_terminal']  # (k,)

        if use_terminal_only:
            # Score per candidate: |oracle - model| at terminal step
            s = np.abs(o_cvt - m_cvt)
            scores.extend(s.tolist())
        else:
            # Score per candidate: max over trajectory steps
            o_cvm = otf['cv_max']
            m_cvm = mtf['cv_max']
            s_terminal = np.abs(o_cvt - m_cvt)
            s_max = np.abs(o_cvm - m_cvm)
            s = np.maximum(s_terminal, s_max)
            scores.extend(s.tolist())

    return np.array(scores)


def conformal_quantile(scores, alpha=0.1):
    """Compute the conformal quantile q̂_α."""
    n = len(scores)
    level = np.ceil((1 - alpha) * (n + 1)) / n
    level = min(level, 1.0)
    return np.quantile(scores, level)


def cancer_conformal_analysis(a3_data):
    """Run conformal safety certificates on Cancer data.

    Split: use seeds {10, 101, 1010} for calibration, {10101, 101010} for test.
    """
    cal_seeds = [10, 101, 1010]
    test_seeds = [10101, 101010]

    print("=" * 80)
    print("CANCER: CONFORMAL SAFETY CERTIFICATES")
    print("=" * 80)
    print(f"Calibration seeds: {cal_seeds}, Test seeds: {test_seeds}")

    for alpha in [0.05, 0.10, 0.20]:
        print(f"\n{'='*60}")
        print(f"  α = {alpha} (target coverage: {100*(1-alpha):.0f}%)")
        print(f"{'='*60}")

        print(f"\n  {'gamma':>5s}  {'tau':>3s}  {'q̂_α':>8s}  {'Cert%':>6s}  "
              f"{'OracleCov%':>11s}  {'OracleSafe%':>12s}  {'ELBO Safe%':>11s}")
        print(f"  {'-'*70}")

        for gamma in GAMMAS:
            for tau in TAUS:
                # Calibration: compute scores on cal_seeds
                cal_scores = []
                for seed in cal_seeds:
                    if seed not in a3_data[gamma]:
                        continue
                    s = compute_cancer_conformal_scores(a3_data, gamma, seed, tau)
                    if s is not None:
                        cal_scores.extend(s.tolist())

                if not cal_scores:
                    continue

                cal_scores = np.array(cal_scores)
                q_hat = conformal_quantile(cal_scores, alpha)

                # Test: apply to test_seeds
                n_candidates = 0
                n_certified = 0
                n_oracle_covered = 0  # true outcome in prediction band
                n_oracle_safe = 0     # true outcome in safety region
                n_elbo_safe = 0       # ELBO-selected is oracle-safe

                for seed in test_seeds:
                    if seed not in a3_data[gamma]:
                        continue
                    raw = a3_data[gamma][seed]
                    vcip = raw['VCIP']
                    if tau not in vcip:
                        continue

                    for case in vcip[tau]:
                        otf = case.get('oracle_traj_features', {})
                        mtf = case.get('model_traj_features', {})
                        if 'cv_terminal' not in otf or 'cv_terminal' not in mtf:
                            continue

                        ml = case['model_losses']
                        o_cvt = otf['cv_terminal']
                        o_cvm = otf['cv_max']
                        o_cdm = otf.get('cd_max', np.zeros(len(ml)))
                        m_cvt = mtf['cv_terminal']

                        k = len(ml)

                        for j in range(k):
                            n_candidates += 1

                            # Prediction band: [m_cvt[j] - q̂, m_cvt[j] + q̂]
                            pred_lower = m_cvt[j] - q_hat
                            pred_upper = m_cvt[j] + q_hat

                            # Certified safe: entire band within target AND safety
                            # Target: cv_terminal ≤ TARGET_UPPER
                            # For terminal: pred_upper ≤ TARGET_UPPER
                            # Safety: cv_max ≤ SAFETY_UPPER (we use terminal as proxy)
                            certified = (pred_upper <= CANCER_TARGET_UPPER) and (pred_lower >= 0)

                            if certified:
                                n_certified += 1

                            # Oracle coverage: true terminal value in prediction band
                            if pred_lower <= o_cvt[j] <= pred_upper:
                                n_oracle_covered += 1

                            # Oracle safe (ground truth)
                            o_safe = (o_cvt[j] <= CANCER_TARGET_UPPER and
                                      o_cvm[j] <= CANCER_SAFETY_UPPER)
                            if o_cdm is not None and len(o_cdm) > j:
                                o_safe = o_safe and (o_cdm[j] <= CANCER_SAFETY_CHEMO)
                            if o_safe:
                                n_oracle_safe += 1

                        # ELBO-best oracle safety
                        elbo_idx = np.argmin(ml)
                        e_safe = (o_cvm[elbo_idx] <= CANCER_SAFETY_UPPER and
                                  o_cvt[elbo_idx] <= CANCER_TARGET_UPPER)
                        if o_cdm is not None and len(o_cdm) > elbo_idx:
                            e_safe = e_safe and (o_cdm[elbo_idx] <= CANCER_SAFETY_CHEMO)
                        n_elbo_safe += 1 if e_safe else 0

                if n_candidates == 0:
                    continue

                cert_pct = 100 * n_certified / n_candidates
                cov_pct = 100 * n_oracle_covered / n_candidates
                safe_pct = 100 * n_oracle_safe / n_candidates
                # Note: n_elbo_safe is per-patient, not per-candidate
                n_patients = n_candidates // 100 if n_candidates >= 100 else n_candidates

                print(f"  {gamma:>5d}  {tau:>3d}  {q_hat:>8.3f}  {cert_pct:>5.1f}%  "
                      f"{cov_pct:>10.1f}%  {safe_pct:>11.1f}%  "
                      f"{'N/A':>11s}")


def cancer_conformal_selection(a3_data):
    """Conformal-RA selection: among certified-safe candidates, pick best ELBO.

    Compare: ELBO (unconstrained) vs Oracle-RA vs Conformal-RA.
    Evaluate all on oracle ground truth.
    """
    cal_seeds = [10, 101, 1010]
    test_seeds = [10101, 101010]
    alpha = 0.10

    print("\n" + "=" * 80)
    print("CANCER: CONFORMAL-RA SELECTION (α=0.10)")
    print("=" * 80)
    print("Selection: best ELBO among conformal-certified candidates")
    print("All metrics evaluated on oracle ground truth\n")

    print(f"  {'gamma':>5s}  {'tau':>3s}  |  {'ELBO':^18s}  |  {'Oracle-RA':^18s}  |  {'Conformal-RA':^18s}")
    print(f"  {'':>5s}  {'':>3s}  |  {'Safe%':>7s} {'InTgt%':>7s}  |  {'Safe%':>7s} {'InTgt%':>7s}  |  {'Safe%':>7s} {'InTgt%':>7s} {'Cert%':>6s}")
    print(f"  {'-'*85}")

    for gamma in GAMMAS:
        for tau in TAUS:
            # Calibration
            cal_scores = []
            for seed in cal_seeds:
                if seed not in a3_data[gamma]:
                    continue
                s = compute_cancer_conformal_scores(a3_data, gamma, seed, tau)
                if s is not None:
                    cal_scores.extend(s.tolist())

            if not cal_scores:
                continue
            q_hat = conformal_quantile(np.array(cal_scores), alpha)

            # Test
            elbo_safe = []
            elbo_intgt = []
            ora_safe = []
            ora_intgt = []
            conf_safe = []
            conf_intgt = []
            conf_cert_rates = []

            for seed in test_seeds:
                if seed not in a3_data[gamma]:
                    continue
                raw = a3_data[gamma][seed]
                vcip = raw['VCIP']
                if tau not in vcip:
                    continue

                for case in vcip[tau]:
                    otf = case.get('oracle_traj_features', {})
                    mtf = case.get('model_traj_features', {})
                    if 'cv_terminal' not in otf or 'cv_terminal' not in mtf:
                        continue

                    ml = case['model_losses']
                    o_cvt = otf['cv_terminal']
                    o_cvm = otf['cv_max']
                    o_cdm = otf.get('cd_max', np.zeros(len(ml)))
                    m_cvt = mtf['cv_terminal']
                    k = len(ml)

                    # ELBO-best
                    ei = np.argmin(ml)
                    elbo_safe.append(1.0 if o_cvm[ei] <= CANCER_SAFETY_UPPER else 0.0)
                    elbo_intgt.append(1.0 if o_cvt[ei] <= CANCER_TARGET_UPPER else 0.0)

                    # Oracle-RA
                    o_feas = ((o_cvt <= CANCER_TARGET_UPPER) &
                              (o_cvm <= CANCER_SAFETY_UPPER))
                    if o_cdm is not None and len(o_cdm) == k:
                        o_feas &= (o_cdm <= CANCER_SAFETY_CHEMO)
                    oi = np.argmin(np.where(o_feas, ml, np.inf)) if o_feas.any() else ei
                    ora_safe.append(1.0 if o_cvm[oi] <= CANCER_SAFETY_UPPER else 0.0)
                    ora_intgt.append(1.0 if o_cvt[oi] <= CANCER_TARGET_UPPER else 0.0)

                    # Conformal-RA: certified = prediction band within target
                    certified = ((m_cvt + q_hat) <= CANCER_TARGET_UPPER) & ((m_cvt - q_hat) >= 0)
                    conf_cert_rates.append(certified.sum() / k)

                    if certified.any():
                        ci = np.argmin(np.where(certified, ml, np.inf))
                    else:
                        ci = ei  # fallback to ELBO
                    conf_safe.append(1.0 if o_cvm[ci] <= CANCER_SAFETY_UPPER else 0.0)
                    conf_intgt.append(1.0 if o_cvt[ci] <= CANCER_TARGET_UPPER else 0.0)

            if not elbo_safe:
                continue

            print(f"  {gamma:>5d}  {tau:>3d}  |  "
                  f"{100*np.mean(elbo_safe):>6.1f}% {100*np.mean(elbo_intgt):>6.1f}%  |  "
                  f"{100*np.mean(ora_safe):>6.1f}% {100*np.mean(ora_intgt):>6.1f}%  |  "
                  f"{100*np.mean(conf_safe):>6.1f}% {100*np.mean(conf_intgt):>6.1f}% "
                  f"{100*np.mean(conf_cert_rates):>5.1f}%")


# ============================================================
# MIMIC-III
# ============================================================

MIMIC_TARGET_LOWER = 60.0
MIMIC_TARGET_UPPER = 90.0
MIMIC_SAFETY_LOWER = 50.0
MIMIC_SAFETY_UPPER = 100.0
MIMIC_TAUS = [2, 3, 4, 5]


def load_mimic_data():
    data = {}
    for seed in SEEDS:
        p = os.path.join(BASE_DIR,
            f'mimic_ra/VCIP/train/case_infos/{seed}/False/case_infos_VCIP.pkl')
        if os.path.exists(p):
            with open(p, 'rb') as f:
                data[seed] = pickle.load(f)['VCIP']
    return data


def mimic_conformal_analysis(mimic_data):
    """Conformal safety certificates on MIMIC.

    Calibration score: |observed_DBP - predicted_DBP_terminal| for ELBO-selected candidate.
    Since observed_DBP is the factual outcome under observed treatment,
    and the model's prediction for the ELBO-selected candidate may differ,
    we use the per-patient residual as the nonconformity score.

    Split: seeds {10, 101, 1010} calibration, {10101, 101010} test.
    """
    cal_seeds = [10, 101, 1010]
    test_seeds = [10101, 101010]

    print("\n" + "=" * 80)
    print("MIMIC-III: CONFORMAL SAFETY CERTIFICATES")
    print("=" * 80)
    print(f"Calibration seeds: {cal_seeds}, Test seeds: {test_seeds}")

    for alpha in [0.05, 0.10, 0.20]:
        print(f"\n  α = {alpha} (target coverage: {100*(1-alpha):.0f}%)")

        for tau in MIMIC_TAUS:
            # Calibration: compute |observed - predicted| for each patient
            cal_scores = []
            for seed in cal_seeds:
                if seed not in mimic_data or tau not in mimic_data[seed]:
                    continue
                for case in mimic_data[seed][tau]:
                    obs_dbp = case.get('observed_dbp', None)
                    if obs_dbp is None:
                        continue
                    # Use all candidates' terminal predictions vs observed
                    dbp_t = case['traj_features']['dbp_terminal']
                    # Score: |obs - pred| for each candidate
                    scores = np.abs(dbp_t - obs_dbp)
                    cal_scores.extend(scores.tolist())

            if not cal_scores:
                continue

            q_hat = conformal_quantile(np.array(cal_scores), alpha)

            # Test
            n_patients = 0
            n_elbo_intgt = 0
            n_ra_intgt = 0
            n_conf_intgt = 0
            n_conf_certified = 0
            n_total_candidates = 0
            n_certified_candidates = 0
            elbo_dbps = []
            conf_dbps = []

            for seed in test_seeds:
                if seed not in mimic_data or tau not in mimic_data[seed]:
                    continue
                for case in mimic_data[seed][tau]:
                    ml = case['model_losses']
                    tf = case['traj_features']
                    dbp_t = tf['dbp_terminal']
                    dbp_traj = tf['dbp_trajectory']
                    k = len(ml)
                    n_patients += 1
                    n_total_candidates += k

                    # ELBO-best
                    ei = np.argmin(ml)
                    elbo_dbp = dbp_t[ei]
                    elbo_dbps.append(elbo_dbp)
                    if MIMIC_TARGET_LOWER <= elbo_dbp <= MIMIC_TARGET_UPPER:
                        n_elbo_intgt += 1

                    # Standard RA (threshold filter)
                    ra_feas = ((dbp_t >= MIMIC_TARGET_LOWER) &
                               (dbp_t <= MIMIC_TARGET_UPPER))
                    for s in range(dbp_traj.shape[1]):
                        ra_feas &= ((dbp_traj[:, s] >= MIMIC_SAFETY_LOWER) &
                                    (dbp_traj[:, s] <= MIMIC_SAFETY_UPPER))
                    if ra_feas.any():
                        ri = np.argmin(np.where(ra_feas, ml, np.inf))
                        if MIMIC_TARGET_LOWER <= dbp_t[ri] <= MIMIC_TARGET_UPPER:
                            n_ra_intgt += 1
                    else:
                        if MIMIC_TARGET_LOWER <= elbo_dbp <= MIMIC_TARGET_UPPER:
                            n_ra_intgt += 1

                    # Conformal-RA: certified = prediction band within target
                    # Terminal: [dbp_t - q̂, dbp_t + q̂] ⊆ [TARGET_LOWER, TARGET_UPPER]
                    # → dbp_t - q̂ ≥ TARGET_LOWER AND dbp_t + q̂ ≤ TARGET_UPPER
                    certified = ((dbp_t - q_hat >= MIMIC_TARGET_LOWER) &
                                 (dbp_t + q_hat <= MIMIC_TARGET_UPPER))
                    # Also check trajectory safety bands
                    for s in range(dbp_traj.shape[1]):
                        certified &= ((dbp_traj[:, s] - q_hat >= MIMIC_SAFETY_LOWER) &
                                       (dbp_traj[:, s] + q_hat <= MIMIC_SAFETY_UPPER))

                    n_cert = certified.sum()
                    n_certified_candidates += n_cert

                    if n_cert > 0:
                        n_conf_certified += 1
                        ci = np.argmin(np.where(certified, ml, np.inf))
                        conf_dbps.append(dbp_t[ci])
                        if MIMIC_TARGET_LOWER <= dbp_t[ci] <= MIMIC_TARGET_UPPER:
                            n_conf_intgt += 1
                    else:
                        # Fallback to ELBO
                        conf_dbps.append(elbo_dbp)
                        if MIMIC_TARGET_LOWER <= elbo_dbp <= MIMIC_TARGET_UPPER:
                            n_conf_intgt += 1

            if n_patients == 0:
                continue

            cert_cand_pct = 100 * n_certified_candidates / n_total_candidates
            cert_patient_pct = 100 * n_conf_certified / n_patients

            print(f"    tau={tau}: q̂={q_hat:.2f} mmHg | "
                  f"ELBO InTgt={100*n_elbo_intgt/n_patients:.1f}% | "
                  f"RA InTgt={100*n_ra_intgt/n_patients:.1f}% | "
                  f"Conf InTgt={100*n_conf_intgt/n_patients:.1f}% | "
                  f"Cert: {cert_cand_pct:.1f}% cand, {cert_patient_pct:.1f}% patients")


def mimic_conformal_coverage_check(mimic_data):
    """Verify empirical coverage on MIMIC test set.

    For each patient, check: does observed_dbp fall in [predicted - q̂, predicted + q̂]?
    """
    cal_seeds = [10, 101, 1010]
    test_seeds = [10101, 101010]

    print("\n" + "=" * 80)
    print("MIMIC-III: CONFORMAL COVERAGE VERIFICATION")
    print("=" * 80)
    print("Does P(obs_DBP ∈ [pred ± q̂]) ≥ 1-α hold on test data?")

    for alpha in [0.05, 0.10, 0.20]:
        print(f"\n  α = {alpha} (target: ≥{100*(1-alpha):.0f}%)")
        for tau in MIMIC_TAUS:
            # Calibration scores
            cal_scores = []
            for seed in cal_seeds:
                if seed not in mimic_data or tau not in mimic_data[seed]:
                    continue
                for case in mimic_data[seed][tau]:
                    obs = case.get('observed_dbp', None)
                    if obs is None:
                        continue
                    dbp_t = case['traj_features']['dbp_terminal']
                    cal_scores.extend(np.abs(dbp_t - obs).tolist())

            if not cal_scores:
                continue
            q_hat = conformal_quantile(np.array(cal_scores), alpha)

            # Test coverage
            n_covered = 0
            n_total = 0
            for seed in test_seeds:
                if seed not in mimic_data or tau not in mimic_data[seed]:
                    continue
                for case in mimic_data[seed][tau]:
                    obs = case.get('observed_dbp', None)
                    if obs is None:
                        continue
                    dbp_t = case['traj_features']['dbp_terminal']
                    # Check: for each candidate, does obs fall in [pred - q̂, pred + q̂]?
                    # Per-patient: use ELBO-selected prediction
                    ml = case['model_losses']
                    ei = np.argmin(ml)
                    pred = dbp_t[ei]
                    n_total += 1
                    if pred - q_hat <= obs <= pred + q_hat:
                        n_covered += 1

            if n_total > 0:
                emp_cov = 100 * n_covered / n_total
                print(f"    tau={tau}: q̂={q_hat:.2f} | "
                      f"Empirical coverage: {n_covered}/{n_total} = {emp_cov:.1f}% "
                      f"(target ≥{100*(1-alpha):.0f}%)")


if __name__ == '__main__':
    print("S6.1: Conformal Safety Certificates\n")

    # Load data
    print("Loading A3 (Cancer) data...")
    a3_data = load_a3_data()
    n_a3 = sum(len(a3_data[g]) for g in GAMMAS)
    print(f"  Loaded {n_a3} Cancer files")

    print("Loading MIMIC data...")
    mimic_data = load_mimic_data()
    print(f"  Loaded {len(mimic_data)} MIMIC seeds")

    # Cancer analysis
    cancer_conformal_analysis(a3_data)
    cancer_conformal_selection(a3_data)

    # MIMIC analysis
    mimic_conformal_analysis(mimic_data)
    mimic_conformal_coverage_check(mimic_data)

    print("\n\nS6.1 analysis complete.")
