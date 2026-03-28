"""
S6.1 v2: Refined Conformal Safety Certificates

Key refinements over v1:
1. One-sided conformal bounds for Cancer (upper bound on CV only)
2. Asymmetric bounds for MIMIC (separate upper/lower calibration)
3. "Accuracy gap" analysis: how good must the model be for useful certificates?
4. Per-constraint separate calibration (target vs safety)

Insight from v1: symmetric bands are too conservative. The model errors are
directionally structured (Cancer model under-predicts, MIMIC has asymmetric errors),
so one-sided/asymmetric bounds are much tighter.
"""

import pickle
import numpy as np
import os

BASE_DIR = '/Users/anisiomlacerda/code/target-counterfactual/results_remote'
SEEDS = [10, 101, 1010, 10101, 101010]
CAL_SEEDS = [10, 101, 1010]
TEST_SEEDS = [10101, 101010]

# Cancer thresholds
CANCER_TARGET_UPPER = 3.0
CANCER_SAFETY_UPPER = 12.0
CANCER_SAFETY_CHEMO = 5.0
GAMMAS = [1, 2, 3, 4]
TAUS = [2, 4, 6, 8]

# MIMIC thresholds
MIMIC_TARGET_LOWER = 60.0
MIMIC_TARGET_UPPER = 90.0
MIMIC_SAFETY_LOWER = 50.0
MIMIC_SAFETY_UPPER = 100.0
MIMIC_TAUS = [2, 3, 4, 5]


def conformal_quantile(scores, alpha=0.1):
    """Compute the conformal quantile q̂_α at level ceil((1-α)(n+1))/n."""
    n = len(scores)
    level = np.ceil((1 - alpha) * (n + 1)) / n
    level = min(level, 1.0)
    return np.quantile(scores, level)


# ============================================================
# DATA LOADING
# ============================================================

def load_a3_data():
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


def load_mimic_data():
    data = {}
    for seed in SEEDS:
        p = os.path.join(BASE_DIR,
            f'mimic_ra/VCIP/train/case_infos/{seed}/False/case_infos_VCIP.pkl')
        if os.path.exists(p):
            with open(p, 'rb') as f:
                data[seed] = pickle.load(f)['VCIP']
    return data


# ============================================================
# CANCER: ONE-SIDED CONFORMAL
# ============================================================

def cancer_onesided(a3_data):
    """One-sided conformal for Cancer: upper bound on CV only.

    For target constraint cv_terminal ≤ T: need upper conformal bound ≤ T.
    Score: s_i = oracle_cv_terminal_i - model_cv_terminal_i (SIGNED residual).
    Upper bound: model_cv_terminal + q̂_upper where q̂_upper = quantile of s_i.
    Certificate: model_cv_terminal + q̂_upper ≤ TARGET_UPPER.

    For safety constraint cv_max ≤ S: same approach with cv_max.
    Score: s_i = oracle_cv_max_i - model_cv_max_i.
    Upper bound: model_cv_max + q̂_safety ≤ SAFETY_UPPER.
    """
    print("=" * 80)
    print("CANCER: ONE-SIDED CONFORMAL SAFETY CERTIFICATES")
    print("=" * 80)
    print("Upper bound only: need model_pred + q̂ ≤ threshold")
    print("Score = oracle - model (signed residual)\n")

    for alpha in [0.05, 0.10, 0.20]:
        print(f"\n--- α = {alpha} (coverage ≥ {100*(1-alpha):.0f}%) ---\n")
        print(f"  {'γ':>3s} {'τ':>3s} | {'q̂_tgt':>7s} {'q̂_saf':>7s} | "
              f"{'Cert%':>6s} {'n_cert':>6s} | {'OracCov%':>9s} | "
              f"{'Conf-RA':^22s} | {'Oracle-RA':^14s} | {'ELBO':^14s}")
        print(f"  {'':>3s} {'':>3s} | {'':>7s} {'':>7s} | {'':>6s} {'':>6s} | {'':>9s} | "
              f"{'Safe%':>7s} {'InTgt%':>7s} {'TL':>6s} | "
              f"{'Safe%':>7s} {'TL':>5s} | {'Safe%':>7s} {'TL':>5s}")
        print(f"  {'-'*120}")

        for gamma in GAMMAS:
            for tau in TAUS:
                # Calibration: collect signed residuals
                cal_target_scores = []  # oracle_cv_terminal - model_cv_terminal
                cal_safety_scores = []  # oracle_cv_max - model_cv_max

                for seed in CAL_SEEDS:
                    if seed not in a3_data[gamma]:
                        continue
                    vcip = a3_data[gamma][seed]['VCIP']
                    if tau not in vcip:
                        continue
                    for case in vcip[tau]:
                        otf = case.get('oracle_traj_features', {})
                        mtf = case.get('model_traj_features', {})
                        if 'cv_terminal' not in otf:
                            continue
                        # Signed residuals
                        s_tgt = otf['cv_terminal'] - mtf['cv_terminal']
                        s_saf = otf['cv_max'] - mtf['cv_max']
                        cal_target_scores.extend(s_tgt.tolist())
                        cal_safety_scores.extend(s_saf.tolist())

                if not cal_target_scores:
                    continue

                q_tgt = conformal_quantile(np.array(cal_target_scores), alpha)
                q_saf = conformal_quantile(np.array(cal_safety_scores), alpha)

                # Test: apply certificates
                n_cand = 0
                n_certified = 0
                n_oracle_covered_tgt = 0
                n_oracle_covered_saf = 0

                # Per-patient selection metrics
                elbo_safe_list = []
                elbo_intgt_list = []
                elbo_tl_list = []
                ora_safe_list = []
                ora_intgt_list = []
                ora_tl_list = []
                conf_safe_list = []
                conf_intgt_list = []
                conf_tl_list = []
                cert_rates = []

                for seed in TEST_SEEDS:
                    if seed not in a3_data[gamma]:
                        continue
                    vcip = a3_data[gamma][seed]['VCIP']
                    if tau not in vcip:
                        continue
                    for case in vcip[tau]:
                        otf = case.get('oracle_traj_features', {})
                        mtf = case.get('model_traj_features', {})
                        if 'cv_terminal' not in otf:
                            continue
                        ml = case['model_losses']
                        tl = case['true_losses']
                        o_cvt = otf['cv_terminal']
                        o_cvm = otf['cv_max']
                        o_cdm = otf.get('cd_max', np.zeros(len(ml)))
                        m_cvt = mtf['cv_terminal']
                        m_cvm = mtf['cv_max']
                        k = len(ml)

                        # Certificate: upper bound ≤ threshold
                        cert_tgt = (m_cvt + q_tgt) <= CANCER_TARGET_UPPER
                        cert_saf = (m_cvm + q_saf) <= CANCER_SAFETY_UPPER
                        certified = cert_tgt & cert_saf

                        n_cert_case = certified.sum()
                        n_certified += n_cert_case
                        n_cand += k
                        cert_rates.append(n_cert_case / k)

                        # Oracle coverage verification
                        for j in range(k):
                            if o_cvt[j] <= m_cvt[j] + q_tgt:
                                n_oracle_covered_tgt += 1
                            if o_cvm[j] <= m_cvm[j] + q_saf:
                                n_oracle_covered_saf += 1

                        # --- Selection comparison ---
                        # ELBO
                        ei = np.argmin(ml)
                        e_safe = (o_cvm[ei] <= CANCER_SAFETY_UPPER)
                        e_intgt = (o_cvt[ei] <= CANCER_TARGET_UPPER)
                        elbo_safe_list.append(float(e_safe))
                        elbo_intgt_list.append(float(e_intgt))
                        elbo_tl_list.append(tl[ei])

                        # Oracle-RA
                        o_feas = ((o_cvt <= CANCER_TARGET_UPPER) &
                                  (o_cvm <= CANCER_SAFETY_UPPER))
                        if o_cdm is not None and len(o_cdm) == k:
                            o_feas &= (o_cdm <= CANCER_SAFETY_CHEMO)
                        if o_feas.any():
                            oi = np.argmin(np.where(o_feas, ml, np.inf))
                        else:
                            oi = ei
                        ora_safe_list.append(float(o_cvm[oi] <= CANCER_SAFETY_UPPER))
                        ora_intgt_list.append(float(o_cvt[oi] <= CANCER_TARGET_UPPER))
                        ora_tl_list.append(tl[oi])

                        # Conformal-RA: best ELBO among certified
                        if certified.any():
                            ci = np.argmin(np.where(certified, ml, np.inf))
                        else:
                            ci = ei  # fallback
                        conf_safe_list.append(float(o_cvm[ci] <= CANCER_SAFETY_UPPER))
                        conf_intgt_list.append(float(o_cvt[ci] <= CANCER_TARGET_UPPER))
                        conf_tl_list.append(tl[ci])

                if n_cand == 0:
                    continue

                cov_tgt = 100 * n_oracle_covered_tgt / n_cand
                cov_saf = 100 * n_oracle_covered_saf / n_cand
                cov_str = f"{min(cov_tgt, cov_saf):.1f}%"
                cert_pct = 100 * np.mean(cert_rates)

                print(f"  {gamma:>3d} {tau:>3d} | {q_tgt:>7.3f} {q_saf:>7.3f} | "
                      f"{cert_pct:>5.1f}% {n_certified:>6d} | {cov_str:>9s} | "
                      f"{100*np.mean(conf_safe_list):>6.1f}% {100*np.mean(conf_intgt_list):>6.1f}% "
                      f"{np.mean(conf_tl_list):>.4f} | "
                      f"{100*np.mean(ora_safe_list):>6.1f}% {np.mean(ora_tl_list):>.3f} | "
                      f"{100*np.mean(elbo_safe_list):>6.1f}% {np.mean(elbo_tl_list):>.3f}")


def cancer_accuracy_gap(a3_data):
    """Compute the 'accuracy gap': how precise must the model be for useful certificates?

    For one-sided certificate: need q̂ ≤ TARGET_UPPER - model_pred_median
    Since model predictions cluster near 0 (A3 finding), the gap is ~TARGET_UPPER - q̂.
    """
    print("\n" + "=" * 80)
    print("CANCER: ACCURACY GAP ANALYSIS")
    print("=" * 80)
    print("How precise must the model be for conformal certificates to certify >50% of candidates?")
    print("Required: q̂ ≤ TARGET - median(model_pred) for target cert")
    print("          q̂ ≤ SAFETY - median(model_pred_max) for safety cert\n")

    alpha = 0.10

    for gamma in GAMMAS:
        print(f"\n  gamma={gamma}:")
        print(f"  {'tau':>3s} | {'q̂_tgt':>7s} {'q̂_saf':>7s} | "
              f"{'med(m_cvt)':>10s} {'med(m_cvm)':>10s} | "
              f"{'MaxAllow_T':>10s} {'MaxAllow_S':>10s} | "
              f"{'Ratio_T':>7s} {'Ratio_S':>7s}")
        print(f"  {'-'*90}")

        for tau in TAUS:
            cal_tgt = []
            cal_saf = []
            m_cvt_all = []
            m_cvm_all = []

            for seed in CAL_SEEDS:
                if seed not in a3_data[gamma]:
                    continue
                vcip = a3_data[gamma][seed]['VCIP']
                if tau not in vcip:
                    continue
                for case in vcip[tau]:
                    otf = case.get('oracle_traj_features', {})
                    mtf = case.get('model_traj_features', {})
                    if 'cv_terminal' not in otf:
                        continue
                    cal_tgt.extend((otf['cv_terminal'] - mtf['cv_terminal']).tolist())
                    cal_saf.extend((otf['cv_max'] - mtf['cv_max']).tolist())
                    m_cvt_all.extend(mtf['cv_terminal'].tolist())
                    m_cvm_all.extend(mtf['cv_max'].tolist())

            if not cal_tgt:
                continue

            q_tgt = conformal_quantile(np.array(cal_tgt), alpha)
            q_saf = conformal_quantile(np.array(cal_saf), alpha)
            med_cvt = np.median(m_cvt_all)
            med_cvm = np.median(m_cvm_all)

            # Max allowable q for >50% certification
            # Cert requires: model_pred + q ≤ threshold
            # At median: med + q ≤ threshold → q ≤ threshold - med
            max_q_tgt = CANCER_TARGET_UPPER - med_cvt
            max_q_saf = CANCER_SAFETY_UPPER - med_cvm

            ratio_tgt = q_tgt / max_q_tgt if max_q_tgt > 0 else float('inf')
            ratio_saf = q_saf / max_q_saf if max_q_saf > 0 else float('inf')

            print(f"  {tau:>3d} | {q_tgt:>7.3f} {q_saf:>7.3f} | "
                  f"{med_cvt:>10.3f} {med_cvm:>10.3f} | "
                  f"{max_q_tgt:>10.3f} {max_q_saf:>10.3f} | "
                  f"{ratio_tgt:>7.2f} {ratio_saf:>7.2f}")

    print("\n  Ratio < 1: model accurate enough for certification at median")
    print("  Ratio > 1: model needs improvement by this factor for certification")


def cancer_score_distribution(a3_data):
    """Analyze the distribution of signed residuals to understand model bias."""
    print("\n" + "=" * 80)
    print("CANCER: SIGNED RESIDUAL DISTRIBUTION (oracle - model)")
    print("=" * 80)
    print("Positive = model under-predicts, Negative = model over-predicts\n")

    for gamma in GAMMAS:
        print(f"\n  gamma={gamma}:")
        print(f"  {'tau':>3s} | {'n':>6s} | {'mean':>8s} {'std':>8s} | "
              f"{'p10':>8s} {'p50':>8s} {'p90':>8s} {'p95':>8s} {'p99':>8s}")
        print(f"  {'-'*85}")

        for tau in TAUS:
            all_scores = []
            for seed in SEEDS:
                if seed not in a3_data[gamma]:
                    continue
                vcip = a3_data[gamma][seed]['VCIP']
                if tau not in vcip:
                    continue
                for case in vcip[tau]:
                    otf = case.get('oracle_traj_features', {})
                    mtf = case.get('model_traj_features', {})
                    if 'cv_terminal' not in otf:
                        continue
                    s = otf['cv_terminal'] - mtf['cv_terminal']
                    all_scores.extend(s.tolist())

            if not all_scores:
                continue
            s = np.array(all_scores)
            print(f"  {tau:>3d} | {len(s):>6d} | {s.mean():>8.3f} {s.std():>8.3f} | "
                  f"{np.percentile(s, 10):>8.3f} {np.percentile(s, 50):>8.3f} "
                  f"{np.percentile(s, 90):>8.3f} {np.percentile(s, 95):>8.3f} "
                  f"{np.percentile(s, 99):>8.3f}")


# ============================================================
# MIMIC: ASYMMETRIC CONFORMAL
# ============================================================

def mimic_asymmetric(mimic_data):
    """Asymmetric conformal for MIMIC.

    For target [60, 90]:
    - Lower bound: need pred - q̂_lower ≥ 60 → q̂_lower = quantile of (pred - obs)
    - Upper bound: need pred + q̂_upper ≤ 90 → q̂_upper = quantile of (obs - pred)

    Using separate one-sided quantiles gives tighter bands.
    """
    print("\n" + "=" * 80)
    print("MIMIC: ASYMMETRIC CONFORMAL SAFETY CERTIFICATES")
    print("=" * 80)
    print("Separate upper/lower calibration for tighter bands")
    print("Lower bound score: pred_i - obs_i (positive = over-prediction)")
    print("Upper bound score: obs_i - pred_i (positive = under-prediction)\n")

    for alpha in [0.05, 0.10, 0.20]:
        print(f"\n--- α = {alpha} (coverage ≥ {100*(1-alpha):.0f}%) ---\n")
        print(f"  {'tau':>3s} | {'q̂_lo':>6s} {'q̂_up':>6s} | {'Band':>8s} | "
              f"{'Cert%':>6s} {'PatCert%':>8s} | "
              f"{'ConfRA':^16s} | {'RA':^12s} | {'ELBO':^12s}")
        print(f"  {'':>3s} | {'':>6s} {'':>6s} | {'':>8s} | {'':>6s} {'':>8s} | "
              f"{'InTgt%':>7s} {'DBP':>7s} | {'InTgt%':>6s} {'DBP':>5s} | {'InTgt%':>6s} {'DBP':>5s}")
        print(f"  {'-'*100}")

        for tau in MIMIC_TAUS:
            # Calibration: per-candidate residuals using ELBO-selected predictions
            # For each patient, score = signed residual of EACH candidate's prediction vs observed
            cal_scores_lower = []  # pred - obs (for lower bound)
            cal_scores_upper = []  # obs - pred (for upper bound)

            for seed in CAL_SEEDS:
                if seed not in mimic_data or tau not in mimic_data[seed]:
                    continue
                for case in mimic_data[seed][tau]:
                    obs = case.get('observed_dbp', None)
                    if obs is None:
                        continue
                    dbp_t = case['traj_features']['dbp_terminal']
                    # Signed residuals
                    cal_scores_lower.extend((dbp_t - obs).tolist())
                    cal_scores_upper.extend((obs - dbp_t).tolist())

            if not cal_scores_lower:
                continue

            # One-sided quantiles
            q_lower = conformal_quantile(np.array(cal_scores_lower), alpha)
            q_upper = conformal_quantile(np.array(cal_scores_upper), alpha)

            # Effective band: [pred - q_lower, pred + q_upper]
            band_width = q_lower + q_upper

            # Test
            n_patients = 0
            n_cert_patients = 0
            n_total_cand = 0
            n_cert_cand = 0
            elbo_intgt = 0
            ra_intgt = 0
            conf_intgt = 0
            elbo_dbps = []
            ra_dbps = []
            conf_dbps = []

            for seed in TEST_SEEDS:
                if seed not in mimic_data or tau not in mimic_data[seed]:
                    continue
                for case in mimic_data[seed][tau]:
                    ml = case['model_losses']
                    tf = case['traj_features']
                    dbp_t = tf['dbp_terminal']
                    dbp_traj = tf['dbp_trajectory']
                    k = len(ml)
                    n_patients += 1
                    n_total_cand += k

                    # ELBO
                    ei = np.argmin(ml)
                    elbo_dbps.append(dbp_t[ei])
                    if MIMIC_TARGET_LOWER <= dbp_t[ei] <= MIMIC_TARGET_UPPER:
                        elbo_intgt += 1

                    # Standard RA
                    ra_feas = ((dbp_t >= MIMIC_TARGET_LOWER) &
                               (dbp_t <= MIMIC_TARGET_UPPER))
                    for s in range(dbp_traj.shape[1]):
                        ra_feas &= ((dbp_traj[:, s] >= MIMIC_SAFETY_LOWER) &
                                     (dbp_traj[:, s] <= MIMIC_SAFETY_UPPER))
                    if ra_feas.any():
                        ri = np.argmin(np.where(ra_feas, ml, np.inf))
                        ra_dbps.append(dbp_t[ri])
                        if MIMIC_TARGET_LOWER <= dbp_t[ri] <= MIMIC_TARGET_UPPER:
                            ra_intgt += 1
                    else:
                        ra_dbps.append(dbp_t[ei])
                        if MIMIC_TARGET_LOWER <= dbp_t[ei] <= MIMIC_TARGET_UPPER:
                            ra_intgt += 1

                    # Asymmetric conformal:
                    # Lower conf bound = pred - q_lower ≥ TARGET_LOWER → pred ≥ TARGET_LOWER + q_lower
                    # Upper conf bound = pred + q_upper ≤ TARGET_UPPER → pred ≤ TARGET_UPPER - q_upper
                    cert_tgt = ((dbp_t >= MIMIC_TARGET_LOWER + q_lower) &
                                (dbp_t <= MIMIC_TARGET_UPPER - q_upper))
                    # Safety on trajectory
                    cert_saf = np.ones(k, dtype=bool)
                    for s in range(dbp_traj.shape[1]):
                        cert_saf &= ((dbp_traj[:, s] >= MIMIC_SAFETY_LOWER + q_lower) &
                                      (dbp_traj[:, s] <= MIMIC_SAFETY_UPPER - q_upper))
                    certified = cert_tgt & cert_saf

                    n_cert = certified.sum()
                    n_cert_cand += n_cert

                    if n_cert > 0:
                        n_cert_patients += 1
                        ci = np.argmin(np.where(certified, ml, np.inf))
                        conf_dbps.append(dbp_t[ci])
                        if MIMIC_TARGET_LOWER <= dbp_t[ci] <= MIMIC_TARGET_UPPER:
                            conf_intgt += 1
                    else:
                        conf_dbps.append(dbp_t[ei])
                        if MIMIC_TARGET_LOWER <= dbp_t[ei] <= MIMIC_TARGET_UPPER:
                            conf_intgt += 1

            if n_patients == 0:
                continue

            print(f"  {tau:>3d} | {q_lower:>6.2f} {q_upper:>6.2f} | {band_width:>7.2f}W | "
                  f"{100*n_cert_cand/n_total_cand:>5.1f}% {100*n_cert_patients/n_patients:>7.1f}% | "
                  f"{100*conf_intgt/n_patients:>6.1f}% {np.mean(conf_dbps):>7.1f} | "
                  f"{100*ra_intgt/n_patients:>5.1f}% {np.mean(ra_dbps):>5.1f} | "
                  f"{100*elbo_intgt/n_patients:>5.1f}% {np.mean(elbo_dbps):>5.1f}")


def mimic_coverage_asymmetric(mimic_data):
    """Verify asymmetric coverage on MIMIC test set."""
    print("\n" + "=" * 80)
    print("MIMIC: ASYMMETRIC COVERAGE VERIFICATION")
    print("=" * 80)
    print("Check: P(pred - q̂_lo ≤ obs ≤ pred + q̂_up) ≥ 1-α\n")

    for alpha in [0.05, 0.10, 0.20]:
        print(f"  α = {alpha} (target: ≥{100*(1-alpha):.0f}%)")
        for tau in MIMIC_TAUS:
            cal_lo = []
            cal_up = []
            for seed in CAL_SEEDS:
                if seed not in mimic_data or tau not in mimic_data[seed]:
                    continue
                for case in mimic_data[seed][tau]:
                    obs = case.get('observed_dbp', None)
                    if obs is None:
                        continue
                    dbp_t = case['traj_features']['dbp_terminal']
                    cal_lo.extend((dbp_t - obs).tolist())
                    cal_up.extend((obs - dbp_t).tolist())

            if not cal_lo:
                continue

            q_lo = conformal_quantile(np.array(cal_lo), alpha)
            q_up = conformal_quantile(np.array(cal_up), alpha)

            # Check coverage on test set (per-patient, ELBO-selected prediction)
            n_cov = 0
            n_tot = 0
            for seed in TEST_SEEDS:
                if seed not in mimic_data or tau not in mimic_data[seed]:
                    continue
                for case in mimic_data[seed][tau]:
                    obs = case.get('observed_dbp', None)
                    if obs is None:
                        continue
                    ml = case['model_losses']
                    dbp_t = case['traj_features']['dbp_terminal']
                    ei = np.argmin(ml)
                    pred = dbp_t[ei]
                    n_tot += 1
                    # Asymmetric band: [pred - q_lo, pred + q_up]
                    if (pred - q_lo) <= obs <= (pred + q_up):
                        n_cov += 1

            if n_tot > 0:
                print(f"    tau={tau}: q̂_lo={q_lo:.2f}, q̂_up={q_up:.2f}, band={q_lo+q_up:.2f} | "
                      f"Coverage: {n_cov}/{n_tot} = {100*n_cov/n_tot:.1f}% (target ≥{100*(1-alpha):.0f}%)")
        print()


def mimic_accuracy_gap(mimic_data):
    """How much must the model improve for useful certificates on MIMIC?"""
    print("\n" + "=" * 80)
    print("MIMIC: ACCURACY GAP ANALYSIS")
    print("=" * 80)
    print(f"Target range: [{MIMIC_TARGET_LOWER}, {MIMIC_TARGET_UPPER}] mmHg (width=30)")
    print("For certification: need q̂_lo + q̂_up < range width")
    print("  Lower cert: pred ≥ TARGET_LO + q̂_lo")
    print("  Upper cert: pred ≤ TARGET_UP - q̂_up")
    print("  Feasible pred range: [TARGET_LO + q̂_lo, TARGET_UP - q̂_up]\n")

    alpha = 0.10
    for tau in MIMIC_TAUS:
        cal_lo = []
        cal_up = []
        pred_all = []
        for seed in CAL_SEEDS:
            if seed not in mimic_data or tau not in mimic_data[seed]:
                continue
            for case in mimic_data[seed][tau]:
                obs = case.get('observed_dbp', None)
                if obs is None:
                    continue
                dbp_t = case['traj_features']['dbp_terminal']
                cal_lo.extend((dbp_t - obs).tolist())
                cal_up.extend((obs - dbp_t).tolist())
                pred_all.extend(dbp_t.tolist())

        if not cal_lo:
            continue
        q_lo = conformal_quantile(np.array(cal_lo), alpha)
        q_up = conformal_quantile(np.array(cal_up), alpha)
        pred_arr = np.array(pred_all)

        feasible_lo = MIMIC_TARGET_LOWER + q_lo
        feasible_up = MIMIC_TARGET_UPPER - q_up
        feasible_width = feasible_up - feasible_lo

        # What fraction of predictions fall in feasible range?
        in_feas = np.mean((pred_arr >= feasible_lo) & (pred_arr <= feasible_up))

        # Required improvement factor: need band < 30 → need q_lo + q_up < 30
        band = q_lo + q_up
        improvement_needed = band / 30.0

        print(f"  tau={tau}: q̂_lo={q_lo:.2f}, q̂_up={q_up:.2f}, band={band:.2f} mmHg")
        print(f"    Feasible pred range: [{feasible_lo:.1f}, {feasible_up:.1f}] "
              f"(width={feasible_width:.1f} mmHg)")
        print(f"    Preds in range: {100*pred_arr.mean():.1f} mean, "
              f"[{np.percentile(pred_arr, 5):.1f}, {np.percentile(pred_arr, 95):.1f}] 90%-CI")
        print(f"    Currently in feasible: {100*in_feas:.1f}%")
        print(f"    Band/Target ratio: {improvement_needed:.2f} "
              f"({'FEASIBLE' if improvement_needed < 1 else f'Need {improvement_needed:.1f}x improvement'})")
        print()


# ============================================================
# SUMMARY TABLE FOR PAPER
# ============================================================

def paper_summary_table(a3_data, mimic_data):
    """Generate the key summary table for the paper."""
    print("\n" + "=" * 80)
    print("PAPER SUMMARY TABLE: CONFORMAL SAFETY CERTIFICATES")
    print("=" * 80)
    print()

    alpha = 0.10

    print("Table: Conformal safety certificates at α=0.10 (90% coverage target)")
    print()
    print("Cancer Simulator (evaluated on oracle ground truth):")
    print(f"  {'γ':>3s} {'τ':>3s} | {'q̂':>6s} | {'Cert%':>6s} | "
          f"{'Conformal-RA':^18s} | {'Oracle-RA':^14s} | {'ELBO':^14s}")
    print(f"  {'':>3s} {'':>3s} | {'':>6s} | {'':>6s} | "
          f"{'Safe':>5s} {'Tgt':>5s} {'TL':>6s} | {'Safe':>5s} {'TL':>6s} | {'Safe':>5s} {'TL':>6s}")
    print(f"  {'-'*80}")

    # Run Cancer with one-sided conformal at alpha=0.10
    for gamma in GAMMAS:
        for tau in TAUS:
            cal_tgt = []
            cal_saf = []
            for seed in CAL_SEEDS:
                if seed not in a3_data[gamma]:
                    continue
                vcip = a3_data[gamma][seed]['VCIP']
                if tau not in vcip:
                    continue
                for case in vcip[tau]:
                    otf = case.get('oracle_traj_features', {})
                    mtf = case.get('model_traj_features', {})
                    if 'cv_terminal' not in otf:
                        continue
                    cal_tgt.extend((otf['cv_terminal'] - mtf['cv_terminal']).tolist())
                    cal_saf.extend((otf['cv_max'] - mtf['cv_max']).tolist())
            if not cal_tgt:
                continue

            q_tgt = conformal_quantile(np.array(cal_tgt), alpha)
            q_saf = conformal_quantile(np.array(cal_saf), alpha)
            q_max = max(q_tgt, q_saf)

            # Test
            elbo_s, elbo_t, elbo_tl = [], [], []
            ora_s, ora_t, ora_tl = [], [], []
            conf_s, conf_t, conf_tl = [], [], []
            cert_rates = []

            for seed in TEST_SEEDS:
                if seed not in a3_data[gamma]:
                    continue
                vcip = a3_data[gamma][seed]['VCIP']
                if tau not in vcip:
                    continue
                for case in vcip[tau]:
                    otf = case.get('oracle_traj_features', {})
                    mtf = case.get('model_traj_features', {})
                    if 'cv_terminal' not in otf:
                        continue
                    ml = case['model_losses']
                    tl = case['true_losses']
                    o_cvt = otf['cv_terminal']
                    o_cvm = otf['cv_max']
                    o_cdm = otf.get('cd_max', np.zeros(len(ml)))
                    m_cvt = mtf['cv_terminal']
                    m_cvm = mtf['cv_max']
                    k = len(ml)

                    cert = ((m_cvt + q_tgt) <= CANCER_TARGET_UPPER) & ((m_cvm + q_saf) <= CANCER_SAFETY_UPPER)
                    cert_rates.append(cert.sum() / k)

                    ei = np.argmin(ml)
                    elbo_s.append(float(o_cvm[ei] <= CANCER_SAFETY_UPPER))
                    elbo_t.append(float(o_cvt[ei] <= CANCER_TARGET_UPPER))
                    elbo_tl.append(tl[ei])

                    o_feas = ((o_cvt <= CANCER_TARGET_UPPER) & (o_cvm <= CANCER_SAFETY_UPPER))
                    if o_cdm is not None and len(o_cdm) == k:
                        o_feas &= (o_cdm <= CANCER_SAFETY_CHEMO)
                    oi = np.argmin(np.where(o_feas, ml, np.inf)) if o_feas.any() else ei
                    ora_s.append(float(o_cvm[oi] <= CANCER_SAFETY_UPPER))
                    ora_t.append(float(o_cvt[oi] <= CANCER_TARGET_UPPER))
                    ora_tl.append(tl[oi])

                    ci = np.argmin(np.where(cert, ml, np.inf)) if cert.any() else ei
                    conf_s.append(float(o_cvm[ci] <= CANCER_SAFETY_UPPER))
                    conf_t.append(float(o_cvt[ci] <= CANCER_TARGET_UPPER))
                    conf_tl.append(tl[ci])

            if not elbo_s:
                continue

            print(f"  {gamma:>3d} {tau:>3d} | {q_max:>6.2f} | {100*np.mean(cert_rates):>5.1f}% | "
                  f"{100*np.mean(conf_s):>4.0f}% {100*np.mean(conf_t):>4.0f}% {np.mean(conf_tl):>.4f} | "
                  f"{100*np.mean(ora_s):>4.0f}% {np.mean(ora_tl):>.4f} | "
                  f"{100*np.mean(elbo_s):>4.0f}% {np.mean(elbo_tl):>.4f}")

    # MIMIC summary
    print(f"\n\nMIMIC-III (evaluated on model-predicted DBP):")
    print(f"  {'τ':>3s} | {'q̂_lo':>5s} {'q̂_up':>5s} {'band':>5s} | {'Cert%':>6s} | "
          f"{'Conf-RA InTgt':>13s} | {'RA InTgt':>8s} | {'ELBO InTgt':>10s}")
    print(f"  {'-'*70}")

    for tau in MIMIC_TAUS:
        cal_lo, cal_up = [], []
        for seed in CAL_SEEDS:
            if seed not in mimic_data or tau not in mimic_data[seed]:
                continue
            for case in mimic_data[seed][tau]:
                obs = case.get('observed_dbp', None)
                if obs is None:
                    continue
                dbp_t = case['traj_features']['dbp_terminal']
                cal_lo.extend((dbp_t - obs).tolist())
                cal_up.extend((obs - dbp_t).tolist())
        if not cal_lo:
            continue
        q_lo = conformal_quantile(np.array(cal_lo), alpha)
        q_up = conformal_quantile(np.array(cal_up), alpha)

        n_pat = 0
        n_cert_cand = 0
        n_tot_cand = 0
        elbo_intgt = 0
        ra_intgt = 0
        conf_intgt = 0

        for seed in TEST_SEEDS:
            if seed not in mimic_data or tau not in mimic_data[seed]:
                continue
            for case in mimic_data[seed][tau]:
                ml = case['model_losses']
                tf = case['traj_features']
                dbp_t = tf['dbp_terminal']
                dbp_traj = tf['dbp_trajectory']
                k = len(ml)
                n_pat += 1
                n_tot_cand += k

                ei = np.argmin(ml)
                if MIMIC_TARGET_LOWER <= dbp_t[ei] <= MIMIC_TARGET_UPPER:
                    elbo_intgt += 1

                ra_feas = ((dbp_t >= MIMIC_TARGET_LOWER) & (dbp_t <= MIMIC_TARGET_UPPER))
                for s in range(dbp_traj.shape[1]):
                    ra_feas &= ((dbp_traj[:, s] >= MIMIC_SAFETY_LOWER) &
                                 (dbp_traj[:, s] <= MIMIC_SAFETY_UPPER))
                if ra_feas.any():
                    ri = np.argmin(np.where(ra_feas, ml, np.inf))
                    if MIMIC_TARGET_LOWER <= dbp_t[ri] <= MIMIC_TARGET_UPPER:
                        ra_intgt += 1
                else:
                    if MIMIC_TARGET_LOWER <= dbp_t[ei] <= MIMIC_TARGET_UPPER:
                        ra_intgt += 1

                cert = ((dbp_t >= MIMIC_TARGET_LOWER + q_lo) &
                        (dbp_t <= MIMIC_TARGET_UPPER - q_up))
                for s in range(dbp_traj.shape[1]):
                    cert &= ((dbp_traj[:, s] >= MIMIC_SAFETY_LOWER + q_lo) &
                              (dbp_traj[:, s] <= MIMIC_SAFETY_UPPER - q_up))
                n_cert_cand += cert.sum()

                if cert.any():
                    ci = np.argmin(np.where(cert, ml, np.inf))
                    if MIMIC_TARGET_LOWER <= dbp_t[ci] <= MIMIC_TARGET_UPPER:
                        conf_intgt += 1
                else:
                    if MIMIC_TARGET_LOWER <= dbp_t[ei] <= MIMIC_TARGET_UPPER:
                        conf_intgt += 1

        if n_pat == 0:
            continue

        print(f"  {tau:>3d} | {q_lo:>5.1f} {q_up:>5.1f} {q_lo+q_up:>5.1f} | "
              f"{100*n_cert_cand/n_tot_cand:>5.1f}% | "
              f"{100*conf_intgt/n_pat:>12.1f}% | {100*ra_intgt/n_pat:>7.1f}% | "
              f"{100*elbo_intgt/n_pat:>9.1f}%")


if __name__ == '__main__':
    print("S6.1 v2: Refined Conformal Safety Certificates\n")

    print("Loading data...")
    a3_data = load_a3_data()
    mimic_data = load_mimic_data()
    n_a3 = sum(len(a3_data[g]) for g in GAMMAS)
    print(f"  Cancer: {n_a3} files, MIMIC: {len(mimic_data)} seeds\n")

    # Cancer analysis
    cancer_score_distribution(a3_data)
    cancer_onesided(a3_data)
    cancer_accuracy_gap(a3_data)

    # MIMIC analysis
    mimic_asymmetric(mimic_data)
    mimic_coverage_asymmetric(mimic_data)
    mimic_accuracy_gap(mimic_data)

    # Paper summary
    paper_summary_table(a3_data, mimic_data)

    print("\n\nS6.1 v2 analysis complete.")
