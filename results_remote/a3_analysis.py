"""
A3: Oracle-vs-Model Analysis for Cancer Simulator

Compares RA filtering using:
- Oracle trajectories (ground-truth from simulator)
- Model-predicted trajectories (from decoder at each ELBO step)

This is the KEY experiment addressing W3: does safety filtering work when we
only have model predictions (no oracle access)?

Three-level validation:
1. Oracle on Cancer (upper bound)
2. Model-predicted on Cancer (this analysis - quantifies the gap)
3. Model-predicted on MIMIC (already in paper)
"""

import pickle
import numpy as np
import os
from scipy.stats import spearmanr

BASE_DIR = '/Users/anisiomlacerda/code/target-counterfactual/results_remote/a3_results'
SEEDS = [10, 101, 1010, 10101, 101010]
GAMMAS = [1, 2, 3, 4]
TAUS = [2, 4, 6, 8]

# RA thresholds (Cancer, cm³)
TARGET_UPPER = 3.0
SAFETY_VOLUME_UPPER = 12.0
SAFETY_CHEMO_UPPER = 5.0


def load_data():
    all_data = {}
    for gamma in GAMMAS:
        all_data[gamma] = {}
        for seed in SEEDS:
            p = os.path.join(BASE_DIR, f'coeff_{gamma}/seed_{seed}/a3_oracle_vs_model.pkl')
            if os.path.exists(p):
                try:
                    with open(p, 'rb') as f:
                        raw = pickle.load(f)
                    all_data[gamma][seed] = raw
                except Exception as e:
                    print(f"  WARNING: corrupt {p}: {e}")
    return all_data


def inspect_data_structure(all_data):
    """Print the structure of loaded data."""
    print("=" * 80)
    print("DATA STRUCTURE INSPECTION")
    print("=" * 80)
    for gamma in GAMMAS:
        for seed in SEEDS:
            if seed in all_data[gamma]:
                raw = all_data[gamma][seed]
                print(f"\n  gamma={gamma}, seed={seed}:")
                print(f"    Top keys: {list(raw.keys())}")
                if 'VCIP' in raw:
                    vcip = raw['VCIP']
                    print(f"    Tau keys: {list(vcip.keys())}")
                    if vcip:
                        tau0 = list(vcip.keys())[0]
                        cases = vcip[tau0]
                        print(f"    N cases at tau={tau0}: {len(cases)}")
                        if cases:
                            c = cases[0]
                            print(f"    Case keys: {sorted(c.keys())}")
                            for k, v in c.items():
                                if hasattr(v, 'shape'):
                                    print(f"      {k}: shape {v.shape}")
                                elif isinstance(v, dict):
                                    print(f"      {k}: dict keys {sorted(v.keys())}")
                                    for sk, sv in v.items():
                                        if hasattr(sv, 'shape'):
                                            print(f"        {sk}: shape {sv.shape}")
                                else:
                                    print(f"      {k}: {type(v).__name__} = {v}")
                if 'scaling' in raw:
                    print(f"    Scaling: {raw['scaling']}")
                return  # Just inspect one


def analyze_oracle_vs_model(all_data):
    """Core analysis: compare oracle and model-predicted trajectories."""
    print("\n" + "=" * 80)
    print("A3: ORACLE vs MODEL TRAJECTORY COMPARISON")
    print("=" * 80)

    for gamma in GAMMAS:
        print(f"\n{'='*80}")
        print(f"GAMMA = {gamma}")
        print(f"{'='*80}")

        for tau in TAUS:
            oracle_terminals = []
            model_terminals = []
            oracle_maxs = []
            model_maxs = []
            correlations = []
            n_cases = 0

            for seed in SEEDS:
                if seed not in all_data[gamma]:
                    continue
                raw = all_data[gamma][seed]
                vcip = raw['VCIP']
                if tau not in vcip:
                    continue

                for case in vcip[tau]:
                    otf = case.get('oracle_traj_features', {})
                    mtf = case.get('model_traj_features', {})

                    if 'cv_terminal' in otf and 'cv_terminal' in mtf:
                        o_t = otf['cv_terminal']
                        m_t = mtf['cv_terminal']
                        oracle_terminals.extend(o_t.tolist())
                        model_terminals.extend(m_t.tolist())
                        oracle_maxs.extend(otf['cv_max'].tolist())
                        model_maxs.extend(mtf['cv_max'].tolist())

                        # Per-patient correlation between oracle and model
                        r, _ = spearmanr(o_t, m_t)
                        if not np.isnan(r):
                            correlations.append(r)
                        n_cases += 1

            if not oracle_terminals:
                print(f"\n  tau={tau}: No data")
                continue

            ot = np.array(oracle_terminals)
            mt = np.array(model_terminals)
            om = np.array(oracle_maxs)
            mm = np.array(model_maxs)

            # Global correlation
            global_r, _ = spearmanr(ot, mt)

            print(f"\n  tau={tau} ({n_cases} patients, {len(ot)} total candidates):")
            print(f"    TERMINAL cancer volume (cm³):")
            print(f"      Oracle:  mean={ot.mean():.3f}, std={ot.std():.3f}, "
                  f"range=[{ot.min():.3f}, {ot.max():.3f}]")
            print(f"      Model:   mean={mt.mean():.3f}, std={mt.std():.3f}, "
                  f"range=[{mt.min():.3f}, {mt.max():.3f}]")
            print(f"      Bias (model - oracle): {(mt-ot).mean():+.3f}")
            print(f"      MAE: {np.abs(mt-ot).mean():.3f}")
            print(f"      RMSE: {np.sqrt(((mt-ot)**2).mean()):.3f}")
            print(f"      Global Spearman r: {global_r:.4f}")
            print(f"      Per-patient Spearman r: mean={np.mean(correlations):.4f}, "
                  f"std={np.std(correlations):.4f}")

            print(f"    MAX cancer volume (cm³):")
            print(f"      Oracle: mean={om.mean():.3f}, std={om.std():.3f}")
            print(f"      Model:  mean={mm.mean():.3f}, std={mm.std():.3f}")


def analyze_ra_filtering_comparison(all_data):
    """Compare RA filtering quality: oracle-based vs model-based selection."""
    print("\n" + "=" * 80)
    print("A3: RA FILTERING — ORACLE vs MODEL SELECTION")
    print("=" * 80)
    print("Key question: Does model-based RA filtering still improve safety?")

    for gamma in GAMMAS:
        print(f"\n{'='*80}")
        print(f"GAMMA = {gamma}")
        print(f"{'='*80}")

        for tau in TAUS:
            results = {
                'elbo': {'true_losses': [], 'oracle_safe': [], 'oracle_in_target': []},
                'oracle_ra': {'true_losses': [], 'oracle_safe': [], 'oracle_in_target': []},
                'model_ra': {'true_losses': [], 'oracle_safe': [], 'oracle_in_target': []},
            }
            n_oracle_feas = []
            n_model_feas = []
            n_cases = 0

            for seed in SEEDS:
                if seed not in all_data[gamma]:
                    continue
                raw = all_data[gamma][seed]
                vcip = raw['VCIP']
                if tau not in vcip:
                    continue

                for case in vcip[tau]:
                    ml = case['model_losses']
                    tl = case.get('true_losses', np.zeros(len(ml)))
                    otf = case.get('oracle_traj_features', {})
                    mtf = case.get('model_traj_features', {})

                    if 'cv_terminal' not in otf or 'cv_terminal' not in mtf:
                        continue

                    k = len(ml)
                    n_cases += 1

                    # Oracle features
                    o_cvt = otf['cv_terminal']
                    o_cvm = otf['cv_max']
                    o_cdm = otf.get('cd_max', np.zeros(k))

                    # Model features
                    m_cvt = mtf['cv_terminal']
                    m_cvm = mtf['cv_max']

                    # Oracle feasibility
                    o_feas = (o_cvt <= TARGET_UPPER) & (o_cvm <= SAFETY_VOLUME_UPPER)
                    if o_cdm is not None and len(o_cdm) == k:
                        o_feas &= (o_cdm <= SAFETY_CHEMO_UPPER)

                    # Model feasibility (no cd_max from model predictions)
                    m_feas = (m_cvt <= TARGET_UPPER) & (m_cvm <= SAFETY_VOLUME_UPPER)

                    n_oracle_feas.append(o_feas.sum())
                    n_model_feas.append(m_feas.sum())

                    # 1. ELBO-best (unconstrained)
                    elbo_idx = np.argmin(ml)
                    results['elbo']['true_losses'].append(tl[elbo_idx] if len(tl) > 0 else 0)
                    results['elbo']['oracle_safe'].append(
                        1.0 if (o_cvm[elbo_idx] <= SAFETY_VOLUME_UPPER and
                                (o_cdm is None or o_cdm[elbo_idx] <= SAFETY_CHEMO_UPPER)) else 0.0)
                    results['elbo']['oracle_in_target'].append(
                        1.0 if o_cvt[elbo_idx] <= TARGET_UPPER else 0.0)

                    # 2. Oracle-RA (filter using oracle trajectories)
                    if o_feas.sum() > 0:
                        o_ml = ml.copy()
                        o_ml[~o_feas] = np.inf
                        o_ra_idx = np.argmin(o_ml)
                    else:
                        o_ra_idx = elbo_idx
                    results['oracle_ra']['true_losses'].append(tl[o_ra_idx] if len(tl) > 0 else 0)
                    results['oracle_ra']['oracle_safe'].append(
                        1.0 if (o_cvm[o_ra_idx] <= SAFETY_VOLUME_UPPER and
                                (o_cdm is None or o_cdm[o_ra_idx] <= SAFETY_CHEMO_UPPER)) else 0.0)
                    results['oracle_ra']['oracle_in_target'].append(
                        1.0 if o_cvt[o_ra_idx] <= TARGET_UPPER else 0.0)

                    # 3. Model-RA (filter using model-predicted trajectories)
                    if m_feas.sum() > 0:
                        m_ml = ml.copy()
                        m_ml[~m_feas] = np.inf
                        m_ra_idx = np.argmin(m_ml)
                    else:
                        m_ra_idx = elbo_idx
                    # Evaluate model-RA selection against ORACLE ground truth
                    results['model_ra']['true_losses'].append(tl[m_ra_idx] if len(tl) > 0 else 0)
                    results['model_ra']['oracle_safe'].append(
                        1.0 if (o_cvm[m_ra_idx] <= SAFETY_VOLUME_UPPER and
                                (o_cdm is None or o_cdm[m_ra_idx] <= SAFETY_CHEMO_UPPER)) else 0.0)
                    results['model_ra']['oracle_in_target'].append(
                        1.0 if o_cvt[m_ra_idx] <= TARGET_UPPER else 0.0)

            if n_cases == 0:
                continue

            print(f"\n  tau={tau} ({n_cases} patients):")
            print(f"    Feasible candidates: oracle={np.mean(n_oracle_feas):.1f}/100, "
                  f"model={np.mean(n_model_feas):.1f}/100")

            print(f"\n    {'Method':<15s}  {'TL':>10s}  {'OracleSafe%':>12s}  {'OracleInTgt%':>13s}")
            print(f"    {'-'*55}")

            for method_name, method_key in [('ELBO', 'elbo'), ('Oracle-RA', 'oracle_ra'), ('Model-RA', 'model_ra')]:
                r = results[method_key]
                tl_mean = np.mean(r['true_losses'])
                safe_pct = 100 * np.mean(r['oracle_safe'])
                in_tgt_pct = 100 * np.mean(r['oracle_in_target'])
                print(f"    {method_name:<15s}  {tl_mean:>10.6f}  {safe_pct:>11.1f}%  {in_tgt_pct:>12.1f}%")

            # Oracle gap: how much worse is model-RA vs oracle-RA?
            o_ra_safe = np.mean(results['oracle_ra']['oracle_safe'])
            m_ra_safe = np.mean(results['model_ra']['oracle_safe'])
            elbo_safe = np.mean(results['elbo']['oracle_safe'])
            if o_ra_safe > elbo_safe:
                gap = (o_ra_safe - m_ra_safe) / (o_ra_safe - elbo_safe)
                print(f"\n    Oracle gap (safety): {100*gap:.1f}% "
                      f"(0% = model-RA matches oracle-RA; 100% = model-RA = ELBO)")
            else:
                print(f"\n    Oracle gap: N/A (oracle-RA not better than ELBO)")


def summary_table(all_data):
    """Print paper-ready summary table."""
    print("\n" + "=" * 80)
    print("PAPER-READY SUMMARY TABLE")
    print("=" * 80)
    print("Evaluated on oracle ground truth (all safety/in-target metrics are oracle-verified)")
    print()

    print(f"{'gamma':>5s}  {'tau':>3s}  | {'ELBO':^22s} | {'Oracle-RA':^22s} | {'Model-RA':^22s}")
    print(f"{'':>5s}  {'':>3s}  | {'Safe%':>7s} {'InTgt%':>7s} {'TL':>6s} | "
          f"{'Safe%':>7s} {'InTgt%':>7s} {'TL':>6s} | "
          f"{'Safe%':>7s} {'InTgt%':>7s} {'TL':>6s}")
    print("-" * 85)

    for gamma in GAMMAS:
        for tau in TAUS:
            elbo_safe = []
            elbo_intgt = []
            elbo_tl = []
            o_safe = []
            o_intgt = []
            o_tl = []
            m_safe = []
            m_intgt = []
            m_tl = []

            for seed in SEEDS:
                if seed not in all_data[gamma]:
                    continue
                raw = all_data[gamma][seed]
                vcip = raw['VCIP']
                if tau not in vcip:
                    continue

                for case in vcip[tau]:
                    ml = case['model_losses']
                    tl = case.get('true_losses', np.zeros(len(ml)))
                    otf = case.get('oracle_traj_features', {})
                    mtf = case.get('model_traj_features', {})
                    if 'cv_terminal' not in otf or 'cv_terminal' not in mtf:
                        continue

                    k = len(ml)
                    o_cvt = otf['cv_terminal']
                    o_cvm = otf['cv_max']
                    o_cdm = otf.get('cd_max', np.zeros(k))
                    m_cvt = mtf['cv_terminal']
                    m_cvm = mtf['cv_max']

                    o_feas = (o_cvt <= TARGET_UPPER) & (o_cvm <= SAFETY_VOLUME_UPPER)
                    if o_cdm is not None and len(o_cdm) == k:
                        o_feas &= (o_cdm <= SAFETY_CHEMO_UPPER)
                    m_feas = (m_cvt <= TARGET_UPPER) & (m_cvm <= SAFETY_VOLUME_UPPER)

                    # ELBO
                    ei = np.argmin(ml)
                    elbo_safe.append(1.0 if o_cvm[ei] <= SAFETY_VOLUME_UPPER else 0.0)
                    elbo_intgt.append(1.0 if o_cvt[ei] <= TARGET_UPPER else 0.0)
                    elbo_tl.append(tl[ei])

                    # Oracle-RA
                    oi = np.argmin(np.where(o_feas, ml, np.inf)) if o_feas.any() else ei
                    o_safe.append(1.0 if o_cvm[oi] <= SAFETY_VOLUME_UPPER else 0.0)
                    o_intgt.append(1.0 if o_cvt[oi] <= TARGET_UPPER else 0.0)
                    o_tl.append(tl[oi])

                    # Model-RA
                    mi = np.argmin(np.where(m_feas, ml, np.inf)) if m_feas.any() else ei
                    m_safe.append(1.0 if o_cvm[mi] <= SAFETY_VOLUME_UPPER else 0.0)
                    m_intgt.append(1.0 if o_cvt[mi] <= TARGET_UPPER else 0.0)
                    m_tl.append(tl[mi])

            if not elbo_safe:
                continue

            def fmt_tl(x):
                v = np.mean(x)
                if v < 0.001:
                    return f"{v:.1e}"
                return f"{v:.4f}"

            print(f"{gamma:>5d}  {tau:>3d}  | "
                  f"{100*np.mean(elbo_safe):>6.1f}% {100*np.mean(elbo_intgt):>6.1f}% {fmt_tl(elbo_tl):>6s} | "
                  f"{100*np.mean(o_safe):>6.1f}% {100*np.mean(o_intgt):>6.1f}% {fmt_tl(o_tl):>6s} | "
                  f"{100*np.mean(m_safe):>6.1f}% {100*np.mean(m_intgt):>6.1f}% {fmt_tl(m_tl):>6s}")


if __name__ == '__main__':
    print("A3: Oracle-vs-Model Analysis for Cancer Simulator\n")
    all_data = load_data()

    loaded = sum(len(all_data[g]) for g in GAMMAS)
    print(f"Loaded: {loaded}/20 files")

    inspect_data_structure(all_data)
    analyze_oracle_vs_model(all_data)
    analyze_ra_filtering_comparison(all_data)
    summary_table(all_data)

    print("\n\nAnalysis complete.")
