"""
S5.1a Case Selection: Select 40 cases for clinician plausibility assessment.

Selects 20 discordant (ELBO ≠ RA) and 20 concordant (ELBO = RA) cases
from MIMIC-III evaluation data at τ=3, seed=10.

Output: s51a_selected_cases.json
"""

import pickle
import json
import numpy as np
from pathlib import Path

# --- Configuration ---
SEED = 10
TAU = 2
N_DISCORDANT = 50  # will be capped to available truly-discordant cases
N_CONCORDANT = 0
MIN_L1 = 0.01  # minimum treatment L1 distance to count as truly discordant
DBP_TARGET = (60, 90)   # mmHg
DBP_SAFETY = (40, 120)  # mmHg

PICKLE_PATH = Path("mimic_ra/VCIP/train/case_infos") / str(SEED) / "False" / "case_infos_VCIP.pkl"
OUTPUT_PATH = Path("s51a_selected_cases.json")


def compute_ra_selection(ci):
    """Compute ELBO and RA selections for a case."""
    losses = ci['model_losses']
    dbp_term = ci['traj_features']['dbp_terminal']
    dbp_min = ci['traj_features']['dbp_min']
    dbp_max = ci['traj_features']['dbp_max']

    elbo_idx = int(np.argmin(losses))

    # RA feasibility: terminal in target AND all intermediate in safety
    feasible = (
        (dbp_term >= DBP_TARGET[0]) & (dbp_term <= DBP_TARGET[1]) &
        (dbp_min >= DBP_SAFETY[0]) & (dbp_max <= DBP_SAFETY[1])
    )

    if feasible.any():
        feasible_indices = np.where(feasible)[0]
        ra_idx = int(feasible_indices[np.argmin(losses[feasible])])
    else:
        ra_idx = elbo_idx  # fallback

    return elbo_idx, ra_idx, feasible.sum()


def extract_case_data(ci, elbo_idx, ra_idx):
    """Extract treatment sequences and predicted outcomes for both selections."""
    seqs = ci['treatment_features']['sequences']  # (100, τ, 2)

    return {
        'individual_id': int(ci['individual_id']),
        'observed_dbp': float(ci['observed_dbp']),
        'n_feasible': int(compute_ra_selection(ci)[2]),
        'elbo': {
            'candidate_idx': elbo_idx,
            'model_loss': float(ci['model_losses'][elbo_idx]),
            'treatment_sequence': seqs[elbo_idx].tolist(),  # (τ, 2) — [vaso, vent] per step
            'dbp_terminal': float(ci['traj_features']['dbp_terminal'][elbo_idx]),
            'dbp_trajectory': ci['traj_features']['dbp_trajectory'][elbo_idx].tolist(),
            'dbp_min': float(ci['traj_features']['dbp_min'][elbo_idx]),
            'dbp_max': float(ci['traj_features']['dbp_max'][elbo_idx]),
            'vaso_total': float(ci['treatment_features']['vaso_total'][elbo_idx]),
            'vent_total': float(ci['treatment_features']['vent_total'][elbo_idx]),
        },
        'ra': {
            'candidate_idx': ra_idx,
            'model_loss': float(ci['model_losses'][ra_idx]),
            'treatment_sequence': seqs[ra_idx].tolist(),
            'dbp_terminal': float(ci['traj_features']['dbp_terminal'][ra_idx]),
            'dbp_trajectory': ci['traj_features']['dbp_trajectory'][ra_idx].tolist(),
            'dbp_min': float(ci['traj_features']['dbp_min'][ra_idx]),
            'dbp_max': float(ci['traj_features']['dbp_max'][ra_idx]),
            'vaso_total': float(ci['treatment_features']['vaso_total'][ra_idx]),
            'vent_total': float(ci['treatment_features']['vent_total'][ra_idx]),
        },
        'observed_treatment': ci['true_sequence'].squeeze().tolist(),  # (τ+offset, 2)
    }


def l1_distance(case_data):
    """L1 distance between ELBO and RA treatment sequences."""
    elbo_seq = np.array(case_data['elbo']['treatment_sequence'])
    ra_seq = np.array(case_data['ra']['treatment_sequence'])
    return float(np.mean(np.abs(elbo_seq - ra_seq)))


def main():
    print(f"Loading pickle: {PICKLE_PATH}")
    with open(PICKLE_PATH, 'rb') as f:
        data = pickle.load(f)

    cases = data['VCIP'][TAU]
    print(f"Loaded {len(cases)} patients at τ={TAU}, seed={SEED}")

    # Classify all cases
    discordant = []
    concordant = []

    for ci in cases:
        elbo_idx, ra_idx, n_feas = compute_ra_selection(ci)
        case_data = extract_case_data(ci, elbo_idx, ra_idx)

        if elbo_idx != ra_idx:
            case_data['l1_distance'] = l1_distance(case_data)
            discordant.append(case_data)
        else:
            case_data['l1_distance'] = 0.0
            concordant.append(case_data)

    print(f"Discordant: {len(discordant)}, Concordant: {len(concordant)}")

    # Filter to truly discordant (different treatment sequences, not just indices)
    truly_discordant = [c for c in discordant if c['l1_distance'] >= MIN_L1]
    print(f"Truly discordant (L1 >= {MIN_L1}): {len(truly_discordant)}")

    # Select top-N by L1 distance
    truly_discordant.sort(key=lambda x: -x['l1_distance'])
    selected_discordant = truly_discordant[:N_DISCORDANT]

    print(f"\nTop {N_DISCORDANT} discordant cases (by L1 distance):")
    for i, c in enumerate(selected_discordant):
        print(f"  {i+1}. Patient {c['individual_id']}: L1={c['l1_distance']:.3f}, "
              f"obs_DBP={c['observed_dbp']:.0f}, "
              f"ELBO_DBP={c['elbo']['dbp_terminal']:.1f}, "
              f"RA_DBP={c['ra']['dbp_terminal']:.1f}")

    # Select concordant controls matched on observed DBP distribution
    disc_dbp = [c['observed_dbp'] for c in selected_discordant]
    disc_dbp_mean = np.mean(disc_dbp)

    # Sort concordant by proximity to discordant mean DBP
    concordant.sort(key=lambda x: abs(x['observed_dbp'] - disc_dbp_mean))
    selected_concordant = concordant[:N_CONCORDANT]

    print(f"\nTop {N_CONCORDANT} concordant controls:")
    for i, c in enumerate(selected_concordant):
        print(f"  {i+1}. Patient {c['individual_id']}: obs_DBP={c['observed_dbp']:.0f}, "
              f"ELBO_DBP={c['elbo']['dbp_terminal']:.1f}")

    # Randomize A/B assignment for blinding
    rng = np.random.RandomState(42)
    all_cases = []

    for case_data in selected_discordant + selected_concordant:
        is_discordant = case_data['l1_distance'] > 0
        # Randomly assign ELBO→A or ELBO→B
        elbo_is_a = bool(rng.randint(2))

        card = {
            'case_type': 'discordant' if is_discordant else 'concordant',
            'individual_id': case_data['individual_id'],
            'observed_dbp': case_data['observed_dbp'],
            'n_feasible': case_data['n_feasible'],
            'l1_distance': case_data['l1_distance'],
            'observed_treatment': case_data['observed_treatment'],
            # Blinded assignment
            'plan_a': case_data['elbo'] if elbo_is_a else case_data['ra'],
            'plan_b': case_data['ra'] if elbo_is_a else case_data['elbo'],
            # Answer key (hidden from clinicians)
            '_answer_key': {
                'plan_a_is': 'elbo' if elbo_is_a else 'ra',
                'plan_b_is': 'ra' if elbo_is_a else 'elbo',
            }
        }
        all_cases.append(card)

    # Shuffle case order
    rng.shuffle(all_cases)

    # Assign case IDs
    for i, card in enumerate(all_cases):
        card['case_id'] = f"C{i+1:02d}"

    # Summary
    n_disc = sum(1 for c in all_cases if c['case_type'] == 'discordant')
    n_conc = sum(1 for c in all_cases if c['case_type'] == 'concordant')
    print(f"\n=== Final selection: {len(all_cases)} cases ({n_disc} discordant, {n_conc} concordant) ===")

    # Save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump({
            'config': {
                'seed': SEED, 'tau': TAU,
                'n_discordant': N_DISCORDANT, 'n_concordant': N_CONCORDANT,
                'dbp_target': DBP_TARGET, 'dbp_safety': DBP_SAFETY,
            },
            'cases': all_cases,
        }, f, indent=2)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
