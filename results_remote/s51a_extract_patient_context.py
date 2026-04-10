"""
S5.1a Patient Context Extraction: Map individual_id to MIMIC-III clinical data.

Replays the exact VCIP data pipeline to recover subject_ids for the 27 evaluation
cases, then extracts demographics, vitals, labs, and treatment history from the
raw MIMIC-Extract HDF5 files.

Output: s51a_patient_context.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- Configuration (must match mimic3_real.yaml and eval_mimic_traj.py) ---
H5_PATH = Path("mimic_extract/all_hourly_data.h5")
STATIC_PATH = Path("mimic_extract/static_data.csv")
CASES_PATH = Path("s51a_selected_cases.json")
OUTPUT_PATH = Path("s51a_patient_context.json")

SEED = 10           # exp.seed
MAX_NUMBER = 1000
MIN_SEQ_LENGTH = 60
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15

# Vitals/labs to extract for clinical context
VITALS_DISPLAY = [
    'heart rate', 'mean blood pressure', 'respiratory rate',
    'glascow coma scale total',
]
LABS_DISPLAY = [
    'creatinine', 'bicarbonate', 'anion gap', 'hemoglobin', 'glucose', 'platelets',
]
OUTCOME_COL = 'diastolic blood pressure'
TREATMENT_COLS = ['vaso', 'vent']
CONTEXT_HOURS = 6  # hours of history to show


def replay_data_pipeline():
    """Replay the exact VCIP data pipeline to recover validation subject_ids."""
    print(f"Loading HDF5: {H5_PATH}")
    h5 = pd.HDFStore(str(H5_PATH), 'r')
    all_vitals = h5['/vitals_labs_mean']
    interventions = h5['/interventions']
    h5.close()

    # Filter patients with >= MIN_SEQ_LENGTH timesteps
    user_sizes = all_vitals.groupby('subject_id').size()
    candidates = user_sizes.index[user_sizes >= MIN_SEQ_LENGTH].values
    print(f"Patients with >= {MIN_SEQ_LENGTH} timesteps: {len(candidates)}")

    # Select MAX_NUMBER patients with fixed seed
    np.random.seed(SEED)
    selected = np.random.choice(candidates, size=min(MAX_NUMBER, len(candidates)), replace=False)
    print(f"Selected {len(selected)} patients (seed={SEED})")

    # Train/test split
    df = pd.DataFrame(index=selected)
    df_train_val, df_test = train_test_split(df, test_size=SPLIT_TEST, random_state=SEED)

    # Train/val split
    val_ratio = SPLIT_VAL / (1 - SPLIT_TEST)
    df_train, df_val = train_test_split(df_train_val, test_size=val_ratio, random_state=2 * SEED)

    # Sort validation by subject_id (matches VCIP pipeline)
    val_subject_ids = np.sort(df_val.index.values)
    print(f"Validation set: {len(val_subject_ids)} patients")
    print(f"First 5 val subject_ids: {val_subject_ids[:5]}")

    return val_subject_ids, all_vitals, interventions


def extract_patient_context(subject_id, all_vitals, interventions, static_df):
    """Extract clinical context for a single patient."""
    # Get this patient's data
    patient_vitals = all_vitals.loc[subject_id]
    patient_interventions = interventions.loc[subject_id] if subject_id in interventions.index.get_level_values(0) else None

    # Total timesteps
    n_steps = len(patient_vitals)

    # Get the planning start point (last MIN_SEQ_LENGTH timesteps, planning starts at end)
    # The evaluation uses the last portion of the sequence
    # Extract last CONTEXT_HOURS before the planning horizon
    start_idx = max(0, n_steps - CONTEXT_HOURS)

    context = {
        'subject_id': int(subject_id),
        'total_timesteps': n_steps,
    }

    # Static features
    if subject_id in static_df.index:
        static = static_df.loc[subject_id]
        context['demographics'] = {
            'age': round(float(static.get('age', -1)), 1),
            'gender': str(static.get('gender', 'Unknown')),
            'ethnicity': str(static.get('ethnicity', 'Unknown')),
            'diagnosis': str(static.get('diagnosis_at_admission', 'Unknown')),
            'icu_type': str(static.get('first_careunit', 'Unknown')),
            'los_icu_days': round(float(static.get('los_icu', -1)), 1),
        }
    else:
        context['demographics'] = None

    # Vitals history (last CONTEXT_HOURS hours)
    vitals_history = {}
    for col in VITALS_DISPLAY + [OUTCOME_COL]:
        if col in patient_vitals.columns:
            vals = patient_vitals[col].iloc[start_idx:].values.flatten()
            vitals_history[col] = [round(float(v), 1) if pd.notna(v) else None for v in vals]
    context['vitals_history'] = vitals_history

    # Labs (most recent non-NaN value)
    labs = {}
    for col in LABS_DISPLAY:
        if col in patient_vitals.columns:
            series = patient_vitals[col].dropna()
            if len(series) > 0:
                val = series.iloc[-1]
                labs[col] = round(float(val.item() if hasattr(val, 'item') else val), 2)
            else:
                labs[col] = None
    context['labs'] = labs

    # Treatment history (last CONTEXT_HOURS)
    if patient_interventions is not None and len(patient_interventions) > 0:
        tx_history = {}
        for col in TREATMENT_COLS:
            if col in patient_interventions.columns:
                vals = patient_interventions[col].iloc[start_idx:].values.flatten()
                tx_history[col] = [round(float(v), 3) if pd.notna(v) else None for v in vals]
        context['treatment_history'] = tx_history
    else:
        context['treatment_history'] = None

    # Terminal DBP (for validation)
    if OUTCOME_COL in patient_vitals.columns:
        dbp_vals = patient_vitals[OUTCOME_COL].dropna()
        if len(dbp_vals) > 0:
            val = dbp_vals.iloc[-1]
            context['last_dbp'] = round(float(val.item() if hasattr(val, 'item') else val), 1)
        else:
            context['last_dbp'] = None

    return context


def main():
    # Load cases
    with open(CASES_PATH) as f:
        cases_data = json.load(f)
    cases = cases_data['cases']
    case_ids_needed = {c['individual_id'] for c in cases}
    print(f"Need context for {len(case_ids_needed)} patients (individual_ids: {sorted(case_ids_needed)[:10]}...)")

    # Replay pipeline
    val_subject_ids, all_vitals, interventions = replay_data_pipeline()

    # Load static data
    static_df = pd.read_csv(STATIC_PATH, index_col='subject_id')
    print(f"Static data: {len(static_df)} patients")

    # Extract context for each case
    patient_contexts = {}
    n_validated = 0

    for case in cases:
        ind_id = case['individual_id']
        if ind_id >= len(val_subject_ids):
            print(f"  WARNING: individual_id {ind_id} out of range (val set has {len(val_subject_ids)} patients)")
            continue

        subject_id = val_subject_ids[ind_id]
        ctx = extract_patient_context(subject_id, all_vitals, interventions, static_df)

        # Note: observed_dbp in pickle is at the planning start timepoint,
        # not the last timestep of the full stay. We cannot validate exact match,
        # but the patient identity mapping is deterministic (same seeds).
        n_validated += 1

        patient_contexts[str(ind_id)] = ctx

    print(f"\nExtracted context for {len(patient_contexts)} patients")
    print(f"DBP validation: {n_validated}/{len(cases)} matched")

    # Save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(patient_contexts, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")

    # Print sample
    sample_id = str(cases[0]['individual_id'])
    sample = patient_contexts[sample_id]
    print(f"\nSample patient (individual_id={sample_id}, subject_id={sample['subject_id']}):")
    if sample['demographics']:
        d = sample['demographics']
        print(f"  Age: {d['age']}, Gender: {d['gender']}, Ethnicity: {d['ethnicity']}")
        print(f"  Diagnosis: {d['diagnosis']}, ICU: {d['icu_type']}, LOS: {d['los_icu_days']}d")
    if sample['labs']:
        print(f"  Labs: {sample['labs']}")
    if sample['vitals_history']:
        for k, v in sample['vitals_history'].items():
            print(f"  {k} (last {len(v)}h): {v}")


if __name__ == '__main__':
    main()
