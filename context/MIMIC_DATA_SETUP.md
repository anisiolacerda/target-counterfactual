# MIMIC-III Data Setup

## Overview

The VCIP MIMIC-III experiments use an hourly-averaged ICU dataset stored as an HDF5 file. This document describes how to obtain and prepare the data.

**Target file:** `lightning-hydra-template-main/src/vendor/VCIP/data/processed/all_hourly_data.h5`

## 1. PhysioNet Access

1. Create an account at [PhysioNet](https://physionet.org/)
2. Complete the CITI "Data or Specimens Only Research" training
3. Submit credentialing application at `physionet.org/settings/credentialing/`
4. Once approved, sign the data use agreement for [MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/)

## 2. Download MIMIC-III

Download the **full** MIMIC-III v1.4 dataset (~6.7 GB compressed, all 26 CSV files). The entire database is required because the preprocessing pipeline uses materialized views with cross-table joins.

Key tables consumed by the pipeline:

| Table | Purpose |
|-------|---------|
| `CHARTEVENTS.csv.gz` | Vitals (heart rate, blood pressure, etc.) |
| `LABEVENTS.csv.gz` | Lab results (creatinine, glucose, hemoglobin, etc.) |
| `INPUTEVENTS_MV.csv.gz` | Medications/interventions — MetaVision (vasopressors) |
| `INPUTEVENTS_CV.csv.gz` | Medications/interventions — CareVue (vasopressors) |
| `PROCEDUREEVENTS_MV.csv.gz` | Procedures (mechanical ventilation) |
| `ICUSTAYS.csv.gz` | ICU stay metadata |
| `PATIENTS.csv.gz` | Demographics (gender, age) |
| `ADMISSIONS.csv.gz` | Admission info (ethnicity) |

## 3. Preprocessing Pipeline: MIMIC-Extract

The `all_hourly_data.h5` file is produced by [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract) (Wang et al., 2020). The VCIP codebase does **not** include preprocessing code — it expects this file to already exist.

### Prerequisites

- **PostgreSQL 9.4+** with the full MIMIC-III database loaded (~50 GB)
- **RAM:** minimum 50 GB
- **Time:** 5-10 hours for full extraction
- **conda** package manager

### Step-by-step

#### 3.1 Load MIMIC-III into PostgreSQL

```bash
# Create database
psql -c "CREATE DATABASE mimic;"

# Load all CSV files using the MIT-LCP loader scripts
# See: https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres
```

#### 3.2 Build materialized views (concepts)

```bash
git clone https://github.com/MIT-LCP/mimic-code.git
cd mimic-code/concepts

# Build PostgreSQL functions and concepts
psql -d mimic -f postgres-functions.sql
bash postgres_make_concepts.sh

# Build extended concepts (required by MIMIC-Extract)
cd utils
bash postgres_make_extended_concepts.sh
psql -d mimic -f niv-durations.sql
```

If you get `relation "code_status" does not exist`, the concepts were not built successfully — re-run `postgres_make_concepts.sh`.

#### 3.3 Run MIMIC-Extract

```bash
git clone https://github.com/MLforHealth/MIMIC_Extract.git
cd MIMIC_Extract

# Create conda environment
conda env create --force -f mimic_extract_env_py36.yml
conda activate mimic_data_extraction

# Install additional packages if needed
pip install datapackage spacy scispacy

# Run extraction (produces all_hourly_data.h5)
python mimic_direct_extract.py \
    --output_dir /path/to/output/ \
    --pop_size 0  # 0 = all patients, set smaller for testing
```

The output file `all_hourly_data.h5` will be in the specified output directory.

### Cohort selection criteria (applied by MIMIC-Extract)

- First ICU visit per patient only
- Patient age >= 15
- ICU stay duration >= 12 hours and < 10 days

### Alternative: request pre-built file

The MIMIC-Extract output file may be shared between researchers under the same PhysioNet data use agreement. Consider contacting the VCIP paper authors for their pre-processed copy, which would save the PostgreSQL setup entirely.

## 4. Expected HDF5 Schema

The file must contain four tables (the VCIP code reads three of them):

| HDF5 Key | Contents | Index | Used by VCIP |
|----------|----------|-------|--------------|
| `/patients` | Static demographics and outcomes | `(subject_id, hadm_id, icustay_id)` | Yes |
| `/vitals_labs_mean` | Hourly-averaged vitals and labs (means) | `(subject_id, hadm_id, icustay_id, hours_in)` | Yes |
| `/interventions` | Binary hourly intervention indicators | `(subject_id, hadm_id, icustay_id, hours_in)` | Yes |
| `/vitals_labs` | Hourly vitals/labs with mean, count, std | Same as above | No |

### Required columns

**Treatments** (`/interventions`): `vaso`, `vent`

**Outcome** (`/vitals_labs_mean`): `diastolic blood pressure`

**Vitals/Labs** (`/vitals_labs_mean`): heart rate, red blood cell count, sodium, mean blood pressure, systemic vascular resistance, glucose, chloride urine, glascow coma scale total, hematocrit, positive end-expiratory pressure set, respiratory rate, prothrombin time pt, cholesterol, hemoglobin, creatinine, blood urea nitrogen, bicarbonate, calcium ionized, partial pressure of carbon dioxide, magnesium, anion gap, phosphorous, venous pvo2, platelets, calcium urine

**Static** (`/patients`): gender, ethnicity, age

## 5. File Placement

Place the preprocessed file at:
```
lightning-hydra-template-main/src/vendor/VCIP/data/processed/all_hourly_data.h5
```

## 6. Verification

Run the built-in verification from the VCIP directory:

```bash
cd lightning-hydra-template-main/src/vendor/VCIP
conda activate target-counterfactual-env
python src/data/mimic_iii/load_data.py
```

This calls `load_mimic3_data_processed()` with `min_seq_length=100, max_seq_length=100` and logs the number of filtered patients. Expected: 500+ patients.

Alternatively, verify the HDF5 schema directly:

```python
import pandas as pd
h5 = pd.HDFStore('data/processed/all_hourly_data.h5', 'r')
print(h5.keys())          # Should contain: /interventions, /vitals_labs_mean, /patients
print(h5['/interventions'].columns.tolist())  # Should contain: vaso, vent
print(h5['/patients'].columns.tolist())       # Should contain: gender, ethnicity, age
h5.close()
```

## 7. Dataset Configuration

The MIMIC-III experiment configuration is at:
`VCIP/configs/dataset/mimic3_real.yaml`

Key parameters:
- `max_number: 1000` — maximum patients in cohort
- `min_seq_length: 60` / `max_seq_length: 60` — hourly sequence length
- `projection_horizon: 5` — supports tau up to 6
- `treatment_mode: multilabel` — binary vasopressor + ventilation
- `treatment_size: 2`, `input_size: 25`, `output_size: 1`, `static_size: 44`

## References

- [MIMIC-Extract (MLforHealth/MIMIC_Extract)](https://github.com/MLforHealth/MIMIC_Extract)
- [MIMIC-III v1.4 on PhysioNet](https://physionet.org/content/mimiciii/1.4/)
- [MIT-LCP mimic-code](https://github.com/MIT-LCP/mimic-code)
