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

Download the full MIMIC-III v1.4 dataset (~6.7 GB compressed). The relevant tables are:

- `CHARTEVENTS.csv.gz` — vital signs and lab results
- `LABEVENTS.csv.gz` — laboratory measurements
- `INPUTEVENTS_MV.csv.gz` — medications/fluids (MetaVision ICU system)
- `INPUTEVENTS_CV.csv.gz` — medications/fluids (CareVue ICU system)
- `ICUSTAYS.csv.gz` — ICU stay information
- `PATIENTS.csv.gz` — patient demographics
- `ADMISSIONS.csv.gz` — hospital admissions

## 3. Preprocessing Pipeline

The `all_hourly_data.h5` file is produced by an external preprocessing pipeline (not included in this repository). The canonical source is the RMSN preprocessing pipeline from Lim et al. (2018).

**Recommended pipeline:** Clone and run the preprocessing from one of:
- `github.com/sjblim/rmsn_nips_2018` (RMSN original)
- `github.com/MLforHealth/ClinicalNotesICU` (alternative)

The pipeline aggregates raw MIMIC-III data into hourly windows per ICU stay, producing an HDFStore with the following keys:

### Expected HDF5 Schema

| Key | Contents | Index |
|-----|----------|-------|
| `/interventions` | Vasopressor and ventilation indicators | `(subject_id, hadm_id, icustay_id, hours_in)` |
| `/vitals_labs_mean` | Hourly-averaged vitals and lab values | `(subject_id, hadm_id, icustay_id, hours_in)` |
| `/patients` | Static patient features (gender, ethnicity, age) | `(subject_id, hadm_id, icustay_id)` |

### Required Columns

**Treatments** (`/interventions`): `vaso`, `vent`

**Outcome** (`/vitals_labs_mean`): `diastolic blood pressure`

**Vitals/Labs** (`/vitals_labs_mean`): heart rate, red blood cell count, sodium, mean blood pressure, systemic vascular resistance, glucose, chloride urine, glascow coma scale total, hematocrit, positive end-expiratory pressure set, respiratory rate, prothrombin time pt, cholesterol, hemoglobin, creatinine, blood urea nitrogen, bicarbonate, calcium ionized, partial pressure of carbon dioxide, magnesium, anion gap, phosphorous, venous pvo2, platelets, calcium urine

**Static** (`/patients`): gender, ethnicity, age

## 4. File Placement

Place the preprocessed file at:
```
lightning-hydra-template-main/src/vendor/VCIP/data/processed/all_hourly_data.h5
```

## 5. Verification

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

## 6. Dataset Configuration

The MIMIC-III experiment configuration is at:
`VCIP/configs/dataset/mimic3_real.yaml`

Key parameters:
- `max_number: 1000` — maximum patients in cohort
- `min_seq_length: 60` / `max_seq_length: 60` — hourly sequence length
- `projection_horizon: 5` — supports tau up to 6
- `treatment_mode: multilabel` — binary vasopressor + ventilation
- `treatment_size: 2`, `input_size: 25`, `output_size: 1`, `static_size: 44`
