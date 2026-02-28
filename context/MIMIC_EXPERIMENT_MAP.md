# MIMIC-III Experiment Map

Reference for replicating the VCIP paper's MIMIC-III results (Wang et al., ICML 2025).

## Execution Overview

All scripts run from: `lightning-hydra-template-main/src/vendor/VCIP/`

**Seeds:** (10, 101, 1010, 10101, 101010) — 5 runs per configuration

**Results output:** `my_outputs/mimic_real/{model_name}/train/`

**CSV output:** `csvs/mimic_real/`

---

## Dataset

- **Source:** MIMIC-III v1.4 (real ICU data)
- **Treatments:** vasopressor (vaso), mechanical ventilation (vent) — binary, multilabel
- **Outcome:** diastolic blood pressure
- **Covariates:** 25 time-varying vitals/labs, 44 static features (one-hot encoded)
- **Cohort:** up to 1000 patients, 60 hourly time steps
- **Config:** `configs/dataset/mimic3_real.yaml`

---

## Model Runs

| Model | Script | Runnable | Config | Conda Env |
|-------|--------|----------|--------|-----------|
| VCIP | `scripts/mimic/train/train_vae.sh` | `runnables/train_vae.py` | `+model=vcip +model/hparams/mimic=0*` | target-counterfactual-env |
| CRN | `scripts/mimic/train/train_crn.sh` | `runnables/train_enc_dec.py` | `+baselines=crn +baselines/crn_hparams/mimic3_real=diastolic_blood_pressure` | baseline-env |
| CT | `scripts/mimic/train/train_ct.sh` | `runnables/train_multi.py` | `+baselines=ct +baselines/ct_hparams/mimic3_real=diastolic_blood_pressure` | baseline-env |
| RMSN | `scripts/mimic/train/train_rmsn.sh` | `runnables/train_rmsn.py` | `+baselines=rmsn +baselines/rmsn_hparams/mimic3_real=diastolic_blood_pressure` | baseline-env |
| ACTIN | `scripts/mimic/train/train_actin.sh` | `runnables/train_actin.py` | `+baselines=actin +baselines/actin_hparams/mimic3_real=diastolic_blood_pressure` | target-counterfactual-env |

---

## Evaluation Metrics

Unlike the Cancer dataset (which has a ground-truth simulator), MIMIC-III uses only observational evaluation:

| Metric | Description | Computed by |
|--------|-------------|-------------|
| GRP (Goal-Reaching Probability) | Normalized rank of true observed sequence among 100 sampled interventions | `optimize_interventions_discrete()` in `vae_model.py` |
| RCS (Rank Correlation with Surrogate) | Spearman correlation between model-predicted and observed outcome rankings | Same as above |
| MSE | One-step-ahead prediction error | Training loop, saved to `csvs/mimic_real/mse.csv` |

**Key difference from Cancer:** No counterfactual ground truth is available. Evaluation relies on ranking-based metrics only (`exp.rank=True` is the default in `mimic3_real.yaml`).

---

## Smoke Test

Single seed, 5 epochs:
```bash
cd lightning-hydra-template-main/src/vendor/VCIP
conda activate target-counterfactual-env
python runnables/train_vae.py +dataset=mimic3_real +model=vcip \
  +model/hparams/mimic=0* exp.seed=10 exp.epochs=5 model.name=VCIP
```

---

## Full Experiment Sequence

```bash
# 1. VCIP (5 seeds, 100 epochs)
bash scripts/mimic/train/train_vae.sh 0

# 2. CRN (5 seeds, 100 epochs)
bash scripts/mimic/train/train_crn.sh 0.01 0

# 3. CT (5 seeds, 100 epochs)
bash scripts/mimic/train/train_ct.sh 0.01 0

# 4. RMSN (5 seeds, 100 epochs)
bash scripts/mimic/train/train_rmsn.sh 0.01 0

# 5. ACTIN (5 seeds, 100 epochs)
bash scripts/mimic/train/train_actin.sh 0.01 0
```

Arguments: first is alpha/lambda_D (regularization), second is GPU index.

---

## Output Directory Structure

```
my_outputs/mimic_real/
├── VCIP/train/
│   └── models/{seed}/model.ckpt
├── CRN/{alpha}/train/
├── CT/{alpha}/train/
├── RMSN/train/
└── ACTIN/{lambda_D}/train/
```

---

## Key Differences from Cancer Experiments

| Aspect | Cancer | MIMIC-III |
|--------|--------|-----------|
| Data type | Synthetic (generated at runtime) | Real (requires HDF5 file) |
| Treatment type | Continuous (chemo + radio doses) | Binary multilabel (vaso + vent) |
| Counterfactual evaluation | Yes (oracle simulator) | No (ranking-based only) |
| Confounding parameter | gamma = 1..4 | Not applicable (real confounding) |
| Output directory | `my_outputs/cancer_sim_cont/22/coeff_{gamma}/` | `my_outputs/mimic_real/` |
| Default evaluation mode | Both rank and optimize | Rank only (`exp.rank=True`) |

---

## Notes

- The `exp.device: cuda` setting in `mimic3_real.yaml` may need to be overridden for local development on macOS: add `exp.device=mps exp.gpus=0` to the command line.
- ACTIN hparam file was renamed from `mimic_real.yaml` to `diastolic_blood_pressure.yaml` for consistency with other baselines.
