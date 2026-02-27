# Paper Results to Codebase Mapping

Reference for replicating each result from the VCIP paper (Wang et al., ICML 2025).

## Execution Overview

All scripts run from: `lightning-hydra-template-main/src/vendor/VCIP/`

**Top-level orchestration:**
- `train_all.sh` — runs all models for gamma=1..4, both rank=True and rank=False
- `train_ablation.sh` — runs ablation variants

**Seeds:** (10, 101, 1010, 10101, 101010) — 5 runs per configuration

**Results output:** `csvs/cancer_cont/gamma_${gamma}/mse.csv`

---

## Table 1: Long-range prediction (identical strategies, gamma=4, tau=1..12)

| Model | Script | Runnable | Config |
|-------|--------|----------|--------|
| VCIP | `scripts/cancer/train/train_vae.sh true 4` | `runnables/train_vae.py` | `+model=vcip +model/hparams/cancer=4*` |
| RMSN | `scripts/cancer/train/train_rmsn.sh true 4` | `runnables/train_rmsn.py` | `+baselines=rmsn +baselines/rmsn_hparams/cancer_sim=4*` |
| CRN | `scripts/cancer/train/train_crn.sh true 4` | `runnables/train_enc_dec.py` | `+baselines=crn +baselines/crn_hparams/cancer_sim_domain_conf=4*` |
| CT | `scripts/cancer/train/train_ct.sh true 4` | `runnables/train_multi.py` | `+baselines=ct +baselines/ct_hparams/cancer_sim_domain_conf=4*` |
| ACTIN | `scripts/cancer/train/train_actin.sh true 4` | `runnables/train_actin.py` | `+baselines=actin +baselines/actin_hparams/cancer=4*` |

**Key:** `exp.rank=True` triggers ranking-based evaluation (`optimize_interventions_discrete()`)

---

## Table 2: Long-range prediction (distinct strategies, gamma=4, tau=1..12)

Same as Table 1 but with `exp.rank=False` (first argument to shell scripts).

**Key:** `exp.rank=False` triggers gradient-based continuous optimization (`optimize_interventions()`)

---

## Table 3: Ablation study (with/without confounding adjustment)

**VCIP ablation (gamma=4):**
```bash
# Same as VCIP training but with:
exp.lambda_step=0 exp.lambda_action=0 model.name=VCIP_ablation
```

**RMSN ablation (gamma=4):**
```bash
# Same as RMSN training but with:
exp.lambda_entropy=1 exp.max=0.1 model.name=RMSN_ab
```

**Metrics:** GPR and RCS at tau=2,4; Target distance at gamma=1,2,3,4

---

## Figure 4: GRP and RCS (tumor dataset, gamma=4, tau=2,4,6,8)

Same runs as Tables 1-2, gamma=4. Extract tau=2,4,6,8 from results.
All 5 models compared: VCIP, ACTIN, CT, CRN, RMSN.

---

## Figure 6: Target distances (tumor dataset, gamma=1,2,3, tau=1..6)

Run all models for gamma=1,2,3 with `exp.rank=False`. All 5 models compared.

---

## Key Parameters

| Parameter | What it controls | Where set |
|-----------|-----------------|-----------|
| `dataset.coeff` | Confounding strength (gamma) | Shell script arg, overrides config |
| `exp.rank` | Ranking (True) vs optimization (False) | Shell script first arg |
| `exp.seed` | Random seed | Shell script loop |
| `exp.lambda_step` | Step-level adjustment (VCIP) | Config default=0.1, gamma-specific override |
| `exp.lambda_action` | Action divergence (VCIP) | Config default=1.0 |
| `exp.test` | Test mode flag | Shell script second arg |
| `exp.epochs` | Training epochs | Model-specific (100 VCIP/CT, 150 CRN/ACTIN) |

## GPU Assignment (defaults)

| Model | GPU |
|-------|-----|
| VCIP | 3 |
| RMSN | 0 |
| CRN | 3 |
| CT | 2 |
| ACTIN | 2 |

Overridable via third shell script argument.
