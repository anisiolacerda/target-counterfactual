# Phase 1: RA Evaluation — Vast.ai Execution Plan

> **v2 — Recalibrated thresholds (Option A).** First run used clinical staging cutoffs
> (target=65.45, safety=523.60) which were ~100x too permissive for the actual simulated volumes.
> All 800k RA scores were log(1)=0. Now using distribution-calibrated thresholds.

## Instance Details

- **SSH:** `<TBD — provision new instance>`
- **GPUs:** 2x RTX 3090 (or similar)
- **Task:** Re-evaluate 20 trained VCIP Cancer models with reach-avoid scoring (no retraining)
- **Estimated time:** ~1 hour (4 gammas × 5 seeds, parallelized across 2 GPUs)
- **Estimated cost:** ~$0.30

### RA Threshold Calibration (Option A)

| Parameter | Old (v1) | New (v2) | Rationale |
|---|---|---|---|
| target_upper | 65.45 | **3.0** | ~p75 volume at t=20-25 (diam 1.8cm) |
| safety_volume_upper | 523.60 | **12.0** | ~p95 volume (diam 2.8cm) |
| safety_chemo_upper | 15.0 | **5.0** | ~p95 cumulative chemo dosage |
| kappa | 10.0 | 10.0 | unchanged |

---

## Step 1: Upload Code + Checkpoints (from local machine)

The VCIP code is ~1.5MB. Model checkpoints are ~4.5MB (20 files × 232KB). Total upload: ~10MB.

```bash
# Create a tarball excluding large/unnecessary directories
cd /Users/anisiomlacerda/code/target-counterfactual/lightning-hydra-template-main/src/vendor/VCIP
tar czf /tmp/vcip_phase1.tar.gz \
    --exclude='my_outputs' \
    --exclude='results' \
    --exclude='imgs' \
    --exclude='data/processed' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    .

# Upload code
scp -P 20525 /tmp/vcip_phase1.tar.gz root@70.69.192.6:/root/

# Upload trained checkpoints (only VCIP model.ckpt + hparams.yaml)
cd /Users/anisiomlacerda/code/target-counterfactual/results_remote
tar czf /tmp/vcip_checkpoints.tar.gz \
    $(find my_outputs/cancer_sim_cont/22/ -path "*/VCIP/train/True/models/*/model.ckpt" -o -path "*/VCIP/train/True/models/*/hparams.yaml" | sort)

scp -P 20525 /tmp/vcip_checkpoints.tar.gz root@70.69.192.6:/root/
```

---

## Step 2: Instance Setup (on Vast.ai)

```bash
ssh -p 20525 root@70.69.192.6

# Create project directory and extract code
mkdir -p /root/VCIP && cd /root/VCIP
tar xzf /root/vcip_phase1.tar.gz

# Extract checkpoints into expected location
cd /root/VCIP
tar xzf /root/vcip_checkpoints.tar.gz

# Verify checkpoints
ls my_outputs/cancer_sim_cont/22/coeff_*/VCIP/train/True/models/*/model.ckpt
# Should show 20 files (5 seeds × 4 gammas)
```

### 2.1 Create conda environment

```bash
# If conda not available, install miniconda
which conda || {
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
    eval "$(/root/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    source ~/.bashrc
}

cd /root/VCIP
conda create -n vcip python=3.10 -y
conda activate vcip
pip install -r requirements_vcip.txt
```

### 2.2 Verify GPU access

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
# Expected: CUDA: True, GPUs: 2
```

---

## Step 3: Smoke Test (~2 min)

Run a single seed/gamma to verify everything works:

```bash
cd /root/VCIP
conda activate vcip

CUDA_VISIBLE_DEVICES=0 python runnables/train_vae.py \
    +dataset=cancer_sim_cont \
    +model=vcip \
    +model/hparams/cancer=4* \
    exp.seed=10 \
    exp.epochs=100 \
    dataset.coeff=4 \
    model.name=VCIP \
    exp.test=False \
    exp.rank=True \
    exp.reach_avoid.target_upper=3.0 \
    exp.reach_avoid.safety_volume_upper=12.0 \
    exp.reach_avoid.safety_chemo_upper=5.0 \
    exp.reach_avoid.kappa=10.0
```

Verify output: `my_outputs/cancer_sim_cont/22/coeff_4/VCIP/train/True/case_infos/10/False/case_infos_VCIP.pkl` should exist and contain `true_ra_scores`.

```bash
python -c "
import pickle
with open('my_outputs/cancer_sim_cont/22/coeff_4/VCIP/train/True/case_infos/10/False/case_infos_VCIP.pkl', 'rb') as f:
    data = pickle.load(f)
ci = list(data['VCIP'].values())[0][0]
print('Keys:', list(ci.keys()))
print('Has RA scores:', 'true_ra_scores' in ci)
if 'true_ra_scores' in ci:
    import numpy as np
    ra = ci['true_ra_scores']
    print(f'RA scores: shape={ra.shape}, min={ra.min():.3f}, max={ra.max():.3f}, mean={ra.mean():.3f}')
    print(f'Unique scores: {len(np.unique(np.round(ra, 2)))}')
    print(f'Non-trivial (> -60): {(ra > -60).mean():.1%}')
    if 'ra_diagnostics' in ci:
        print(f'Diagnostics: {ci[\"ra_diagnostics\"]}')
"
```

---

## Step 4: Run Full Phase 1 (~1 hour)

```bash
cd /root/VCIP
nohup bash scripts/phase1/run_phase1.sh > phase1.log 2>&1 &

# Monitor progress
tail -f phase1.log
# Or check GPU utilization
watch -n 5 nvidia-smi
```

### Expected output structure

After completion, 20 augmented case_infos files:
```
my_outputs/cancer_sim_cont/22/coeff_{1,2,3,4}/VCIP/train/True/case_infos/{10,101,1010,10101,101010}/False/case_infos_VCIP.pkl
```

---

## Step 5: Download Results (from local machine)

```bash
# From local machine
cd /Users/anisiomlacerda/code/target-counterfactual

# Download all case_infos
scp -P 20525 -r root@70.69.192.6:/root/VCIP/my_outputs/cancer_sim_cont/22/ \
    results_remote/phase1_ra/cancer_sim_cont/22/

# Or as a tarball (faster)
ssh -p 20525 root@70.69.192.6 "cd /root/VCIP && tar czf /tmp/phase1_results.tar.gz \
    \$(find my_outputs/cancer_sim_cont/22/ -name 'case_infos_VCIP.pkl')"
scp -P 20525 root@70.69.192.6:/tmp/phase1_results.tar.gz /tmp/

cd /Users/anisiomlacerda/code/target-counterfactual/results_remote
mkdir -p phase1_ra
cd phase1_ra
tar xzf /tmp/phase1_results.tar.gz
```

---

## Step 6: Run Analysis Notebook (local)

Open `VCIP/results/phase1/analysis.ipynb` and update the `RESULTS_BASE` path to point to `phase1_ra/`.

Key metrics to evaluate at decision gate:
- **E1:** Does RA Top-1 agreement improve over ELBO's 19% at tau=2?
- **E2:** Are RA margins larger (more robust to ranking inversions)?
- **E3:** Is RA ranking more stable across seeds?

---

## Checklist

- [ ] Upload code tarball
- [ ] Upload checkpoint tarball
- [ ] Create conda env and install deps
- [ ] Verify GPU access
- [ ] Smoke test (single seed, gamma=4)
- [ ] Verify RA scores in output pkl
- [ ] Run full Phase 1 (all gammas, 2 GPUs)
- [ ] Download results
- [ ] Run analysis notebook
- [ ] Decision gate: proceed to Phase 2?
