# MIMIC-III Experiments — Vast.ai Execution Plan

## Instance Details

- **SSH:** `ssh -p 63498 root@142.112.39.215 -L 8080:localhost:8080`
- **GPUs:** 2x RTX 3090 (24 GB VRAM each)
- **CPU:** AMD Ryzen Threadripper 2950X (16 cores / 32 threads)
- **RAM:** 64 GB
- **Disk:** ~880 GB (FIKWOT FN501 Pro 2TB)
- **Cost:** $0.265/hr
- **Max duration:** 14 days

## Estimated Budget

| Phase | Duration | Cost |
|-------|----------|------|
| Upload MIMIC-III CSVs (~6.7 GB) | ~1 hr | $0.27 |
| PostgreSQL setup + CSV loading | ~2-3 hrs | $0.53-0.80 |
| MIMIC-Extract concepts + extraction | ~5-10 hrs | $1.33-2.65 |
| Conda env setup | ~0.5 hr | $0.13 |
| Training (25 runs, 2 GPUs parallel) | ~8-10 hrs | $2.12-2.65 |
| Buffer (debugging, analysis) | ~3 hrs | $0.80 |
| **Total** | **~20-27 hrs** | **~$5-7** |

---

## Phase 1: Upload MIMIC-III Data from Local Machine

From your **local machine**, upload the MIMIC-III CSV files:

```bash
scp -P 63498 /Users/anisiomlacerda/Downloads/mimic-iii/*.csv.gz root@142.112.39.215:/root/mimic-iii-data/
```

---

## Phase 2: Instance Setup (on Vast.ai)

### 2.1 Install system dependencies

```bash
apt-get update && apt-get install -y \
    postgresql postgresql-contrib \
    git wget curl \
    python3-pip python3-venv \
    libhdf5-dev
```

### 2.2 Install conda (if not pre-installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc
```

---

## Phase 3: PostgreSQL + MIMIC-Extract

### 3.1 Set up PostgreSQL and load MIMIC-III

```bash
# Start PostgreSQL
service postgresql start
sudo -u postgres createuser -s root
createdb mimic

# Clone the MIMIC-III loader scripts
git clone https://github.com/MIT-LCP/mimic-code.git /root/mimic-code

# Load CSVs into PostgreSQL
# The loader scripts expect uncompressed CSVs in a directory
mkdir -p /root/mimic-iii-csv
cd /root/mimic-iii-data
gunzip -k *.csv.gz  # Keep originals, decompress alongside
mv *.csv /root/mimic-iii-csv/

# Run the PostgreSQL build script
cd /root/mimic-code/mimic-iii/buildmimic/postgres
# Edit postgres_create_tables.sql and postgres_load_data.sql if needed
# to point to /root/mimic-iii-csv/
make mimic-gz datadir=/root/mimic-iii-data/
```

**Note:** The exact loading command depends on the mimic-code version. Consult the README in `mimic-code/mimic-iii/buildmimic/postgres/`. The key is that all 26 tables get loaded into the `mimic` database.

After confirming successful loading (e.g., `psql -d mimic -c "SELECT count(*) FROM mimiciii.patients;"`), reclaim disk space:

```bash
# Delete uncompressed CSVs (~40+ GB, CHARTEVENTS alone is ~33 GB)
rm -rf /root/mimic-iii-csv/
# Optionally delete compressed originals too if space is tight
# rm -rf /root/mimic-iii-data/*.csv.gz
```

### 3.2 Build materialized views (concepts)

```bash
cd /root/mimic-code/mimic-iii/concepts

# Build functions and concepts
psql -d mimic -f postgres-functions.sql
bash postgres_make_concepts.sh

# Build extended concepts (required by MIMIC-Extract)
cd /root/mimic-code/mimic-iii/concepts/utils
bash postgres_make_extended_concepts.sh
psql -d mimic -f niv-durations.sql
```

### 3.3 Run MIMIC-Extract

```bash
git clone https://github.com/MLforHealth/MIMIC_Extract.git /root/MIMIC_Extract
cd /root/MIMIC_Extract

# Create conda environment
conda env create --force -f mimic_extract_env_py36.yml
conda activate mimic_data_extraction

# Install additional packages
pip install datapackage

# Run extraction (produces all_hourly_data.h5)
python mimic_direct_extract.py \
    --output_dir /root/mimic_extract_output/ \
    --pop_size 0
```

### 3.4 Verify the HDF5 file

```python
python3 -c "
import pandas as pd
h5 = pd.HDFStore('/root/mimic_extract_output/all_hourly_data.h5', 'r')
print('Keys:', h5.keys())
print('Interventions columns:', h5['/interventions'].columns.tolist())
print('Patients columns:', h5['/patients'].columns.tolist())
print('Num unique patients:', h5['/patients'].index.get_level_values('subject_id').nunique())
h5.close()
"
```

---

## Phase 4: Clone Repo and Set Up Training Environments

### 4.1 Clone the project

```bash
cd /root
git clone <repo-url> target-counterfactual
cd target-counterfactual
git checkout feature/mimic-iii-experiments
```

Or upload from local:
```bash
# From local machine:
scp -P 63498 -r /Users/anisiomlacerda/code/target-counterfactual root@142.112.39.215:/root/target-counterfactual
```

### 4.2 Place the HDF5 file

```bash
cp /root/mimic_extract_output/all_hourly_data.h5 \
   /root/target-counterfactual/lightning-hydra-template-main/src/vendor/VCIP/data/processed/
```

### 4.3 Create conda environments

**VCIP environment** (for VCIP and ACTIN):
```bash
cd /root/target-counterfactual/lightning-hydra-template-main/src/vendor/VCIP
conda create -n vcip python=3.10 -y
conda activate vcip
pip install -r requirements_vcip.txt
```

**Baseline environment** (for CRN, CT, RMSN):
```bash
conda create -n ct python=3.10 -y
conda activate ct
pip install -r requirements_ct.txt
```

**Note:** The requirements pin `torch==2.0.1`. The CUDA-enabled version should install automatically on the Vast.ai instance. If not, install with:
```bash
pip install torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### 4.4 Verify GPU access

```bash
conda activate vcip
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}'); print(f'GPU 0: {torch.cuda.get_device_name(0)}'); print(f'GPU 1: {torch.cuda.get_device_name(1)}')"
```

---

## Phase 5: Smoke Test

```bash
cd /root/target-counterfactual/lightning-hydra-template-main/src/vendor/VCIP
conda activate vcip

# 5 epochs, single seed, GPU 0
CUDA_VISIBLE_DEVICES=0 python runnables/train_vae.py \
    +dataset=mimic3_real +model=vcip +model/hparams/mimic=0* \
    exp.seed=10 exp.epochs=5 model.name=VCIP
```

Expected: training completes without error, outputs saved to `my_outputs/mimic_real/VCIP/train/`.

---

## Phase 6: Full Experiments (2 GPUs in Parallel)

The 2x RTX 3090 allows running two models simultaneously. Strategy: pair models on GPU 0 and GPU 1 to minimize total wall-clock time.

### Execution schedule

**Round 1 — GPU 0: VCIP, GPU 1: CRN** (run in parallel)
```bash
# Terminal 1 (GPU 0)
cd /root/target-counterfactual/lightning-hydra-template-main/src/vendor/VCIP
conda activate vcip
bash scripts/mimic/train/train_vae.sh 0

# Terminal 2 (GPU 1)
cd /root/target-counterfactual/lightning-hydra-template-main/src/vendor/VCIP
conda activate ct
bash scripts/mimic/train/train_crn.sh 0.01 1
```

**Round 2 — GPU 0: ACTIN, GPU 1: CT** (after Round 1 finishes)
```bash
# Terminal 1 (GPU 0)
conda activate vcip
bash scripts/mimic/train/train_actin.sh 0.01 0

# Terminal 2 (GPU 1)
conda activate ct
bash scripts/mimic/train/train_ct.sh 0.01 1
```

**Round 3 — GPU 0: RMSN** (after Round 2 finishes)
```bash
# Terminal 1 (GPU 0)
conda activate ct
bash scripts/mimic/train/train_rmsn.sh 0.01 0
```

### Estimated wall-clock per round

| Round | GPU 0 | GPU 1 | Est. time |
|-------|-------|-------|-----------|
| 1 | VCIP (5 seeds x 100 ep) | CRN (5 seeds x 200 ep) | ~2.5 hrs |
| 2 | ACTIN (5 seeds x 100 ep) | CT (5 seeds x 300 ep) | ~3.5 hrs |
| 3 | RMSN (5 seeds x 200 ep) | — | ~2.5 hrs |
| **Total** | | | **~8.5 hrs** |

### Alternative: automated script

Create a single orchestration script (`run_all_mimic.sh`):

```bash
#!/bin/bash
set -e
VCIP_DIR=/root/target-counterfactual/lightning-hydra-template-main/src/vendor/VCIP
cd $VCIP_DIR

echo "=== Round 1: VCIP (GPU 0) + CRN (GPU 1) ==="
(eval "$(conda shell.bash hook)" && conda activate vcip && bash scripts/mimic/train/train_vae.sh 0) &
(eval "$(conda shell.bash hook)" && conda activate ct && bash scripts/mimic/train/train_crn.sh 0.01 1) &
wait

echo "=== Round 2: ACTIN (GPU 0) + CT (GPU 1) ==="
(eval "$(conda shell.bash hook)" && conda activate vcip && bash scripts/mimic/train/train_actin.sh 0.01 0) &
(eval "$(conda shell.bash hook)" && conda activate ct && bash scripts/mimic/train/train_ct.sh 0.01 1) &
wait

echo "=== Round 3: RMSN (GPU 0) ==="
eval "$(conda shell.bash hook)" && conda activate ct && bash scripts/mimic/train/train_rmsn.sh 0.01 0

echo "=== All experiments complete ==="
```

Run with `nohup bash run_all_mimic.sh > experiments.log 2>&1 &` to survive SSH disconnects.

---

## Phase 7: Collect Results

### 7.1 Download results to local machine

```bash
# From local machine:
scp -P 63498 -r root@142.112.39.215:/root/target-counterfactual/lightning-hydra-template-main/src/vendor/VCIP/my_outputs/mimic_real ./results_mimic_real

scp -P 63498 -r root@142.112.39.215:/root/target-counterfactual/lightning-hydra-template-main/src/vendor/VCIP/csvs ./results_csvs
```

### 7.2 Run analysis notebook locally

Copy results into the local repo and run `results/mimic/analysis.ipynb`.

---

## Checklist

- [ ] Upload MIMIC-III CSVs to instance
- [ ] Install PostgreSQL and load MIMIC-III
- [ ] Build materialized views
- [ ] Run MIMIC-Extract → `all_hourly_data.h5`
- [ ] Verify HDF5 file
- [ ] Clone repo / upload code
- [ ] Create conda environments (vcip, ct)
- [ ] Verify GPU access
- [ ] Smoke test (5 epochs, 1 seed)
- [ ] Round 1: VCIP + CRN
- [ ] Round 2: ACTIN + CT
- [ ] Round 3: RMSN
- [ ] Download results to local machine
- [ ] Run analysis notebook

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| SSH disconnect during long runs | Use `nohup` or `tmux`/`screen` for all training |
| RAM pressure during MIMIC-Extract | Monitor with `htop`; reduce `--pop_size` if OOM |
| PostgreSQL concepts fail | Check for `code_status` relation error; re-run `postgres_make_concepts.sh` |
| Disk space after CSV decompression | Handled: CSVs deleted after PostgreSQL loading (Phase 3.1) |
| Instance expires before completion | Use `nohup` logs to track progress; download partial results |
