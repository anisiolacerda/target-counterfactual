# MIMIC-III RA Evaluation — Vast.ai Execution Plan

## Objective

Re-evaluate existing VCIP models on MIMIC-III to extract predicted diastolic BP trajectories for each perturbation. This enables offline RA-constrained selection analysis.

## Prerequisites

- Vast.ai instance with GPU (1x RTX 3090 sufficient, 2x for parallelism)
- MIMIC-III data: `all_hourly_data.h5` (from MIMIC-Extract)
- Trained VCIP models: 5 seeds × 1 model = 5 checkpoints

## Estimated Time & Cost

| Step | Duration | Cost (at $0.27/hr) |
|------|----------|--------------------|
| Instance setup | 30 min | $0.14 |
| Upload code + models | 10 min | $0.05 |
| Upload MIMIC data | 10-30 min | $0.05-0.14 |
| Evaluation (5 seeds × 4 taus × 100 individuals × 100 perturbations) | ~2-4 hours | $0.54-1.08 |
| Download results | 5 min | $0.02 |
| **Total** | **~3-5 hours** | **~$0.80-1.43** |

## Step-by-Step

### 1. Provision Vast.ai Instance

Requirements:
- 1-2x GPU with ≥12 GB VRAM (RTX 3090, RTX 4090, etc.)
- ≥32 GB RAM
- Python 3.10+

### 2. Upload Code and Data

From local machine:

```bash
# Set SSH params
VAST_HOST="root@<IP>"
VAST_PORT=<PORT>

# Upload VCIP code (includes eval script and model checkpoints)
rsync -avz --progress -e "ssh -p $VAST_PORT" \
    lightning-hydra-template-main/src/vendor/VCIP/ \
    $VAST_HOST:/root/VCIP/

# Upload MIMIC data (if not already on instance)
# Option A: From local (if you have it)
scp -P $VAST_PORT ~/path/to/all_hourly_data.h5 \
    $VAST_HOST:/root/VCIP/src/data/mimic_iii/

# Option B: Re-run MIMIC-Extract (see context/MIMIC_VASTAI_EXECUTION_PLAN.md)
```

### 3. Instance Setup

```bash
ssh -p $VAST_PORT $VAST_HOST

# Create venv
python3 -m venv /root/vcip_env
source /root/vcip_env/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install hydra-core hydra-colorlog omegaconf
pip install scipy numpy pandas matplotlib seaborn
pip install tables  # for HDF5 reading
```

### 4. Verify Setup

```bash
cd /root/VCIP

# Check model checkpoints exist
ls my_outputs/mimic_real/VCIP/train/models/*/model.ckpt

# Check MIMIC data exists
ls src/data/mimic_iii/all_hourly_data.h5

# Quick test (1 seed, should complete in ~10-20 min)
CUDA_VISIBLE_DEVICES=0 python scripts/mimic_ra/eval_mimic_traj.py \
    +dataset=mimic3_real +model=VCIP \
    exp.seed=10 exp.test=False exp.num_samples=10
```

### 5. Run Full Evaluation

```bash
nohup bash scripts/mimic_ra/run_mimic_ra.sh > logs/mimic_ra_main.log 2>&1 &

# Monitor
tail -f logs/mimic_ra/seed_10.log
```

### 6. Download Results

From local machine:

```bash
mkdir -p results_remote/mimic_ra
scp -P $VAST_PORT -r \
    $VAST_HOST:/root/VCIP/my_outputs/mimic_ra \
    results_remote/mimic_ra/my_outputs/
```

### 7. Run Local Analysis

The E7 cells in `analysis.ipynb` expect results at:
```
results_remote/mimic_ra/my_outputs/mimic_ra/VCIP/train/case_infos/{seed}/False/case_infos_VCIP.pkl
```

Re-execute the notebook:
```bash
jupyter nbconvert --to notebook --execute \
    lightning-hydra-template-main/src/reach_avoid/analysis.ipynb \
    --output analysis_executed.ipynb
```

## Output Format

Each pickle contains:
```python
{
    'VCIP': {
        2: [case_info_0, ..., case_info_99],  # tau=2
        4: [...],                               # tau=4
        6: [...],                               # tau=6
        8: [...],                               # tau=8
    }
}
```

Each case_info has:
```python
{
    'individual_id': int,
    'model_losses': np.array(100,),      # ELBO scores
    'true_losses': np.zeros(100,),       # zeros (no ground truth)
    'true_sequence': np.array(1, tau, 2),
    'true_sequence_rank': int,
    'traj_features': {
        'dbp_terminal': np.array(100,),  # predicted DBP at t+tau (mmHg)
        'dbp_min': np.array(100,),       # min DBP along trajectory
        'dbp_max': np.array(100,),       # max DBP along trajectory
        'dbp_trajectory': np.array(100, tau),  # full predicted trajectory
    },
    'treatment_features': {
        'vaso_total': np.array(100,),    # total vasopressor steps
        'vent_total': np.array(100,),    # total ventilation steps
        'sequences': np.array(100, tau, 2),  # full treatment sequences
    },
    'observed_dbp': float,               # actual observed DBP (mmHg)
}
```
