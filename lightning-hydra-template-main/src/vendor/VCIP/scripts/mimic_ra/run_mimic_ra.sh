#!/bin/bash
# MIMIC-III RA evaluation: extract predicted DBP trajectories
# Run on Vast.ai with 2x GPUs
#
# Prerequisites:
# 1. VCIP code at /root/VCIP/
# 2. MIMIC data (all_hourly_data.h5) at correct path
# 3. Trained VCIP models at my_outputs/mimic_real/VCIP/train/models/{seed}/model.ckpt
# 4. Python env at /root/vcip_env/
#
# Usage: bash scripts/mimic_ra/run_mimic_ra.sh

set -e

cd /root/VCIP
source /root/vcip_env/bin/activate

SEEDS=(10 101 1010 10101 101010)
LOG_DIR="logs/mimic_ra"
mkdir -p "$LOG_DIR"

echo "=== MIMIC-III RA Trajectory Evaluation ==="
echo "Seeds: ${SEEDS[@]}"
echo "Start: $(date)"

# Run seeds in parallel across 2 GPUs
# GPU 0: seeds 10, 1010, 101010
# GPU 1: seeds 101, 10101
run_seed() {
    local seed=$1
    local gpu=$2
    echo "[$(date +%H:%M:%S)] Starting seed=$seed on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/mimic_ra/eval_mimic_traj.py \
        +dataset=mimic3_real +model=vcip \
        exp.seed=$seed exp.test=False exp.num_samples=10 \
        > "$LOG_DIR/seed_${seed}.log" 2>&1
    echo "[$(date +%H:%M:%S)] Done seed=$seed"
}

# Phase 1: 2 seeds in parallel
run_seed 10 0 &
run_seed 101 1 &
wait

# Phase 2: 2 seeds in parallel
run_seed 1010 0 &
run_seed 10101 1 &
wait

# Phase 3: last seed
run_seed 101010 0

echo ""
echo "=== Complete: $(date) ==="
echo "Results at: my_outputs/mimic_ra/VCIP/train/case_infos/"
ls -la my_outputs/mimic_ra/VCIP/train/case_infos/*/False/case_infos_VCIP.pkl 2>/dev/null

# Verify outputs
echo ""
echo "=== Verification ==="
python3 -c "
import pickle, os
seeds = [10, 101, 1010, 10101, 101010]
for seed in seeds:
    path = f'my_outputs/mimic_ra/VCIP/train/case_infos/{seed}/False/case_infos_VCIP.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        taus = list(data['VCIP'].keys())
        n = len(data['VCIP'][taus[0]])
        has_traj = 'traj_features' in data['VCIP'][taus[0]][0]
        print(f'  seed={seed}: taus={taus}, n_individuals={n}, has_traj={has_traj}')
    else:
        print(f'  seed={seed}: MISSING')
"
