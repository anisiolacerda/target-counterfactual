#!/bin/bash
# Phase 2: RA-aware retraining on Cancer simulator
#
# Trains ReachAvoidVAEModel with:
#   - Weighted intermediate + terminal reconstruction loss (lambda_intermediate=0.5)
#   - VCI-inspired disentanglement regularizer (lambda_disent=0.1)
#   - RA trajectory collection during evaluation
#
# Matrix: 5 seeds x 2 gammas = 10 runs
# Parallelized across 2 GPUs: GPU 0 = gamma=4, GPU 1 = gamma=1
#
# Estimated time: ~5 hours wall time (2 GPUs)
# Estimated cost: ~$5-8 on Vast.ai (2x RTX 3090)
#
# Usage (from VCIP root):
#   nohup bash src/reach_avoid/scripts/run_phase2_train.sh > phase2.log 2>&1 &
#
# Output:
#   my_outputs/cancer_sim_cont/22/coeff_{gamma}/VCIP_RA/train/True/
#     models/{seed}/model.ckpt
#     case_infos/{seed}/False/case_infos_VCIP_RA.pkl

set -e

# Ensure we're at VCIP root
if [ ! -f "runnables/train_vae.py" ]; then
    echo "ERROR: Must run from VCIP root directory (e.g., /root/VCIP)"
    exit 1
fi

# Activate environment
if [ -f "/root/vcip_env/bin/activate" ]; then
    source /root/vcip_env/bin/activate
fi

SEEDS=(10 101 1010 10101 101010)
LOG_DIR="logs/phase2"
mkdir -p "$LOG_DIR"

echo "========== Phase 2: RA-Aware Retraining =========="
echo "Start: $(date)"
echo "Seeds: ${SEEDS[@]}"
echo "Gammas: 4 (GPU 0), 1 (GPU 1)"
echo ""

# Dry-run import test
echo "Verifying ReachAvoidVAEModel import..."
python -c "from src.reach_avoid.model import ReachAvoidVAEModel; print('  Import OK')"
if [ $? -ne 0 ]; then
    echo "ERROR: Cannot import ReachAvoidVAEModel. Check src/reach_avoid/ is present."
    exit 1
fi

run_train() {
    local seed=$1
    local gamma=$2
    local gpu=$3
    local hparam="${gamma}_ra"

    echo "[$(date '+%H:%M:%S')] Starting gamma=${gamma} seed=${seed} on GPU ${gpu}"

    CUDA_VISIBLE_DEVICES=$gpu python runnables/train_vae.py \
        +dataset=cancer_sim_cont \
        +model=vcip \
        "+model/hparams/cancer=${hparam}" \
        exp.seed=${seed} \
        dataset.coeff=${gamma} \
        exp.rank=True \
        exp.test=False \
        > "$LOG_DIR/gamma${gamma}_seed${seed}.log" 2>&1

    local status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Done gamma=${gamma} seed=${seed}"
    else
        echo "[$(date '+%H:%M:%S')] FAILED gamma=${gamma} seed=${seed} (exit $status)"
    fi
    return $status
}

# GPU 0: gamma=4 (5 seeds sequentially)
(
    for seed in "${SEEDS[@]}"; do
        run_train $seed 4 0
    done
) > "$LOG_DIR/gpu0.log" 2>&1 &
PID_GPU0=$!

# GPU 1: gamma=1 (5 seeds sequentially)
(
    for seed in "${SEEDS[@]}"; do
        run_train $seed 1 1
    done
) > "$LOG_DIR/gpu1.log" 2>&1 &
PID_GPU1=$!

echo "GPU 0 PID: $PID_GPU0 (gamma=4)"
echo "GPU 1 PID: $PID_GPU1 (gamma=1)"
echo "Logs: $LOG_DIR/gpu0.log, $LOG_DIR/gpu1.log"
echo ""
echo "Waiting for both GPUs to finish..."

wait $PID_GPU0
status0=$?
echo "[$(date '+%H:%M:%S')] GPU 0 (gamma=4) finished with exit code $status0"

wait $PID_GPU1
status1=$?
echo "[$(date '+%H:%M:%S')] GPU 1 (gamma=1) finished with exit code $status1"

echo ""
echo "========== Phase 2 Complete at $(date) =========="

if [ $status0 -eq 0 ] && [ $status1 -eq 0 ]; then
    echo "All training runs succeeded."
    echo ""
    echo "Results:"
    for gamma in 4 1; do
        for seed in "${SEEDS[@]}"; do
            pkl="my_outputs/cancer_sim_cont/22/coeff_${gamma}/VCIP_RA/train/True/case_infos/${seed}/False/case_infos_VCIP_RA.pkl"
            if [ -f "$pkl" ]; then
                echo "  [OK] gamma=${gamma} seed=${seed}"
            else
                echo "  [MISSING] gamma=${gamma} seed=${seed}: $pkl"
            fi
        done
    done
    echo ""
    echo "Next: download results and compare with vanilla VCIP"
else
    echo "WARNING: Some runs failed. Check logs in $LOG_DIR/"
fi
