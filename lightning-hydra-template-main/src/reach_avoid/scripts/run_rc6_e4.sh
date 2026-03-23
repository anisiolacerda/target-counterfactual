#!/bin/bash
# RC6 + E4: Intermediate Prediction Quality & VCI Diagnostic
# Runs gamma=4, all 5 seeds, parallel across 2 GPUs
#
# Usage (from VCIP root):
#   nohup bash src/reach_avoid/scripts/run_rc6_e4.sh > rc6_e4.log 2>&1 &

set -e

if [ ! -f "runnables/train_vae.py" ]; then
    echo "ERROR: Must run from VCIP root directory"
    exit 1
fi

if [ -f "/root/vcip_env/bin/activate" ]; then
    source /root/vcip_env/bin/activate
fi

GAMMA=4
mkdir -p logs

echo "========== RC6 + E4 =========="
echo "Start: $(date)"

run_eval() {
    local seed=$1
    local gpu=$2
    echo "[$(date '+%H:%M:%S')] GPU $gpu: seed=$seed"
    CUDA_VISIBLE_DEVICES=$gpu python src/reach_avoid/eval_intermediate_and_diagnostic.py \
        +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${GAMMA}*" \
        exp.seed=$seed dataset.coeff=$GAMMA exp.rank=True exp.test=False
    echo "[$(date '+%H:%M:%S')] GPU $gpu: done seed=$seed"
}

# GPU 0: seeds 10, 1010, 101010
(
    for seed in 10 1010 101010; do
        run_eval $seed 0
    done
) > logs/rc6_e4_gpu0.log 2>&1 &
PID0=$!

# GPU 1: seeds 101, 10101
(
    for seed in 101 10101; do
        run_eval $seed 1
    done
) > logs/rc6_e4_gpu1.log 2>&1 &
PID1=$!

echo "GPU 0 PID: $PID0 (seeds 10, 1010, 101010)"
echo "GPU 1 PID: $PID1 (seeds 101, 10101)"
echo "Waiting..."

wait $PID0; s0=$?
echo "[$(date '+%H:%M:%S')] GPU 0 done (exit $s0)"
wait $PID1; s1=$?
echo "[$(date '+%H:%M:%S')] GPU 1 done (exit $s1)"

echo ""
echo "========== Complete at $(date) =========="
ls -la my_outputs/rc6_e4/ 2>/dev/null
