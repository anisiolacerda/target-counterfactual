#!/bin/bash
# Phase 1 Orchestration: Run RA evaluation on 2 GPUs in parallel
#
# Strategy: split 4 gammas across 2 GPUs
#   GPU 0: gamma=4 (highest priority) then gamma=2
#   GPU 1: gamma=3 then gamma=1
#
# Estimated time: ~1 hour with 2 GPUs (~30 min per gamma)
#
# Usage: bash scripts/phase1/run_phase1.sh
#   Run with: nohup bash scripts/phase1/run_phase1.sh > phase1.log 2>&1 &

set -e
VCIP_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$VCIP_DIR"

echo "========== Phase 1: RA Evaluation Started at $(date) =========="
echo "Working directory: $(pwd)"
echo ""

# GPU 0: gamma=4 then gamma=2 (gamma=4 first — highest priority for decision gate)
(
    bash scripts/cancer/eval_ra_all_gammas.sh 4 0
    echo "[GPU 0] Starting gamma=2..."
    bash scripts/cancer/eval_ra_all_gammas.sh 2 0
    echo "[GPU 0] Complete."
) > phase1_gpu0.log 2>&1 &
PID_GPU0=$!

# GPU 1: gamma=3 then gamma=1
(
    bash scripts/cancer/eval_ra_all_gammas.sh 3 1
    echo "[GPU 1] Starting gamma=1..."
    bash scripts/cancer/eval_ra_all_gammas.sh 1 1
    echo "[GPU 1] Complete."
) > phase1_gpu1.log 2>&1 &
PID_GPU1=$!

echo "GPU 0 PID: $PID_GPU0 (gamma=4,2)"
echo "GPU 1 PID: $PID_GPU1 (gamma=3,1)"
echo "Logs: phase1_gpu0.log, phase1_gpu1.log"
echo ""
echo "Waiting for both GPUs to finish..."

wait $PID_GPU0
status0=$?
echo "[$(date '+%H:%M:%S')] GPU 0 finished with exit code $status0"

wait $PID_GPU1
status1=$?
echo "[$(date '+%H:%M:%S')] GPU 1 finished with exit code $status1"

echo ""
echo "========== Phase 1 Complete at $(date) =========="

if [ $status0 -eq 0 ] && [ $status1 -eq 0 ]; then
    echo "All evaluations succeeded."
    echo ""
    echo "Results at:"
    for gamma in 1 2 3 4; do
        echo "  gamma=${gamma}: my_outputs/cancer_sim_cont/22/coeff_${gamma}/VCIP/train/True/case_infos/{seed}/False/case_infos_VCIP.pkl"
    done
    echo ""
    echo "Next: download results and run results/phase1/analysis.ipynb"
else
    echo "WARNING: Some evaluations failed. Check phase1_gpu0.log and phase1_gpu1.log"
fi
