#!/bin/bash
# Phase 1: Re-evaluate existing VCIP models with reach-avoid scoring across ALL gammas
# This loads already-trained model checkpoints and runs evaluation with RA config enabled.
# The augmented case_infos will contain: ELBOs, true_losses, true_ra_scores, all_sequences.
#
# Usage: bash scripts/cancer/eval_ra_all_gammas.sh [gamma] [gpu_id]
#   gamma: confounding strength (1,2,3,4 or "all" for all gammas). Default: all
#   gpu_id: GPU device ID (default: 0)
#
# Prerequisites: trained models must exist at
#   my_outputs/cancer_sim_cont/22/coeff_{gamma}/VCIP/train/True/models/{seed}/model.ckpt
#
# Cost estimate:
#   Single gamma: ~30 min (5 seeds × 4 taus × 100 individuals × 100 sequences)
#   All gammas:   ~2 hrs on a single GPU, ~1 hr with 2 GPUs (see run_phase1.sh)

# Activate environment (supports both conda and venv)
if [ -d "/root/vcip_env" ]; then
    source /root/vcip_env/bin/activate
else
    eval "$(conda shell.bash hook)"
    conda activate vcip
fi

gamma_arg=${1:-all}
gpu=${2:-0}
seeds=(10 101 1010 10101 101010)

# RA threshold parameters — Option A (calibrated from simulated volume distribution at t=20-25)
# target_upper=3.0 → ~p75 cancer volume at evaluation time (diam ~1.8cm)
# safety_volume_upper=12.0 → ~p95 cancer volume (diam ~2.8cm)
# safety_chemo_upper=5.0 → ~p95 cumulative chemo dosage
# kappa=10.0 → sigmoid hardness

if [ "$gamma_arg" = "all" ]; then
    gammas=(1 2 3 4)
else
    gammas=($gamma_arg)
fi

for gamma in "${gammas[@]}"
do
    for seed in "${seeds[@]}"
    do
        echo "========== [$(date '+%H:%M:%S')] Evaluating gamma=${gamma}, seed=${seed} with RA scoring (GPU ${gpu}) =========="
        CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_vae.py \
            +dataset=cancer_sim_cont \
            +model=vcip \
            +model/hparams/cancer=${gamma}* \
            exp.seed=${seed} \
            exp.epochs=100 \
            dataset.coeff=${gamma} \
            model.name=VCIP \
            exp.test=False \
            exp.rank=True \
            +exp.reach_avoid.target_upper=3.0 \
            +exp.reach_avoid.safety_volume_upper=12.0 \
            +exp.reach_avoid.safety_chemo_upper=5.0 \
            +exp.reach_avoid.kappa=10.0
        echo "========== [$(date '+%H:%M:%S')] Done gamma=${gamma}, seed=${seed} =========="
    done
done

echo "========== All evaluations complete. =========="
echo "Case infos saved with RA scores at my_outputs/cancer_sim_cont/22/coeff_{gamma}/VCIP/train/True/case_infos/{seed}/False/"
