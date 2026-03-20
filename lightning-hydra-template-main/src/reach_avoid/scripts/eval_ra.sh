#!/bin/bash
# Phase 1: Re-evaluate existing VCIP models with reach-avoid scoring
# This loads already-trained model checkpoints and runs evaluation with RA config enabled.
# The augmented case_infos will contain: ELBOs, true_losses, true_ra_scores, all_sequences.
#
# Usage: bash scripts/cancer/eval_ra.sh [gpu_id]
#   gpu_id: GPU device ID (default: 0)
#
# Prerequisites: trained models must exist at my_outputs/cancer_sim_cont/22/coeff_4/VCIP/train/True/models/{seed}/model.ckpt
# Cost: ~30 min total on a single GPU (5 seeds × 4 taus × 100 individuals × 100 sequences)

eval "$(conda shell.bash hook)"
conda activate vcip

gpu=${1:-0}
gamma=4
seeds=(10 101 1010 10101 101010)

# RA threshold parameters — Option A (calibrated from simulated volume distribution at t=20-25)
# target_upper=3.0 → ~p75 cancer volume at evaluation time (diam ~1.8cm)
# safety_volume_upper=12.0 → ~p95 cancer volume (diam ~2.8cm)
# safety_chemo_upper=5.0 → ~p95 cumulative chemo dosage
# kappa=10.0 → sigmoid hardness

for seed in "${seeds[@]}"
do
    echo "========== Evaluating seed=${seed}, gamma=${gamma} with RA scoring =========="
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
done

echo "========== Done. Case infos saved with RA scores. =========="
echo "Download case_infos_VCIP.pkl files for analysis."
