#!/bin/bash
# Diagnostic: run 1 seed, 1 gamma with very wide thresholds to see actual data ranges
# The trajectory stats will be saved in the pkl and printed to stdout

if [ -d "/root/vcip_env" ]; then
    source /root/vcip_env/bin/activate
else
    eval "$(conda shell.bash hook)"
    conda activate vcip
fi

gpu=${1:-0}

echo "========== RA Diagnostic: gamma=4, seed=10 =========="
CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_vae.py \
    +dataset=cancer_sim_cont \
    +model=vcip \
    +model/hparams/cancer=4* \
    exp.seed=10 \
    exp.epochs=100 \
    dataset.coeff=4 \
    model.name=VCIP \
    exp.test=False \
    exp.rank=True \
    +exp.reach_avoid.target_upper=1200.0 \
    +exp.reach_avoid.safety_volume_upper=1200.0 \
    +exp.reach_avoid.safety_chemo_upper=100.0 \
    +exp.reach_avoid.kappa=10.0

echo "========== Done =========="
