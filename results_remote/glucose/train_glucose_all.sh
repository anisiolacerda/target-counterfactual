#!/bin/bash
set -e
cd /root/VCIP
source /root/vcip_env/bin/activate

for SEED in 10 101 1010 10101 101010; do
    echo ""
    echo "========================================="
    echo "Training VCIP on glucose, seed=${SEED}"
    echo "========================================="

    python3 runnables/train_vae.py \
        +dataset=glucose_sim \
        +model=vcip \
        exp.seed=${SEED} \
        exp.global_seed=22 \
        dataset.gamma=4 \
        exp.max_epochs=100 \
        exp.mode=train \
        exp.sam=True \
        2>&1 | tail -10

    echo "Seed ${SEED} complete."
done

echo ""
echo "All seeds complete!"
echo "Checking outputs:"
find my_outputs/glucose_sim -name 'model.ckpt' -o -name '*.ckpt' 2>/dev/null
