#!/bin/bash
# Run all 4 decoder variants on Cancer gamma=4, all 5 seeds
# Each variant takes ~2-3 hours (5 seeds × 4 taus × 100 patients × 100 candidates)
# Total: ~8-12 hours sequential

set -e
source /root/vcip_env/bin/activate
cd /root/VCIP

LOG_DIR=/root/results_decoder_mods/logs
mkdir -p $LOG_DIR

echo "$(date): Starting decoder modification experiments"

# 1. Vanilla baseline (no fine-tuning, just model-predicted evaluation)
echo "$(date): Starting vanilla..."
python3 /root/train_and_eval_decoder_mods.py \
    --variant vanilla --gamma 4 --all_seeds \
    > $LOG_DIR/vanilla.log 2>&1
echo "$(date): Vanilla DONE"

# 2. Heteroscedastic (NLL loss with learned logvar)
echo "$(date): Starting heteroscedastic..."
python3 /root/train_and_eval_decoder_mods.py \
    --variant heteroscedastic --gamma 4 --all_seeds --fine_tune_epochs 50 \
    > $LOG_DIR/heteroscedastic.log 2>&1
echo "$(date): Heteroscedastic DONE"

# 3. Wider decoder ([64,32] hidden dims)
echo "$(date): Starting wider..."
python3 /root/train_and_eval_decoder_mods.py \
    --variant wider --gamma 4 --all_seeds --fine_tune_epochs 50 \
    > $LOG_DIR/wider.log 2>&1
echo "$(date): Wider DONE"

# 4. MC-Dropout (dropout in decoder, 20 MC passes at test)
echo "$(date): Starting mcdropout..."
python3 /root/train_and_eval_decoder_mods.py \
    --variant mcdropout --gamma 4 --all_seeds --fine_tune_epochs 50 \
    > $LOG_DIR/mcdropout.log 2>&1
echo "$(date): MC-Dropout DONE"

echo "$(date): ALL VARIANTS COMPLETE"
echo "Results in /root/results_decoder_mods/"
ls -la /root/results_decoder_mods/*.pkl
