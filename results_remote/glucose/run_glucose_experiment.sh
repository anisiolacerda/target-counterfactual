#!/bin/bash
# S5.2: Complete Glucose-Insulin Experiment Pipeline
# Run on Vast.ai instance with 2x RTX 3090
set -e

VCIP_DIR=/root/VCIP
GLUCOSE_DIR=/root/glucose_experiment
DATA_DIR=${GLUCOSE_DIR}/glucose_data
GAMMA=4

echo "============================================="
echo "S5.2: Glucose-Insulin Experiment"
echo "gamma=${GAMMA}"
echo "============================================="

# Step 0: Setup
echo ""
echo "[Step 0] Setting up environment..."
source /root/vcip_env/bin/activate
mkdir -p ${GLUCOSE_DIR}

# Copy data generation script
cp /root/glucose_files/generate_glucose_data.py ${GLUCOSE_DIR}/
cp /root/glucose_files/glucose_dataset.py ${VCIP_DIR}/src/data/glucose.py

# Add glucose import to VCIP's data __init__
if ! grep -q "GlucoseDatasetCollection" ${VCIP_DIR}/src/data/__init__.py; then
    echo "from src.data.glucose import GlucoseDatasetCollection" >> ${VCIP_DIR}/src/data/__init__.py
    echo "  Added glucose import to VCIP"
fi

# Copy config
cp /root/glucose_files/glucose_sim.yaml ${VCIP_DIR}/configs/dataset/glucose_sim.yaml

# Step 1: Generate data
echo ""
echo "[Step 1] Generating glucose data (gamma=${GAMMA})..."
cd ${GLUCOSE_DIR}
python3 generate_glucose_data.py \
    --gamma ${GAMMA} \
    --max_seq_length 48 \
    --num_train 1000 \
    --num_val 100 \
    --num_test 100 \
    --num_candidates 100 \
    --tau 6 \
    --output_dir ${DATA_DIR} \
    --seed 42

echo ""
echo "[Step 1b] Generating additional tau values..."
for TAU in 2 4 8; do
    echo "  Generating CF data for tau=${TAU}..."
    python3 -c "
import pickle, numpy as np, os, sys
sys.path.insert(0, '.')
from generate_glucose_data import generate_counterfactual_data

# Load test factual data
with open('${DATA_DIR}/gamma_${GAMMA}/test_factual.pkl', 'rb') as f:
    test_factual = pickle.load(f)

cf_data = generate_counterfactual_data(test_factual, num_candidates=100, tau=${TAU}, seed=42+300000)
with open('${DATA_DIR}/gamma_${GAMMA}/test_cf_tau${TAU}.pkl', 'wb') as f:
    pickle.dump(cf_data, f)

terminal = cf_data['true_glucose_trajectories'][:,:,-1]
in_target = ((terminal >= 70) & (terminal <= 180)).mean()
print('  tau=${TAU}: CF terminal BG mean=%.1f, in-target=%.1f%%' % (terminal.mean(), 100*in_target))
"
done

# Step 2: Train VCIP
echo ""
echo "[Step 2] Training VCIP on glucose data..."
cd ${VCIP_DIR}

# Create symlink for data access
ln -sf ${DATA_DIR} ${VCIP_DIR}/glucose_data

for SEED in 10 101 1010 10101 101010; do
    echo ""
    echo "  Training seed=${SEED}..."

    # Training (encoder + decoder)
    python3 runnables/train_enc_dec.py \
        +dataset=glucose_sim \
        +model=VCIP \
        exp.seed=${SEED} \
        exp.global_seed=22 \
        dataset.gamma=${GAMMA} \
        exp.max_epochs=100 \
        exp.mode=train \
        exp.sam=True \
        2>&1 | tail -5

    echo "  Seed ${SEED} training complete."
done

echo ""
echo "[Step 2] Training complete for all seeds."

# Step 3: Evaluate with RA filtering
echo ""
echo "[Step 3] RA evaluation..."
# This will be handled by a separate Python script
python3 /root/glucose_files/evaluate_glucose_ra.py \
    --data_dir ${DATA_DIR} \
    --model_dir ${VCIP_DIR}/my_outputs/glucose_sim/22/coeff_${GAMMA}/VCIP/train/True \
    --gamma ${GAMMA} \
    --output_dir ${GLUCOSE_DIR}/results

echo ""
echo "============================================="
echo "Experiment complete!"
echo "============================================="
