#!/bin/bash
cd /root/VCIP
source /root/vcip_env/bin/activate

echo "Starting full glucose VCIP RA evaluation — all seeds"
echo "Start time: $(date)"

python3 /root/glucose_experiment/eval_glucose_vcip.py --all_seeds 2>&1

echo ""
echo "End time: $(date)"
echo "Done."
