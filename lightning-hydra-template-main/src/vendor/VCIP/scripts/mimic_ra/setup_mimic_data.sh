#!/bin/bash
# MIMIC-III data setup on Vast.ai for RA evaluation
# This script sets up PostgreSQL, loads MIMIC-III data, and runs MIMIC-Extract
# to produce all_hourly_data.h5
#
# Prerequisites: MIMIC-III CSVs uploaded to /root/mimic-iii-data/
#
# Usage: bash scripts/mimic_ra/setup_mimic_data.sh 2>&1 | tee /root/mimic_setup.log

set -e

echo "=== MIMIC-III Data Setup ==="
echo "Start: $(date)"

# Phase 1: Install system dependencies
echo ""
echo "=== Phase 1: System dependencies ==="
apt-get update -qq
apt-get install -y -qq postgresql postgresql-contrib libhdf5-dev git wget

# Phase 2: Install Miniconda
echo ""
echo "=== Phase 2: Miniconda ==="
if [ ! -d "/root/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /root/miniconda3
    rm /tmp/miniconda.sh
fi
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc

# Phase 3: Set up PostgreSQL
echo ""
echo "=== Phase 3: PostgreSQL ==="
service postgresql start
sudo -u postgres createuser -s root 2>/dev/null || true
createdb mimic 2>/dev/null || true

# Phase 4: Load MIMIC-III CSVs
echo ""
echo "=== Phase 4: Load MIMIC-III data ==="
if [ ! -d "/root/mimic-code" ]; then
    git clone https://github.com/MIT-LCP/mimic-code.git /root/mimic-code
fi

cd /root/mimic-code/mimic-iii/buildmimic/postgres

# Check if data is already loaded
PATIENT_COUNT=$(psql -d mimic -t -c "SELECT count(*) FROM mimiciii.patients;" 2>/dev/null | tr -d ' ' || echo "0")
if [ "$PATIENT_COUNT" -gt "0" ]; then
    echo "MIMIC-III already loaded ($PATIENT_COUNT patients). Skipping."
else
    echo "Loading MIMIC-III CSVs (this takes 30-60 minutes)..."
    make mimic-gz datadir=/root/mimic-iii-data/
    PATIENT_COUNT=$(psql -d mimic -t -c "SELECT count(*) FROM mimiciii.patients;" | tr -d ' ')
    echo "Loaded $PATIENT_COUNT patients"
fi

# Phase 5: Build concepts (materialized views)
echo ""
echo "=== Phase 5: Build concepts ==="
CONCEPT_COUNT=$(psql -d mimic -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE '%durations%';" 2>/dev/null | tr -d ' ' || echo "0")
if [ "$CONCEPT_COUNT" -gt "0" ]; then
    echo "Concepts likely already built. Skipping."
else
    cd /root/mimic-code/mimic-iii/concepts
    psql -d mimic -f postgres-functions.sql
    bash postgres_make_concepts.sh
    # Extended concepts
    if [ -d "utils" ]; then
        cd utils
        bash postgres_make_extended_concepts.sh 2>/dev/null || true
        psql -d mimic -f niv-durations.sql 2>/dev/null || true
    fi
fi

# Phase 6: Run MIMIC-Extract
echo ""
echo "=== Phase 6: MIMIC-Extract ==="
if [ -f "/root/mimic_extract_output/all_hourly_data.h5" ]; then
    echo "all_hourly_data.h5 already exists. Skipping."
else
    if [ ! -d "/root/MIMIC_Extract" ]; then
        git clone https://github.com/MLforHealth/MIMIC_Extract.git /root/MIMIC_Extract
    fi
    cd /root/MIMIC_Extract

    # Create conda env
    conda env create --force -f mimic_extract_env_py36.yml 2>/dev/null || true
    conda activate mimic_data_extraction
    pip install datapackage 2>/dev/null || true

    echo "Running MIMIC-Extract (this takes 2-5 hours)..."
    python mimic_direct_extract.py --output_dir /root/mimic_extract_output/ --pop_size 0

    echo "Extraction complete."
fi

# Phase 7: Copy HDF5 to VCIP data directory
echo ""
echo "=== Phase 7: Link data ==="
if [ -f "/root/mimic_extract_output/all_hourly_data.h5" ]; then
    cp /root/mimic_extract_output/all_hourly_data.h5 /root/VCIP/src/data/mimic_iii/all_hourly_data.h5
    echo "Copied all_hourly_data.h5 to VCIP data directory"
else
    echo "ERROR: all_hourly_data.h5 not found!"
    exit 1
fi

# Phase 8: Verify
echo ""
echo "=== Phase 8: Verification ==="
source /root/vcip_env/bin/activate
python3 -c "
import pandas as pd
h5 = pd.HDFStore('/root/VCIP/src/data/mimic_iii/all_hourly_data.h5', 'r')
print('Keys:', h5.keys())
n_patients = h5['/patients'].index.get_level_values('subject_id').nunique()
print(f'Patients: {n_patients}')
interventions = h5['/interventions']
print(f'Interventions: {interventions.columns.tolist()}')
print(f'Interventions shape: {interventions.shape}')
h5.close()
print('Verification PASSED')
"

echo ""
echo "=== Setup complete: $(date) ==="
echo "Next: bash scripts/mimic_ra/run_mimic_ra.sh"
