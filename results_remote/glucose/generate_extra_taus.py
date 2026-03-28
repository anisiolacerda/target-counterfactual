"""Generate counterfactual data for additional tau values.

Must be run from the same directory as generate_glucose_data.py.
Uses a pickle hack to handle BergmanPatient class resolution.
"""
import sys
sys.path.insert(0, '.')
import generate_glucose_data
from generate_glucose_data import generate_counterfactual_data, BergmanPatient
import pickle
import numpy as np

# Pickle expects BergmanPatient in __main__ module
sys.modules['__main__'].BergmanPatient = BergmanPatient

DATA_DIR = 'glucose_data/gamma_4'

# Load test factual data
with open('%s/test_factual.pkl' % DATA_DIR, 'rb') as f:
    test_factual = pickle.load(f)

for tau in [2, 4, 8]:
    print('Generating CF for tau=%d...' % tau)
    cf = generate_counterfactual_data(test_factual, num_candidates=100, tau=tau, seed=42 + 300000)

    save_path = '%s/test_cf_tau%d.pkl' % (DATA_DIR, tau)
    with open(save_path, 'wb') as f:
        pickle.dump(cf, f)

    t = cf['true_glucose_trajectories'][:, :, -1]
    mx = cf['true_glucose_trajectories'].max(axis=-1)
    mn = cf['true_glucose_trajectories'].min(axis=-1)
    intgt = ((t >= 70) & (t <= 180)).mean()
    safe = ((mx <= 250) & (mn >= 50)).mean()
    print('  tau=%d: terminal mean=%.1f, in-target=%.1f%%, safe=%.1f%%' % (tau, t.mean(), 100 * intgt, 100 * safe))

print('Done.')
