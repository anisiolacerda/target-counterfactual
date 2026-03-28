"""
S5.2: Glucose-Insulin Simulator Data Generation

Implements the Bergman Minimal Model for glucose-insulin dynamics.
Generates VCIP-compatible training/validation/test data with confounded treatment assignment.

Bergman Minimal Model:
  dG/dt = -(p1 + X) * G + p1 * Gb + D(t) / Vg
  dX/dt = -p2 * X + p3 * (I - Ib)
  dI/dt = -n * I + u(t) / Vi

Where:
  G = blood glucose (mg/dL), X = remote insulin effect (1/min),
  I = plasma insulin (mU/L), u(t) = exogenous insulin (mU/min),
  D(t) = meal glucose appearance (mg/dL/min)

Target: BG in [70, 180] mg/dL (euglycemic range)
Safety: BG in [50, 250] mg/dL (avoid severe hypo/hyperglycemia)
"""

import numpy as np
import pickle
import os
import argparse
from scipy.special import expit as sigmoid


# ============================================================
# BERGMAN MINIMAL MODEL
# ============================================================

class BergmanPatient:
    """A virtual patient with Bergman minimal model dynamics."""

    # Population parameter distributions (mean, std for lognormal)
    PARAM_DISTS = {
        'p1': (0.028, 0.008),     # glucose effectiveness (1/min)
        'p2': (0.025, 0.008),     # insulin action decay (1/min)
        'p3': (5e-6, 2e-6),       # insulin action gain (mU/L)^-1 min^-2
        'n':  (0.23, 0.07),       # insulin clearance (1/min)
        'Gb': (100.0, 20.0),      # basal glucose (mg/dL)
        'Ib': (10.0, 3.0),        # basal insulin (mU/L)
        'Vg': (1.49, 0.3),        # glucose volume (dL/kg)
        'Vi': (0.04, 0.01),       # insulin volume (L/kg)
        'BW': (70.0, 15.0),       # body weight (kg)
        'meal_sensitivity': (1.0, 0.3),  # individual meal absorption rate
    }

    # Patient "types" for static feature (0=insulin-sensitive, 1=normal, 2=insulin-resistant)
    TYPE_PARAMS = {
        0: {'p3_mult': 1.5, 'Gb_shift': -15},   # insulin-sensitive: lower basal, stronger insulin
        1: {'p3_mult': 1.0, 'Gb_shift': 0},      # normal
        2: {'p3_mult': 0.5, 'Gb_shift': 20},     # insulin-resistant: higher basal, weaker insulin
    }

    def __init__(self, patient_id, rng=None):
        if rng is None:
            rng = np.random.RandomState(patient_id)
        self.patient_id = patient_id
        self.rng = rng

        # Sample patient type
        self.patient_type = rng.choice([0, 1, 2], p=[0.25, 0.5, 0.25])
        tp = self.TYPE_PARAMS[self.patient_type]

        # Sample parameters from population distributions
        self.p1 = max(0.005, rng.normal(*self.PARAM_DISTS['p1']))
        self.p2 = max(0.005, rng.normal(*self.PARAM_DISTS['p2']))
        self.p3 = max(1e-7, rng.normal(*self.PARAM_DISTS['p3'])) * tp['p3_mult']
        self.n = max(0.05, rng.normal(*self.PARAM_DISTS['n']))
        self.Gb = max(60, rng.normal(*self.PARAM_DISTS['Gb'])) + tp['Gb_shift']
        self.Ib = max(2, rng.normal(*self.PARAM_DISTS['Ib']))
        self.Vg = max(0.5, rng.normal(*self.PARAM_DISTS['Vg']))
        self.Vi = max(0.01, rng.normal(*self.PARAM_DISTS['Vi']))
        self.BW = max(30, rng.normal(*self.PARAM_DISTS['BW']))
        self.meal_sens = max(0.3, rng.normal(*self.PARAM_DISTS['meal_sensitivity']))

        # Derived
        self.Vg_total = self.Vg * self.BW  # dL
        self.Vi_total = self.Vi * self.BW   # L

    def get_state(self):
        """Return current ODE state [G, X, I]."""
        return np.array([self.G, self.X, self.I])

    def set_state(self, state):
        """Set ODE state from array [G, X, I]."""
        self.G, self.X, self.I = state

    def reset(self):
        """Initialize patient to steady state + noise."""
        self.G = self.Gb + self.rng.normal(0, 10)
        self.X = 0.0
        self.I = self.Ib
        return self.get_state()

    def step(self, insulin_dose, meal_cho=0, dt=1.0, substeps=10):
        """Advance ODE by dt minutes with given insulin and meal.

        Args:
            insulin_dose: exogenous insulin rate (U/hr, will be converted)
            meal_cho: carbohydrate intake (g), converted to glucose appearance
            dt: total time step (minutes)
            substeps: number of Euler substeps

        Returns:
            New state [G, X, I]
        """
        h = dt / substeps

        # Convert insulin: U/hr -> mU/min (1 U = 1000 mU, /60 for min)
        u = insulin_dose * 1000.0 / 60.0  # mU/min

        # Meal glucose appearance rate: CHO (g/min) -> mg/dL/min
        # 1g CHO ≈ 1000mg glucose, ~50% absorbed, distributed in Vg_total (dL)
        D = meal_cho * 1000.0 * 0.5 * self.meal_sens / self.Vg_total if meal_cho > 0 else 0.0

        for _ in range(substeps):
            G, X, I = self.G, self.X, self.I

            dG = -(self.p1 + X) * G + self.p1 * self.Gb + D
            dX = -self.p2 * X + self.p3 * (I - self.Ib)
            dI = -self.n * (I - self.Ib) + u / self.Vi_total  # Vi in L, u in mU/min → mU/L/min

            self.G = max(10, G + dG * h)  # floor at 10 mg/dL
            self.X = max(0, X + dX * h)
            self.I = max(0, I + dI * h)

        return self.get_state()


# ============================================================
# MEAL SCHEDULE
# ============================================================

def generate_meal_schedule(rng, n_hours=48):
    """Generate a realistic meal schedule over n_hours.

    Meals at ~7am, 12pm, 6pm with some randomness.
    Returns: dict mapping hour -> CHO (grams)
    """
    meals = {}
    for day in range(n_hours // 24 + 1):
        # Breakfast: 7 ± 1 hr, 30-60g CHO
        t = day * 24 + int(rng.normal(7, 0.5))
        if 0 <= t < n_hours:
            meals[t] = rng.uniform(30, 60)
        # Lunch: 12 ± 1 hr, 40-80g CHO
        t = day * 24 + int(rng.normal(12, 0.5))
        if 0 <= t < n_hours:
            meals[t] = rng.uniform(40, 80)
        # Dinner: 18 ± 1 hr, 50-90g CHO
        t = day * 24 + int(rng.normal(18, 0.5))
        if 0 <= t < n_hours:
            meals[t] = rng.uniform(50, 90)
        # Snack: 50% chance, 15 ± 1 hr, 10-30g
        if rng.random() > 0.5:
            t = day * 24 + int(rng.normal(15, 1))
            if 0 <= t < n_hours:
                meals[t] = rng.uniform(10, 30)
    return meals


# ============================================================
# CONFOUNDED TREATMENT POLICY
# ============================================================

def confounded_insulin_policy(bg, patient, gamma, rng):
    """Generate confounded insulin dose based on current BG.

    Higher gamma = stronger confounding (high BG → more insulin).
    Returns insulin dose in U/hr (typical range: 0.5 - 3.0 U/hr).

    Base rate depends on patient weight: ~0.5 U/kg/day = 0.02 U/kg/hr
    """
    base_rate = 0.02 * patient.BW / 24.0  # U/hr, ~0.7-1.5 U/hr

    # Confounding: probability of giving extra insulin increases with BG
    # At gamma=0: random; at gamma=4: strongly driven by BG
    bg_signal = (bg - 120) / 40.0  # normalized: 0 at 120, 1 at 160
    extra_prob = sigmoid(gamma * bg_signal)

    # Extra insulin: 0 to 2x base rate
    extra = extra_prob * base_rate * 2.0

    # Add noise
    noise = rng.normal(0, base_rate * 0.2)
    dose = max(0, base_rate + extra + noise)

    return dose


# ============================================================
# DATA GENERATION
# ============================================================

def simulate_factual(num_patients, max_seq_length, gamma, seed=42):
    """Generate factual observational data with confounded treatment.

    Returns dict with:
        glucose: (N, T) blood glucose values
        insulin: (N, T) insulin doses
        meals: (N, T) meal CHO
        patient_types: (N,) patient type labels
        sequence_lengths: (N,) actual lengths
        states: (N, T, 3) ODE states at each step (for counterfactual branching)
        patient_params: list of BergmanPatient objects
    """
    rng = np.random.RandomState(seed)

    glucose = np.zeros((num_patients, max_seq_length))
    insulin = np.zeros((num_patients, max_seq_length))
    meals_arr = np.zeros((num_patients, max_seq_length))
    patient_types = np.zeros(num_patients, dtype=int)
    sequence_lengths = np.full(num_patients, max_seq_length, dtype=int)
    states = np.zeros((num_patients, max_seq_length, 3))
    patients = []

    for i in range(num_patients):
        patient = BergmanPatient(seed * 10000 + i, rng=np.random.RandomState(seed * 10000 + i))
        patients.append(patient)
        patient_types[i] = patient.patient_type

        # Generate meal schedule
        meal_schedule = generate_meal_schedule(
            np.random.RandomState(seed * 20000 + i), max_seq_length
        )

        # Initialize
        state = patient.reset()

        for t in range(max_seq_length):
            # Record state
            states[i, t] = patient.get_state()
            glucose[i, t] = patient.G

            # Get meal for this hour
            meal_cho = meal_schedule.get(t, 0)
            meals_arr[i, t] = meal_cho

            # Confounded treatment decision
            dose = confounded_insulin_policy(patient.G, patient, gamma, rng)
            insulin[i, t] = dose

            # Simulate 1 hour (60 minutes) with this dose and meal
            # Meal absorbed over first 30 min of the hour
            for minute in range(60):
                meal_this_min = meal_cho if minute < 30 else 0
                patient.step(dose, meal_cho=meal_this_min / 30.0, dt=1.0, substeps=5)

    return {
        'glucose': glucose,
        'insulin': insulin,
        'meals': meals_arr,
        'patient_types': patient_types,
        'sequence_lengths': sequence_lengths,
        'states': states,
        'patients': patients,
    }


def simulate_counterfactual(patient_params, states, tau, intervention_insulin,
                            meal_schedule_seed, t_branch, max_seq_length):
    """Simulate counterfactual trajectory from a branch point.

    Args:
        patient_params: BergmanPatient object
        states: (T, 3) ODE states
        tau: number of future steps
        intervention_insulin: (tau,) insulin doses for intervention
        meal_schedule_seed: seed for meal schedule
        t_branch: branch time step
        max_seq_length: for meal schedule generation

    Returns:
        glucose_traj: (tau,) glucose values under intervention
    """
    patient = BergmanPatient.__new__(BergmanPatient)
    patient.__dict__.update(patient_params.__dict__)
    patient.set_state(states[t_branch].copy())

    meal_schedule = generate_meal_schedule(
        np.random.RandomState(meal_schedule_seed), max_seq_length
    )

    glucose_traj = np.zeros(tau)
    for s in range(tau):
        t = t_branch + s + 1
        dose = intervention_insulin[s]
        meal_cho = meal_schedule.get(t, 0)

        for minute in range(60):
            meal_this_min = meal_cho if minute < 30 else 0
            patient.step(dose, meal_cho=meal_this_min / 30.0, dt=1.0, substeps=5)

        glucose_traj[s] = patient.G

    return glucose_traj


def generate_counterfactual_data(factual_data, num_candidates=100, tau=6, seed=42):
    """Generate counterfactual test data for RA evaluation.

    For each test patient, at a branch point, generate k=num_candidates
    random insulin sequences and simulate their true outcomes.

    Returns dict with:
        all_sequences: (N, k, tau) insulin sequences
        true_glucose_trajectories: (N, k, tau) true glucose outcomes
        branch_points: (N,) branch time indices
    """
    rng = np.random.RandomState(seed)
    N = len(factual_data['patients'])

    all_sequences = np.zeros((N, num_candidates, tau))
    true_trajectories = np.zeros((N, num_candidates, tau))
    branch_points = np.zeros(N, dtype=int)

    for i in range(N):
        patient = factual_data['patients'][i]
        states = factual_data['states'][i]
        seq_len = factual_data['sequence_lengths'][i]

        # Branch point: ~60% through the sequence
        t_branch = min(int(seq_len * 0.6), seq_len - tau - 1)
        t_branch = max(5, t_branch)
        branch_points[i] = t_branch

        # Base insulin rate for this patient
        base_rate = 0.02 * patient.BW / 24.0

        for j in range(num_candidates):
            # Random insulin sequence: diverse range including extreme doses
            # Mix: 50% uniform around base, 25% very low, 25% very high
            r = rng.random()
            if r < 0.5:
                seq = rng.uniform(0, base_rate * 3, size=tau)
            elif r < 0.75:
                seq = rng.uniform(0, base_rate * 0.5, size=tau)  # under-dosing
            else:
                seq = rng.uniform(base_rate * 2, base_rate * 6, size=tau)  # over-dosing
            all_sequences[i, j] = seq

            # Simulate counterfactual
            true_traj = simulate_counterfactual(
                patient, states, tau, seq,
                meal_schedule_seed=42 * 20000 + i,  # same meals as factual
                t_branch=t_branch,
                max_seq_length=factual_data['sequence_lengths'][i]
            )
            true_trajectories[i, j] = true_traj

        if (i + 1) % 20 == 0:
            print("  CF generation: %d/%d patients" % (i + 1, N))

    return {
        'all_sequences': all_sequences,
        'true_glucose_trajectories': true_trajectories,
        'branch_points': branch_points,
    }


# ============================================================
# VCIP FORMAT CONVERSION
# ============================================================

def to_vcip_format(factual_data, split='train'):
    """Convert factual data to VCIP-compatible dictionary.

    VCIP expects:
        current_covariates: [N, T-1, cov_dim]  (glucose + patient_type)
        prev_treatments: [N, T, treat_dim]      (zero-padded insulin)
        current_treatments: [N, T-1, treat_dim]
        outputs: [N, T-1, 1]                    (next glucose, normalized)
        prev_outputs: [N, T-1, 1]
        static_features: [N, 1]
        active_entries: [N, T-1, 1]
        sequence_lengths: [N]
        unscaled_outputs: [N, T-1, 1]
    """
    glucose = factual_data['glucose']  # (N, T)
    insulin = factual_data['insulin']  # (N, T)
    patient_types = factual_data['patient_types']  # (N,)
    seq_lengths = factual_data['sequence_lengths']  # (N,)
    N, T = glucose.shape

    # Compute scaling parameters from training data
    active_glucose = glucose[glucose > 0]  # exclude padding
    g_mean = np.mean(active_glucose)
    g_std = np.std(active_glucose)
    active_insulin = insulin[insulin >= 0]
    i_mean = np.mean(active_insulin)
    i_std = np.std(active_insulin) if np.std(active_insulin) > 0.001 else 1.0
    pt_mean = np.mean(patient_types)
    pt_std = np.std(patient_types) if np.std(patient_types) > 0.001 else 1.0

    scaling = {
        'glucose_mean': g_mean, 'glucose_std': g_std,
        'insulin_mean': i_mean, 'insulin_std': i_std,
        'patient_type_mean': pt_mean, 'patient_type_std': pt_std,
    }

    # Normalize
    glucose_norm = (glucose - g_mean) / g_std
    insulin_norm = (insulin - i_mean) / i_std
    pt_norm = (patient_types - pt_mean) / pt_std

    # Build VCIP arrays
    # Covariates: [glucose_t, patient_type] at each step (exclude last step)
    pt_expanded = np.broadcast_to(pt_norm[:, None], (N, T - 1))
    current_covariates = np.stack([glucose_norm[:, :-1], pt_expanded], axis=-1)  # (N, T-1, 2)

    # Treatments: insulin dose
    prev_treatments = np.zeros((N, T, 1))
    prev_treatments[:, 1:, 0] = insulin_norm[:, :-1]  # zero at t=0
    current_treatments = insulin_norm[:, :-1, None]  # (N, T-1, 1)

    # Outputs: glucose at next step
    outputs = glucose_norm[:, 1:, None]  # (N, T-1, 1)
    unscaled_outputs = glucose[:, 1:, None]  # (N, T-1, 1)
    prev_outputs = glucose_norm[:, :-1, None]  # (N, T-1, 1)

    # Static features
    static_features = pt_norm[:, None]  # (N, 1)

    # Active entries
    active_entries = np.ones((N, T - 1, 1))
    for i in range(N):
        if seq_lengths[i] < T:
            active_entries[i, seq_lengths[i] - 1:] = 0

    data = {
        'current_covariates': current_covariates.astype(np.float32),
        'prev_treatments': prev_treatments.astype(np.float32),
        'current_treatments': current_treatments.astype(np.float32),
        'outputs': outputs.astype(np.float32),
        'unscaled_outputs': unscaled_outputs.astype(np.float32),
        'prev_outputs': prev_outputs.astype(np.float32),
        'static_features': static_features.astype(np.float32),
        'active_entries': active_entries.astype(np.float32),
        'sequence_lengths': (seq_lengths - 1).astype(np.int64),
    }

    return data, scaling


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=int, default=4, help='Confounding strength')
    parser.add_argument('--max_seq_length', type=int, default=48, help='Sequence length (hours)')
    parser.add_argument('--num_train', type=int, default=1000)
    parser.add_argument('--num_val', type=int, default=100)
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--num_candidates', type=int, default=100, help='CF candidates per patient')
    parser.add_argument('--tau', type=int, default=6, help='Prediction horizon')
    parser.add_argument('--output_dir', type=str, default='glucose_data')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Glucose-Insulin Data Generation")
    print("gamma=%d, T=%d, tau=%d, k=%d" % (args.gamma, args.max_seq_length, args.tau, args.num_candidates))
    print("=" * 60)

    # Generate factual data
    for split, n, seed_offset in [('train', args.num_train, 0),
                                   ('val', args.num_val, 100000),
                                   ('test', args.num_test, 200000)]:
        print("\nGenerating %s data (%d patients)..." % (split, n))
        factual = simulate_factual(
            n, args.max_seq_length, args.gamma,
            seed=args.seed + seed_offset
        )

        # Summary statistics
        g = factual['glucose']
        ins = factual['insulin']
        print("  Glucose: mean=%.1f, std=%.1f, range=[%.1f, %.1f]" %
              (g.mean(), g.std(), g.min(), g.max()))
        print("  Insulin: mean=%.3f, std=%.3f" % (ins.mean(), ins.std()))
        print("  Patient types: %s" % np.bincount(factual['patient_types']))

        # Convert to VCIP format
        vcip_data, scaling = to_vcip_format(factual, split)
        print("  VCIP shapes: covariates=%s, treatments=%s, outputs=%s" % (
            vcip_data['current_covariates'].shape,
            vcip_data['current_treatments'].shape,
            vcip_data['outputs'].shape
        ))

        # Save factual + VCIP format
        save_path = os.path.join(args.output_dir, 'gamma_%d' % args.gamma)
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, '%s_factual.pkl' % split), 'wb') as f:
            pickle.dump(factual, f)
        with open(os.path.join(save_path, '%s_vcip.pkl' % split), 'wb') as f:
            pickle.dump(vcip_data, f)
        if split == 'train':
            with open(os.path.join(save_path, 'scaling.pkl'), 'wb') as f:
                pickle.dump(scaling, f)

    # Generate counterfactual test data
    print("\nGenerating counterfactual test data (k=%d, tau=%d)..." % (args.num_candidates, args.tau))
    test_factual_path = os.path.join(args.output_dir, 'gamma_%d' % args.gamma, 'test_factual.pkl')
    with open(test_factual_path, 'rb') as f:
        test_factual = pickle.load(f)

    cf_data = generate_counterfactual_data(
        test_factual, num_candidates=args.num_candidates,
        tau=args.tau, seed=args.seed + 300000
    )

    # CF summary
    true_traj = cf_data['true_glucose_trajectories']
    print("  CF glucose range: [%.1f, %.1f]" % (true_traj.min(), true_traj.max()))
    print("  CF terminal glucose: mean=%.1f, std=%.1f" % (
        true_traj[:, :, -1].mean(), true_traj[:, :, -1].std()))

    # Safety/target analysis
    target_lo, target_hi = 70, 180
    safety_lo, safety_hi = 50, 250
    terminal = true_traj[:, :, -1]
    in_target = ((terminal >= target_lo) & (terminal <= target_hi)).mean()
    max_bg = true_traj.max(axis=-1)
    min_bg = true_traj.min(axis=-1)
    safe = ((max_bg <= safety_hi) & (min_bg >= safety_lo)).mean()
    feasible = in_target * safe  # approximate
    print("  In-target: %.1f%%, Safe: %.1f%%" % (100 * in_target, 100 * safe))

    cf_save_path = os.path.join(args.output_dir, 'gamma_%d' % args.gamma, 'test_cf_tau%d.pkl' % args.tau)
    with open(cf_save_path, 'wb') as f:
        pickle.dump(cf_data, f)

    print("\nData generation complete. Files saved in %s/" % args.output_dir)


if __name__ == '__main__':
    main()
