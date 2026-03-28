"""
Glucose-Insulin Dataset for VCIP.

Loads pre-generated Bergman minimal model data in VCIP-compatible format.
Mirrors the SyntheticCancerDatasetCollectionCont interface.
"""

import numpy as np
import pickle
import os
import logging
from copy import deepcopy
from torch.utils.data import Dataset

from src.data.dataset_collection import SyntheticDatasetCollection

logger = logging.getLogger(__name__)


class GlucoseDataset(Dataset):
    """Pytorch Dataset wrapping pre-generated glucose simulation data."""

    def __init__(self, data_dir, gamma, subset_name, treatment_mode='continuous'):
        self.subset_name = subset_name
        self.treatment_mode = treatment_mode

        # Load pre-generated VCIP-format data
        vcip_path = os.path.join(data_dir, 'gamma_%d' % gamma, '%s_vcip.pkl' % subset_name)
        with open(vcip_path, 'rb') as f:
            self.data = pickle.load(f)

        # Load scaling params (computed from training data)
        scaling_path = os.path.join(data_dir, 'gamma_%d' % gamma, 'scaling.pkl')
        with open(scaling_path, 'rb') as f:
            self.scaling = pickle.load(f)

        self.scaling_params = {
            'output_means': np.array([self.scaling['glucose_mean']]),
            'output_stds': np.array([self.scaling['glucose_std']]),
        }

        self.processed = True
        self.processed_sequential = False
        self.processed_autoregressive = False
        self.exploded = False
        self.norm_const = 500.0  # glucose death threshold (mg/dL)

        # For compatibility with Cancer dataset interface
        self.autoregressive = True
        self.has_vitals = False

        n = len(self.data['sequence_lengths'])
        logger.info("Loaded glucose %s data: %d patients" % (subset_name, n))

    def __getitem__(self, index):
        result = {k: v[index] for k, v in self.data.items()
                  if hasattr(v, '__len__') and len(v) == len(self)}
        return result

    def __len__(self):
        return len(self.data['sequence_lengths'])

    def get_scaling_params(self):
        return self.scaling_params

    def process_data(self, scaling_params):
        """Data is already processed during generation."""
        self.scaling_params = scaling_params
        logger.info("Glucose %s data already processed" % self.subset_name)
        return self.data

    def process_sequential(self, encoder_r, projection_horizon, save_encoder_r=False):
        """Pre-process for multi-step prediction (training)."""
        if self.processed_sequential:
            return self.data

        logger.info("Processing glucose %s for sequential training" % self.subset_name)

        outputs = self.data['outputs']
        sequence_lengths = self.data['sequence_lengths']
        active_entries = self.data['active_entries']
        current_treatments = self.data['current_treatments']
        previous_treatments = self.data['prev_treatments'][:, 1:, :]
        current_covariates = self.data['current_covariates']

        num_patients, seq_length, num_features = outputs.shape
        num_seq2seq_rows = num_patients * seq_length

        seq2seq_state_inits = np.zeros((num_seq2seq_rows, encoder_r.shape[-1]))
        seq2seq_active_encoder_r = np.zeros((num_seq2seq_rows, seq_length))
        seq2seq_original_index = np.zeros((num_seq2seq_rows,))
        seq2seq_previous_treatments = np.zeros((num_seq2seq_rows, projection_horizon, previous_treatments.shape[-1]))
        seq2seq_current_treatments = np.zeros((num_seq2seq_rows, projection_horizon, current_treatments.shape[-1]))
        seq2seq_current_covariates = np.zeros((num_seq2seq_rows, projection_horizon, current_covariates.shape[-1]))
        seq2seq_outputs = np.zeros((num_seq2seq_rows, projection_horizon, outputs.shape[-1]))
        seq2seq_active_entries = np.zeros((num_seq2seq_rows, projection_horizon, active_entries.shape[-1]))
        seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)

        total_seq2seq_rows = 0

        for i in range(num_patients):
            sequence_length = int(sequence_lengths[i])
            for t in range(1, sequence_length - projection_horizon):
                seq2seq_state_inits[total_seq2seq_rows] = encoder_r[i, t - 1]
                seq2seq_original_index[total_seq2seq_rows] = i
                seq2seq_active_encoder_r[total_seq2seq_rows, :t] = 1.0

                max_projection = min(projection_horizon, sequence_length - t)
                seq2seq_active_entries[total_seq2seq_rows, :max_projection] = active_entries[i, t:t + max_projection]
                seq2seq_previous_treatments[total_seq2seq_rows, :max_projection] = previous_treatments[i, t - 1:t + max_projection - 1]
                seq2seq_current_treatments[total_seq2seq_rows, :max_projection] = current_treatments[i, t:t + max_projection]
                seq2seq_outputs[total_seq2seq_rows, :max_projection] = outputs[i, t:t + max_projection]
                seq2seq_sequence_lengths[total_seq2seq_rows] = max_projection
                seq2seq_current_covariates[total_seq2seq_rows, :max_projection] = current_covariates[i, t:t + max_projection]

                total_seq2seq_rows += 1

        # Truncate
        seq2seq_data = {
            'init_state': seq2seq_state_inits[:total_seq2seq_rows],
            'original_index': seq2seq_original_index[:total_seq2seq_rows],
            'active_encoder_r': seq2seq_active_encoder_r[:total_seq2seq_rows],
            'prev_treatments': seq2seq_previous_treatments[:total_seq2seq_rows],
            'current_treatments': seq2seq_current_treatments[:total_seq2seq_rows],
            'current_covariates': seq2seq_current_covariates[:total_seq2seq_rows],
            'prev_outputs': seq2seq_current_covariates[:total_seq2seq_rows, :, :1],
            'static_features': seq2seq_current_covariates[:total_seq2seq_rows, 0, 1:],
            'outputs': seq2seq_outputs[:total_seq2seq_rows],
            'sequence_lengths': seq2seq_sequence_lengths[:total_seq2seq_rows],
            'active_entries': seq2seq_active_entries[:total_seq2seq_rows],
            'unscaled_outputs': seq2seq_outputs[:total_seq2seq_rows] * self.scaling_params['output_stds'] + self.scaling_params['output_means'],
        }

        self.data_original = deepcopy(self.data)
        self.data = seq2seq_data
        self.processed_sequential = True
        self.exploded = True

        if save_encoder_r:
            self.encoder_r = encoder_r[:, :seq_length, :]

        return self.data

    def process_sequential_test(self, projection_horizon, encoder_r=None, save_encoder_r=False):
        """Pre-process test dataset for multi-step prediction."""
        if self.processed_sequential:
            return self.data

        logger.info("Processing glucose %s for sequential testing" % self.subset_name)

        sequence_lengths = self.data['sequence_lengths']
        outputs = self.data['outputs']
        current_treatments = self.data['current_treatments']
        previous_treatments = self.data['prev_treatments'][:, 1:, :]
        current_covariates = self.data['current_covariates']

        num_patient_points, max_seq_length, num_features = outputs.shape

        if encoder_r is not None:
            seq2seq_state_inits = np.zeros((num_patient_points, encoder_r.shape[-1]))
        seq2seq_active_encoder_r = np.zeros((num_patient_points, max_seq_length - projection_horizon))
        seq2seq_previous_treatments = np.zeros((num_patient_points, projection_horizon, previous_treatments.shape[-1]))
        seq2seq_current_treatments = np.zeros((num_patient_points, projection_horizon, current_treatments.shape[-1]))
        seq2seq_current_covariates = np.zeros((num_patient_points, projection_horizon, current_covariates.shape[-1]))
        seq2seq_outputs = np.zeros((num_patient_points, projection_horizon, outputs.shape[-1]))
        seq2seq_active_entries = np.zeros((num_patient_points, projection_horizon, 1))
        seq2seq_sequence_lengths = np.zeros(num_patient_points)

        for i in range(num_patient_points):
            fact_length = int(sequence_lengths[i]) - projection_horizon
            if encoder_r is not None:
                seq2seq_state_inits[i] = encoder_r[i, fact_length - 1]
            seq2seq_active_encoder_r[i, :fact_length] = 1.0
            seq2seq_active_entries[i] = np.ones(shape=(projection_horizon, 1))
            seq2seq_previous_treatments[i] = previous_treatments[i, fact_length - 1:fact_length + projection_horizon - 1]
            seq2seq_current_treatments[i] = current_treatments[i, fact_length:fact_length + projection_horizon]
            seq2seq_outputs[i] = outputs[i, fact_length:fact_length + projection_horizon]
            seq2seq_sequence_lengths[i] = projection_horizon
            seq2seq_current_covariates[i] = np.repeat([current_covariates[i, fact_length - 1]], projection_horizon, axis=0)

        seq2seq_data = {
            'prev_treatments': seq2seq_previous_treatments,
            'current_treatments': seq2seq_current_treatments,
            'current_covariates': seq2seq_current_covariates,
            'prev_outputs': seq2seq_current_covariates[:, :, :1],
            'static_features': seq2seq_current_covariates[:, 0, 1:],
            'outputs': seq2seq_outputs,
            'sequence_lengths': seq2seq_sequence_lengths,
            'active_entries': seq2seq_active_entries,
            'unscaled_outputs': seq2seq_outputs * self.scaling_params['output_stds'] + self.scaling_params['output_means'],
        }
        if encoder_r is not None:
            seq2seq_data['init_state'] = seq2seq_state_inits
        seq2seq_data['active_encoder_r'] = seq2seq_active_encoder_r

        self.data_original = deepcopy(self.data)
        self.data = seq2seq_data
        self.processed_sequential = True

        if save_encoder_r and encoder_r is not None:
            self.encoder_r = encoder_r[:, :max_seq_length, :]

        return self.data


class GlucoseDatasetCollection(SyntheticDatasetCollection):
    """Dataset collection for glucose-insulin simulator."""

    def __init__(self, data_dir, gamma=4, seed=100, projection_horizon=5,
                 treatment_mode='continuous', **kwargs):
        super(GlucoseDatasetCollection, self).__init__()
        self.seed = seed
        np.random.seed(seed)

        self.train_f = GlucoseDataset(data_dir, gamma, 'train', treatment_mode)
        self.val_f = GlucoseDataset(data_dir, gamma, 'val', treatment_mode)
        self.test_f = GlucoseDataset(data_dir, gamma, 'test', treatment_mode)

        # For VCIP compatibility: test_cf datasets point to test_f
        # (counterfactual evaluation done separately via our RA scripts)
        self.test_cf_one_step = self.test_f
        self.test_cf_treatment_seq = self.test_f

        self.projection_horizon = projection_horizon
        self.autoregressive = True
        self.has_vitals = False
        self.train_scaling_params = self.train_f.get_scaling_params()
