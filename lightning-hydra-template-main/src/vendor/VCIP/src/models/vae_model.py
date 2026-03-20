import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np  
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.auxiliary_model import AuxiliaryModel
from src.models.generative_model import GenerativeModel
from src.models.inference_model import InferenceModel
from src.utils.helper_functions import compute_kl_divergence
from src.data.cip_dataset import CIPDataset, get_dataloader
from src.utils.utils import set_seed
from scipy.stats import spearmanr
# from scipy.stats import spearmanr
from scipy import stats
from src.utils.helper_functions import generate_perturbed_sequences
import os
import pickle

class VAEModel(pl.LightningModule):
    def __init__(self, config, dataset_collection):
        super(VAEModel, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.dataset_collection = dataset_collection
        self.generative_model = GenerativeModel(config)
        self.inference_model = InferenceModel(config)
        self.auxiliary_model = AuxiliaryModel(config, dataset_collection)
        
        self.lr = config['model']['lr']
        self.tau = config['exp']['tau']
        self.a_dim = config['dataset']['treatment_size']
        self.predict_X = config['dataset']['predict_X']
        # self.init_AuxiliaryModel(config)
        self.automatic_optimization = False
        self.validation_step_outputs = []
        self.train_step_outputs = []
        # print(self.dataset_collection.train_scaling_params['output_stds'])
        try:
            self.std = self.dataset_collection.train_scaling_params[1]['cancer_volume']
        except:
            self.std = 1
    
    def init_AuxiliaryModel(self, config):
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['exp']['patience'],
            mode='min')
        logger_board = TensorBoardLogger(
            save_dir=config['exp']['current_dir'], 
            name='', 
            version='')
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=None,
            monitor='val_loss',
            filename='checkpoint-{epoch:02d}-{val_loss:.3f}',
            save_top_k=3,
            mode='min')
        trainer = pl.Trainer(
            logger=logger_board,
            max_epochs=config['exp']['epochs'],
            enable_progress_bar=False,
            enable_model_summary=False, 
            devices=config['exp']['gpus'],
            callbacks=[early_stop_callback, checkpoint_callback])
        trainer.fit(self.auxiliary_model)

    def train_dataloader(self) -> DataLoader:
        return get_dataloader(CIPDataset(self.dataset_collection.train_f.data, self.config, train=True), batch_size=self.config['exp']['batch_size'], shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return get_dataloader(CIPDataset(self.dataset_collection.val_f.data, self.config), batch_size=self.config['exp']['batch_size'], shuffle=False)

    def forward(self, H_t, Y_targets, a_seq, X_targets=None):
        elbo, reg_loss, _ = self.calculate_elbo(H_t, Y_targets, a_seq, X_targets=X_targets, num_samples=self.config.exp.num_samples, optimize_a=False)
        return elbo, reg_loss

    def training_step(self, batch, batch_idx):
        self.train()
        
        H_t, targets = batch
        Y_targets = targets['outputs']
        X_targets = None if not self.predict_X else targets['vitals']
        a_seq = targets['current_treatments']

        opt = self.optimizers()
        opt.zero_grad()

        elbo, reg_loss = self(H_t, Y_targets, a_seq, X_targets=X_targets)
        self.manual_backward(elbo)
        opt.step()

        self.log('train_loss', elbo)
        self.train_step_outputs.append(reg_loss)
        return elbo

    def validation_step(self, batch, batch_idx):
        H_t, targets = batch
        Y_targets = targets['outputs']
        X_targets = None if not self.predict_X else targets['vitals']
        a_seq = targets['current_treatments']
        elbo, reg_loss = self(H_t, Y_targets, a_seq, X_targets=X_targets)
        self.log('val_loss', elbo)
        self.log('val_reg_loss', reg_loss)
        self.validation_step_outputs.append(reg_loss)
        return reg_loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x for x in self.validation_step_outputs]).mean()
        print(f"val_reg_loss loss: {avg_loss.item()}")
        self.validation_step_outputs.clear()
        return

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x for x in self.train_step_outputs]).mean()
        print(f"train_reg_loss loss: {avg_loss.item()}")
        self.train_step_outputs.clear()
        return

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def configure_optimizers(self):
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"Warning: {name} does not require gradients")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.config['exp']['weight_decay'])
        param_groups = optimizer.param_groups[0]['params']
        all_params = set(self.parameters())
        if len(param_groups) != len(list(all_params)):
            print("Warning: Not all parameters are in optimizer")
        print(f"Number of parameters: {len(list(all_params))}")
        
        return optimizer


    def optimize_interventions_(self, num_iterations=100, learning_rate=0.01):
        set_seed(self.config['exp']['seed'])
        dataloader = get_dataloader(CIPDataset(self.dataset_collection.val_f.data, self.config), batch_size=self.config['exp']['batch_size'], shuffle=False)
        # dataloader = get_dataloader(CIPDataset(self.dataset_collection.val_f.data, self.config), batch_size=1, shuffle=False)
        
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        self.inference_model.to(device)
        self.generative_model.to(device)
        self.auxiliary_model.to(device)

        self.inference_model.train()
        self.generative_model.train()
        self.auxiliary_model.train()

        results = []
        losses = []
        losses_2 = []
        ture_output_list = []
        output_after_actions_list = []
        ture_output_actions_list = []
        for batch in dataloader:
            # 如果未提供初始干预序列，则随机初始化
            initial_a_seq = None
            H_t, targets = batch
            for key in H_t:
                H_t[key] = H_t[key].to(device)
            for key in targets:
                targets[key] = targets[key].to(device)
            Y_targets = targets['outputs']
            batch_size = Y_targets.size(0)
            
            if initial_a_seq is None:
                # a_seq = torch.randn(batch_size, self.tau, self.a_dim, device=device, requires_grad=True)
                a_seq = torch.rand(batch_size, self.tau, self.a_dim, device=device, requires_grad=True)
            else:
                a_seq = initial_a_seq.clone().detach().requires_grad_(True)
            
            # 优化器（用于优化干预序列 a_seq）
            optimizer = torch.optim.Adam([a_seq], lr=learning_rate)
            print(f"-" * 50)
            for iteration in range(num_iterations):
                a_seq_sigmoid = torch.sigmoid(a_seq)
                optimizer.zero_grad()
                elbo, reg_loss = self.calculate_elbo(H_t, Y_targets, a_seq_sigmoid, num_samples=self.config.exp.num_samples, optimize_a=True)
                loss = elbo
                loss.backward()
                optimizer.step()
            optimized_a_seq = a_seq.detach()
            if num_iterations > 0:
                optimized_a_seq = torch.sigmoid(optimized_a_seq)

            output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(H_t, optimized_a_seq, self.dataset_collection.train_scaling_params)
            ture_output = targets['outputs'][:, -1, :].detach().cpu().numpy()

            true_actions = targets['current_treatments']
            ture_output_actions = self.dataset_collection.val_f.simulate_output_after_actions(H_t, true_actions, self.dataset_collection.train_scaling_params)

            # print(f'shape of output_after_actions: {output_after_actions.shape}, shape of ture_output: {ture_output.shape}')
            loss = np.sqrt(((output_after_actions-ture_output) ** 2).mean())
            losses.append(loss)

            loss_2 = np.sqrt(((ture_output_actions-ture_output) ** 2).mean())
            losses_2.append(loss_2)

            ture_output_list.append(ture_output)
            output_after_actions_list.append(output_after_actions)
            ture_output_actions_list.append(ture_output_actions)

            results.append(optimized_a_seq)

        true_actions = targets['current_treatments']
        # print(f"true_actions: {true_actions}")
        ture_output_actions = self.dataset_collection.val_f.simulate_output_after_actions(H_t, true_actions, self.dataset_collection.train_scaling_params)

         
        print(f"Mean loss v1: {sum(losses) / len(losses)}")
        print(f"Mean loss_2 v2: {sum(losses_2) / len(losses_2)}")
        ture_output_list = np.concatenate(ture_output_list, axis=0)
        output_after_actions_list = np.concatenate(output_after_actions_list, axis=0)
        ture_output_actions_list = np.concatenate(ture_output_actions_list, axis=0)
        loss = np.sqrt(((output_after_actions_list-ture_output_list) ** 2).mean())
        loss_2 = np.sqrt(((ture_output_actions_list-ture_output_list) ** 2).mean()) 
        print(f"Mean loss v2: {loss}")
        print(f"Mean loss_2 v2: {loss_2}")

        return torch.cat(results, dim=0), loss

    def optimize_interventions(self, num_iterations=100, learning_rate=0.01, batch_size=64):
        results = ['\n' + '-' * 50]
        tau = self.tau
        for i in range(1, 13):
        # for i in [2, 6, 9]:
            print(f'start predicting results for tau={i} ...')
            self.tau = i
            self.config.exp.tau = i
            print(self.tau, self.config.exp.tau)
            _, loss = self.optimize_interventions_onetime(num_iterations=num_iterations, learning_rate=learning_rate, batch_size=batch_size)
            results.append(f"Optimized interventions for tau={i}, {num_iterations} iterations, lr={learning_rate}: {loss}")
        return '\n'.join(results)

    def optimize_interventions_onetime(self, num_iterations=100, learning_rate=0.01, batch_size=64):
        set_seed(self.config['exp']['seed'])
        
        if not self.config.exp.test:
            data = self.dataset_collection.val_f.data
        else:
            data = self.dataset_collection.test_f.data
        # data = self.dataset_collection.val_f.data
        dataloader = get_dataloader(CIPDataset(data, self.config), batch_size=batch_size, shuffle=False)
        # dataloader = get_dataloader(CIPDataset(self.dataset_collection.test_f.data, self.config), batch_size=batch_size, shuffle=False)
        
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        self.inference_model.to(device)
        self.generative_model.to(device)
        self.auxiliary_model.to(device)

        self.inference_model.train()
        self.generative_model.train()
        self.auxiliary_model.train()

        results = []
        losses = []
        losses_2 = []
        ture_output_list = []
        output_after_actions_list = []
        ture_output_actions_list = []
        elbos = []
        
        for (i, batch) in enumerate(dataloader):
            # if i not in [46, 99, 53, 36, 83]:
            # if i not in [99]:
            #     continue
            print(f"-" * 50 + f"Batch {i}" + "-" * 50)
            H_t, targets = batch
            for key in H_t:
                H_t[key] = H_t[key].to(device)
            for key in targets:
                targets[key] = targets[key].to(device)
            Y_targets = targets['outputs']
            X_targets = None if not self.predict_X else targets['vitals']
            print(f"shape of Y_targets: {Y_targets.shape}")
            batch_size = Y_targets.size(0)
            
            # Initialize best tracking variables
            # best_a_seq = None
            best_elbo = float('inf')
            best_step = 0

            best_reg = float('inf')
            best_step_reg = 0
            
            a_seq = torch.rand(batch_size, self.tau, self.a_dim, device=device, requires_grad=True)
            # a_seq = torch.normal(mean=2, std=0.1, 
            #         size=(batch_size, self.tau, self.config.dataset.treatment_size),
            #         device=device, requires_grad=True)
            best_a_seq = torch.sigmoid(a_seq).clone().detach()
            best_a_seq_reg = a_seq.clone().detach()
            optimizer = torch.optim.Adam([a_seq], lr=learning_rate)

            print(f"-" * 50)
            for iteration in range(num_iterations):
                print('*' * 50)
                a_seq_sigmoid = torch.sigmoid(a_seq)
                optimizer.zero_grad()
                elbo, reg_loss, reg_loss_2 = self.calculate_elbo(H_t, Y_targets, a_seq_sigmoid, X_targets=X_targets,
                                                num_samples=self.config.exp.num_samples, 
                                                optimize_a=True)
                loss = elbo
                loss.backward()
                optimizer.step()
                # print(f"Iteration {iteration}: ELBO = {elbo.item()}, reg_loss = {reg_loss.item()}")
                
                # Check if current sequence is better
                if elbo.item() < best_elbo:
                    best_elbo = elbo.item()
                    best_a_seq = torch.sigmoid(a_seq.clone().detach())
                    elbos.append(best_elbo)
                    best_step = iteration

                if reg_loss.item() < best_reg:
                    best_reg = reg_loss.item()
                    best_step_reg = iteration
                    best_a_seq_reg = torch.sigmoid(a_seq.clone().detach())
                
                loss_outcome_after_actions, output_after_actions = self.get_loss_outcome_after_actions(H_t, Y_targets, a_seq_sigmoid.detach())
                ture_output = targets['outputs'][:, -1, :].detach().cpu().numpy()
                loss = np.sqrt(((output_after_actions-ture_output) ** 2).mean())
                print(f"Iteration {iteration}: elbo = {elbo.item()}, loss = {loss}")

                # print(f"Iteration {iteration}: elbo = {elbo.item()}, loss_after_actions = {loss_outcome_after_actions}, reg_loss: {reg_loss}")
                # print(f"targets['current_treatments']: {targets['current_treatments']}")
                # print(f"a_seq_sigmoid: {a_seq_sigmoid}")

            # Use the best sequence found
            print(f"Best step: {best_step}, best ELBO: {best_elbo}")

            if best_reg < 0:
                best_a_seq = best_a_seq_reg

            optimized_a_seq = best_a_seq
            
            output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(
                H_t, optimized_a_seq, self.dataset_collection.train_scaling_params)
            ture_output = targets['outputs'][:, -1, :].detach().cpu().numpy()

            true_actions = targets['current_treatments']
            ture_output_actions = self.dataset_collection.val_f.simulate_output_after_actions(
                H_t, true_actions, self.dataset_collection.train_scaling_params)

            loss = np.sqrt(((output_after_actions-ture_output) ** 2).mean())
            losses.append(loss)

            loss_2 = np.sqrt(((ture_output_actions-ture_output) ** 2).mean())
            losses_2.append(loss_2)

            ture_output_list.append(ture_output)
            output_after_actions_list.append(output_after_actions)
            ture_output_actions_list.append(ture_output_actions)

            results.append(optimized_a_seq)
        
        # print(f"Mean elbo: {sum(elbos) / len(elbos)}")

        true_actions = targets['current_treatments']
        ture_output_actions = self.dataset_collection.val_f.simulate_output_after_actions(
            H_t, true_actions, self.dataset_collection.train_scaling_params)

        print(f"Mean loss v1: {sum(losses) / len(losses)}")
        print(f"losses: {losses}")
        print(f"Mean loss_2 v2: {sum(losses_2) / len(losses_2)}")
        
        ture_output_list = np.concatenate(ture_output_list, axis=0)
        output_after_actions_list = np.concatenate(output_after_actions_list, axis=0)
        ture_output_actions_list = np.concatenate(ture_output_actions_list, axis=0)
        print(f"ture_output_list shape: {ture_output_list.shape}")
        print(f"first 10 ture_output_list: {ture_output_list[:10]}")
        
        loss = np.sqrt(((output_after_actions_list-ture_output_list) ** 2).mean())
        loss_2 = np.sqrt(((ture_output_actions_list-ture_output_list) ** 2).mean()) 
        
        print(f"Mean loss v2: {loss}")
        print(f"Mean loss_2 v2: {loss_2}")

        return torch.cat(results, dim=0), loss * self.std

    def get_loss_outcome_after_actions(self, H_t, Y_targets, a_seq):
        output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(H_t, a_seq, self.dataset_collection.train_scaling_params)
        ture_output = Y_targets[:, -1, :].detach().cpu().numpy()
        loss = np.sqrt(((output_after_actions-ture_output) ** 2).mean())
        loss = F.mse_loss(torch.tensor(output_after_actions, device=Y_targets.device, dtype=Y_targets.dtype), Y_targets[:, -1, :])
        return loss, output_after_actions

    def calculate_elbo(self, H_t, Y_targets, a_seq, X_targets=None, num_samples=10, optimize_a=False):
        H_rep = self.auxiliary_model.build_representations(H_t, only_last=True)
        
        Z_s_i, loss_hy = self.inference_model.init_hidden_history(H_t)

        if self.config.dataset.treatment_mode == 'multilabel':
            predict_action_loss = self.generative_model.loss_predict_actions_bern(Z_s_i, a_seq)
        else:
            predict_action_loss = self.generative_model.loss_predict_actions_beta(Z_s_i, a_seq)
        
        # Reset states
        self.inference_model.reset_states()
        self.generative_model.reset_states()
        
        # Pre-compute action encodings
        a_seq_hiddens_1 = self.generative_model.build_action_encoding(a_seq)
        
        a_seq_hiddens = self.generative_model.build_reverse_action_encoding(a_seq)
        
        kl_losses = []
        reg_losses = []
        action_losses = []

        Y_last = Y_targets[:, -1, :]
        reg_loss_2 = None
        
        hidden_states = self.generative_model.build_hidden_states(Z_s_i, a_seq)
        reversed_hidden_states = self.generative_model.build_reverse_hidden_states(hidden_states)

        for s in range(self.tau):
            a_s = a_seq[:, s, :]
            a_s_hidden = a_seq_hiddens[:, s, :]
                
            Z_s_inf = torch.cat([Z_s_i, Z_s_i], dim=-1)
            
            if s < self.tau - 1:
                a_s_next = a_seq[:, s+1, :]
                if self.config.dataset.treatment_mode == 'multilabel':
                    action_loss = self.generative_model.bern_loss(Z_s_i, a_s_next)
                else:
                    action_loss = self.generative_model.beta_loss(Z_s_i, a_s_next)
                # action_loss = self.generative_model.mmd_loss(Z_s_i, a_s_next)
                action_losses.append(action_loss)
            
            # Inference network
            q_mu, q_logvar = self.inference_model(Z_s_inf, a_s_hidden, H_rep, Y_last)

            # Generative network
            p_mu, p_logvar = self.generative_model(Z_s_i, a_s_hidden)
            
            # Multiple sampling
            Z_samples = self.inference_model.reparameterize_multiple(q_mu, q_logvar, num_samples)
            # print(f"Z_samples shape: {Z_samples.shape}")
            
            # Compute losses
            kl_loss = compute_kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
            kl_losses.append(kl_loss)

            if s > 0 and s < self.tau - 1:
                Y_current = Y_targets[:, s, :].unsqueeze(0).expand(num_samples, -1, -1)
                a = a_seq_hiddens_1[:, -1, :].unsqueeze(0).expand(num_samples, -1, -1)
                reg_loss = self.generative_model.decoding_Y_loss_2(Z_samples, Y_current, a)
                if self.predict_X:
                    X_current = X_targets[:, s, :].unsqueeze(0).expand(num_samples, -1, -1)
                    reg_loss += self.generative_model.decoding_X_loss(Z_samples, X_current)
                reg_losses.append(reg_loss)

            if s == self.tau - 1:
                Y_current = Y_targets[:, s, :]
                a = a_seq_hiddens_1[:, -1, :]
                reg_losses.append(self.generative_model.decoding_Y_loss_2(Z_samples.mean(dim=0), Y_current, a))

                # Y_current = Y_targets[:, s, :].unsqueeze(0).expand(num_samples, -1, -1)
                # a = a_seq_hiddens_1[:, s, :].unsqueeze(0).expand(num_samples, -1, -1)
                # reg_losses.append(self.generative_model.decoding_Y_loss_2(Z_samples, Y_current, a))
                
                # reg_losses.append(self.generative_model.decoding_Y_loss(Z_samples.mean(dim=0), Y_current))
                
            if optimize_a:
                if s == self.tau - 1:
                    Y_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(H_t, a_seq.detach(), self.dataset_collection.train_scaling_params)
                    Y_preds = self.generative_model.decode(Z_samples)
                    Y_after_actions = torch.tensor(Y_after_actions, device=Y_preds.device, dtype=Y_preds.dtype)
                    # mean over each num_samples
                    Y_preds = Y_preds.mean(dim=0)
                    # print(f"Y_after_actions shape: {Y_after_actions.shape}")
                    # print(f"Y_preds shape: {Y_preds.shape}")
                    reg_loss_2 = F.mse_loss(Y_after_actions, Y_preds)

                    topk1 = self.print_topk_diff(Y_after_actions.flatten(), Y_preds.flatten())
                    topk2 = self.print_topk_diff(Y_after_actions.flatten(), Y_current.mean(dim=0).flatten())
                    topk3 = self.print_topk_diff(Y_preds.flatten(), Y_current.mean(dim=0).flatten())

                    reg_loss_3 = F.mse_loss(Y_current.mean(dim=0), Y_after_actions)
                    # print(f"reg_loss_2: {reg_loss_2.item()}, reg_loss_3: {reg_loss_3.item()}")
                    output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(H_t, a_seq.detach(), self.dataset_collection.train_scaling_params)
                    ture_output = Y_current.mean(dim=0).cpu().numpy()
                    loss_1 = np.sqrt(((output_after_actions-ture_output) ** 2).mean()) * self.std
                    loss_2 = np.sqrt(((Y_preds.detach().cpu().numpy()-ture_output) ** 2).mean()) * self.std
                    loss_3 = np.sqrt(((output_after_actions-Y_preds.detach().cpu().numpy()) ** 2).mean()) * self.std
                    
                    # print(f"loss_1: {loss_1}, loss_2: {loss_2}, loss_3: {loss_3}")

            # Update state
            Z_s_i = Z_samples.mean(dim=0)
            
        
        # Aggregate losses
        kl_loss = torch.stack(kl_losses).mean()
        if optimize_a:
            reg_loss = reg_losses[-1]
        else:
            reg_loss = torch.stack(reg_losses).mean()
            if self.config.exp.remove_aux:
                reg_loss = reg_losses[-1]
            reg_loss = reg_losses[-1]

        action_loss = torch.stack(action_losses).mean() if action_losses else 0
        
        # Compute final ELBO
        if optimize_a:
            elbo = reg_loss * self.config.exp.lambda_reg + \
                   kl_loss * self.config.exp.lambda_kl + \
                   action_loss * self.config.exp.lambda_step + \
                   predict_action_loss * self.config.exp.lambda_action
                    
            # elbo = reg_loss + kl_loss
        else:
            elbo  = reg_loss * self.config.exp.lambda_reg + \
                   kl_loss * self.config.exp.lambda_kl + \
                   action_loss * self.config.exp.lambda_step + \
                   predict_action_loss * self.config.exp.lambda_action + \
                   loss_hy * self.config.exp.lambda_hy

        return elbo, reg_loss, reg_loss_2

    def print_topk_diff(self, a, b, k=5):
        k = min(k, len(a))
        diff = torch.abs(a - b)
        topk = torch.topk(diff, k=k)
        # print(f"Top 5 differences: {topk.values}")
        # print(f"Indices: {topk.indices}")
        return topk

    def compute_counterfactual_consistency(self, H_t, Y_targets, a_seq_observed,
                                             a_seq_alternatives, num_samples=10):
        """Compute VCI-style latent divergence diagnostic (Wu et al., ICLR 2025).

        For each timestep s, measures DKL[q(Z_s | a_obs) || q(Z_s | a_alt)] to assess
        how much the latent representation depends on the treatment sequence. Low divergence
        indicates Z_s captures patient-specific features rather than treatment information,
        suggesting more reliable counterfactual predictions.

        Args:
            H_t: Patient history dict (same format as calculate_elbo input).
            Y_targets: Target outcomes (batch_size, tau, output_dim).
            a_seq_observed: Observed action sequence (batch_size, tau, treatment_dim).
            a_seq_alternatives: List of alternative action sequences, each (batch_size, tau, treatment_dim).
            num_samples: Number of MC samples for reparameterization.

        Returns:
            dict with keys:
                'mean_kl': Scalar mean KL divergence across timesteps and alternatives.
                'per_step_kl': List of KL values per timestep (averaged over alternatives).
                'per_alt_kl': List of KL values per alternative (averaged over timesteps).
        """
        self.eval()
        with torch.no_grad():
            # Forward pass under observed actions
            H_rep = self.auxiliary_model.build_representations(H_t, only_last=True)
            Z_s_i_obs, _ = self.inference_model.init_hidden_history(H_t)
            self.inference_model.reset_states()
            self.generative_model.reset_states()

            a_seq_hiddens_obs = self.generative_model.build_reverse_action_encoding(a_seq_observed)
            Y_last = Y_targets[:, -1, :]

            # Collect q distributions under observed actions
            obs_q_mus = []
            obs_q_logvars = []
            Z_s_i = Z_s_i_obs.clone()

            for s in range(self.tau):
                a_s_hidden = a_seq_hiddens_obs[:, s, :]
                Z_s_inf = torch.cat([Z_s_i, Z_s_i], dim=-1)
                q_mu, q_logvar = self.inference_model(Z_s_inf, a_s_hidden, H_rep, Y_last)
                obs_q_mus.append(q_mu.clone())
                obs_q_logvars.append(q_logvar.clone())
                Z_samples = self.inference_model.reparameterize_multiple(q_mu, q_logvar, num_samples)
                Z_s_i = Z_samples.mean(dim=0)

            # For each alternative action sequence, compute q distributions and KL
            all_kls = []  # shape will be (num_alternatives, tau)
            for a_seq_alt in a_seq_alternatives:
                self.inference_model.reset_states()
                self.generative_model.reset_states()

                a_seq_hiddens_alt = self.generative_model.build_reverse_action_encoding(a_seq_alt)
                Z_s_i = Z_s_i_obs.clone()
                alt_kls = []

                for s in range(self.tau):
                    a_s_hidden_alt = a_seq_hiddens_alt[:, s, :]
                    Z_s_inf = torch.cat([Z_s_i, Z_s_i], dim=-1)
                    q_mu_alt, q_logvar_alt = self.inference_model(
                        Z_s_inf, a_s_hidden_alt, H_rep, Y_last
                    )
                    # KL between observed and alternative posteriors
                    kl = compute_kl_divergence(
                        obs_q_mus[s], obs_q_logvars[s],
                        q_mu_alt, q_logvar_alt
                    )
                    alt_kls.append(kl.item())
                    # Advance state using alternative actions
                    Z_samples = self.inference_model.reparameterize_multiple(
                        q_mu_alt, q_logvar_alt, num_samples
                    )
                    Z_s_i = Z_samples.mean(dim=0)

                all_kls.append(alt_kls)

        all_kls = np.array(all_kls)  # (num_alternatives, tau)
        return {
            'mean_kl': float(all_kls.mean()),
            'per_step_kl': all_kls.mean(axis=0).tolist(),  # (tau,)
            'per_alt_kl': all_kls.mean(axis=1).tolist(),    # (num_alternatives,)
        }

    def optimize_interventions_discrete(self):
        results = {}
        model = self.config.model.name.split('/')[0]
        self.eval()
        tau = self.tau
        for i in [2, 4, 6, 8]:
            print(f'start predicting results for tau={i} ...')
            self.tau = i
            self.config.exp.tau = i
            print(self.tau, self.config.exp.tau)
            infos = self.optimize_interventions_discrete_onetime()
            results[i] = infos
        final = {model:results}
        save_dir = f'case_infos/{self.config.exp.seed}/{self.config.exp.test}'
        self.tau = tau

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_dir + f'/case_infos_{model}.pkl', 'wb') as f:
            pickle.dump(final, f)

        write_str = []
        for key in results:
            rank = np.mean([results[key][i]['true_sequence_rank'] for i in range(len(results[key]))])
            write_str.append(f"{key}: Average rank of true sequences across all individuals: {rank}")
        with open(save_dir + f'/case_infos_{model}.txt', 'w') as f:
            f.write('\n'.join(write_str))

        return final

    def optimize_interventions_discrete_onetime(self, k=100):
        set_seed(self.config['exp']['seed'])
        if not self.config.exp.test:
            data = self.dataset_collection.val_f.data
        else:
            data = self.dataset_collection.test_f.data
        dataloader = get_dataloader(CIPDataset(data, self.config),
                                batch_size=1, shuffle=False)
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        self.inference_model.to(device)
        self.generative_model.to(device)
        self.auxiliary_model.to(device)

        self.inference_model.eval()
        self.generative_model.eval()
        self.auxiliary_model.eval()

        all_ranks = []
        all_correlation = []
        case_infos = []

        for (i, batch) in enumerate(dataloader):
            if i > 99:
                break
            print(f"-" * 50 + f"Individual {i}" + "-" * 50)
            H_t, targets = batch
            for key in H_t:
                H_t[key] = H_t[key].to(device)
            for key in targets:
                targets[key] = targets[key].to(device)
            Y_targets = targets['outputs']
            X_targets = None if not self.predict_X else targets['vitals']
            true_actions = targets['current_treatments']

            all_sequences = generate_perturbed_sequences(true_actions, k, self.tau, self.a_dim, device, treatment_mode=self.config.dataset.treatment_mode)

            elbos = []
            true_losses = []
            with torch.no_grad():
                for seq in all_sequences:
                    # Compute ELBO
                    elbo, _, _ = self.calculate_elbo(H_t, Y_targets, seq, X_targets=X_targets,
                                                num_samples=self.config.exp.num_samples,
                                                optimize_a=True)
                    elbos.append(elbo.item())

                    output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(
                        H_t, seq, self.dataset_collection.train_scaling_params)

                    true_output = targets['outputs'][:, -1, :].detach().cpu().numpy()
                    true_loss = np.sqrt(((output_after_actions - true_output) ** 2).mean())
                    true_losses.append(true_loss)

            model_losses = np.array(elbos)
            true_losses = np.array(true_losses)

            corr_model_true, _ = spearmanr(model_losses, true_losses)

            case_info = {
                'individual_id': i,
                'model_losses': model_losses,
                'true_losses': true_losses,
                'correlations': {
                    'model_true': corr_model_true,
                },
                'true_sequence': true_actions.cpu().numpy(),
                'true_sequence_rank': np.sum(model_losses < model_losses[-1]) + 1,
                'all_sequences': all_sequences.cpu().numpy(),  # (k, 1, tau, a_dim)
            }

            case_infos.append(case_info)

            elbos = np.array(elbos)
            true_losses = np.array(true_losses)

            correlation, p_value = spearmanr(elbos, true_losses)
            all_correlation.append(correlation)

            true_seq_rank = np.sum(elbos < elbos[-1]) + 1
            all_ranks.append(true_seq_rank)

            print(f"Individual {i} - True sequence rank: {true_seq_rank} out of {k}")
            print(f"ELBO for true sequence: {elbos[-1]}")
            print(f"True Loss for true sequence: {true_losses[-1]}")
            print(f"Rank correlation for this individual: {correlation:.3f} (p-value: {p_value:.3f})")

        all_correlation = [c for c in all_correlation if not np.isnan(c)]
        avg_rank = sum(all_ranks) / len(all_ranks)
        if len(all_correlation) > 0:
            avg_correlation = sum(all_correlation) / len(all_correlation)
        else:
            avg_correlation = 0
        print(f"Average rank of true sequences across all individuals: {avg_rank:.2f} out of {k}")
        print(f"Average rank correlation across all individuals: {avg_correlation:.3f}")

        return case_infos
        