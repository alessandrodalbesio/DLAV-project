# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from motionnet.models.qcnet.losses import MixtureNLLLoss
from motionnet.models.qcnet.losses import NLLLoss
from motionnet.models.qcnet.metrics import Brier
from motionnet.models.qcnet.metrics import MR
from motionnet.models.qcnet.metrics import minADE
from motionnet.models.qcnet.metrics import minAHE
from motionnet.models.qcnet.metrics import minFDE
from motionnet.models.qcnet.metrics import minFHE
from motionnet.models.qcnet.modules.qcnet_decoder import QCNetDecoder
from motionnet.models.qcnet.modules.qcnet_encoder import QCNetEncoder
from motionnet.models.base_model.base_model import BaseModel


class QCNet(BaseModel):

    def __init__(self,config):    
        super(QCNet, self).__init__(config)

        self.dataset = config['dataset']
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.output_head = config['output_head']
        self.num_historical_steps = config['num_historical_steps']
        self.num_future_steps = config['num_future_steps']
        self.num_modes = config['num_modes']
        self.num_recurrent_steps = config['num_recurrent_steps']
        self.num_freq_bands = config['num_freq_bands']
        self.num_map_layers = config['num_map_layers']
        self.num_agent_layers = config['num_agent_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.num_heads = config['num_heads']
        self.head_dim = config['head_dim']
        self.dropout = config['dropout']
        self.pl2pl_radius = config['pl2pl_radius']
        self.time_span = config['time_span']
        self.pl2a_radius = config['pl2a_radius']
        self.a2a_radius = config['a2a_radius']
        self.num_t2m_steps = config['num_t2m_steps']
        self.pl2m_radius = config['pl2m_radius']
        self.a2m_radius = config['a2m_radius']
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.T_max = config['T_max']

        self.encoder = QCNetEncoder(
            dataset=self.dataset,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_historical_steps=self.num_historical_steps,
            pl2pl_radius=self.pl2pl_radius,
            time_span=self.time_span,
            pl2a_radius=self.pl2a_radius,
            a2a_radius=self.a2a_radius,
            num_freq_bands=self.num_freq_bands,
            num_map_layers=self.num_map_layers,
            num_agent_layers=self.num_agent_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=self.dropout,
        )
        self.decoder = QCNetDecoder(
            dataset=self.dataset,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            output_head=self.output_head,
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
            num_modes=self.num_modes,
            num_recurrent_steps=self.num_recurrent_steps,
            num_t2m_steps=self.num_t2m_steps,
            pl2m_radius=self.pl2m_radius,
            a2m_radius=self.a2m_radius,
            num_freq_bands=self.num_freq_bands,
            num_layers=self.num_dec_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=self.dropout,
        )

        self.reg_loss = NLLLoss(
            component_distribution=['laplace'] * self.output_dim + ['von_mises'] * self.output_head,
            reduction='none'
        )
        self.cls_loss = MixtureNLLLoss(
            component_distribution=['laplace'] * self.output_dim + ['von_mises'] * self.output_head,
            reduction='none'
        )

        self.Brier = Brier(max_guesses=6)
        self.minADE = minADE(max_guesses=6)
        self.minAHE = minAHE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)
        self.minFHE = minFHE(max_guesses=6)
        self.MR = MR(max_guesses=6)
        

        self.test_predictions = dict()

    def forward(self, data, batch_idx):
        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)

        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]

        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        loss = reg_loss_propose + reg_loss_refine + cls_loss
        #                 32, 6, 60, x>
        # pred_obs: shape [B, c, T, 5] trajectories for the ego agents with every point being the params of
        #                                 Bivariate Gaussian distribution.
        # mode_probs: shape [B, c] mode probability predictions P(z|X_{1:T_obs})
        
  
        # Modify the output to make it compatible with the log_info function
        max_num_agents = self.config['max_num_agents']
        indexes = max_num_agents * torch.arange(data['agent']['target'].shape[0] // max_num_agents)
        pi_filtered = pi[indexes]
        traj_filtered = traj_refine[indexes]
        
        output = dict()
        output['predicted_trajectory'] = traj_filtered
        output['predicted_probability'] = F.softmax(pi_filtered, dim=-1)

        return output, loss


    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    def log_info(self, inputs, prediction, status='train'): 
        # Get the dimensions
        total_future_len = self.config['num_future_steps']
        T = self.config['num_future_steps']
        B = inputs['center_gt_trajs'].shape[0] // total_future_len

        # Modify the ground truth to make it compatible with the log_info function
        gt_traj = inputs['center_gt_trajs'].reshape(B,T,-1).unsqueeze(1)
        gt_traj_mask = inputs['center_gt_trajs_mask'].reshape(B,T,-1).squeeze(-1).unsqueeze(1)
        center_gt_final_valid_idx = torch.tensor(inputs['center_gt_final_valid_idx']).detach().cpu()

        predicted_traj = prediction['predicted_trajectory']
        predicted_prob = prediction['predicted_probability'].detach().cpu().numpy()

        # Calculate ADE losses
        ade_diff = torch.norm(predicted_traj[:, :, :, :2] - gt_traj[:, :, :, :2], 2, dim=-1)
        ade_losses = torch.sum(ade_diff * gt_traj_mask, dim=-1) / torch.sum(gt_traj_mask, dim=-1)
        ade_losses = ade_losses.cpu().detach().numpy()
        minade = np.min(ade_losses, axis=1)

        # Calculate FDE losses
        bs,modes,future_len = ade_diff.shape
        center_gt_final_valid_idx = center_gt_final_valid_idx.view(-1,1,1).repeat(1,modes,1).to(torch.int64)

        fde = torch.gather(ade_diff.cpu().detach(),-1,center_gt_final_valid_idx).cpu().detach().numpy().squeeze(-1)
        minfde = np.min(fde, axis=-1)

        best_fde_idx = np.argmin(fde, axis=-1)
        predicted_prob = predicted_prob[np.arange(bs),best_fde_idx]
        miss_rate = (minfde > 2.0)
        brier_fde = minfde + (1 - predicted_prob)

        loss_dict = {
                    'minADE6': minade,
                    'minFDE6': minfde,
                    'miss_rate': miss_rate.astype(np.float32),
                    'brier_fde': brier_fde}

        new_dict = {}
        dataset_names = inputs['dataset_name']
        unique_dataset_names = np.unique(dataset_names)
        for dataset_name in unique_dataset_names:
            batch_idx_for_this_dataset = np.argwhere([n == str(dataset_name) for n in dataset_names])[:,0]
            for key in loss_dict.keys():
                new_dict[dataset_name+'/'+key] = loss_dict[key][batch_idx_for_this_dataset]
        # merge new_dict with log_dict
        loss_dict.update(new_dict)

        if status == 'val' and self.config.get('eval', False):
            important_metrics = ["brier_fde", "minADE6", "minFDE6", "miss_rate"]

            # Split scores based on trajectory type
            new_dict = {}
            trajectory_types = inputs["trajectory_type"]
            trajectory_correspondance = {0: "stationary", 1: "straight", 2: "straight_right",
                3: "straight_left", 4: "right_u_turn", 5: "right_turn",
                6: "left_u_turn", 7: "left_turn"}
            for traj_type in range(8):
                batch_idx_for_traj_type = np.where(trajectory_types == traj_type)[0]
                if len(batch_idx_for_traj_type) > 0:
                    for key in important_metrics:
                        new_dict["traj_type/" + trajectory_correspondance[traj_type] + "_" + key] = loss_dict[key][batch_idx_for_traj_type]
            loss_dict.update(new_dict)

            # Split scores based on kalman_difficulty @6s
            new_dict = {}
            kalman_difficulties = inputs["kalman_difficulty"][:, -1] # Last is difficulty at 6s (others are 2s and 4s)
            for kalman_bucket, (low, high) in {"easy": [0, 30], "medium": [30, 60], "hard": [60, 9999999]}.items():
                batch_idx_for_kalman_diff = np.where(np.logical_and(low <= kalman_difficulties, kalman_difficulties < high))[0]
                if len(batch_idx_for_kalman_diff) > 0:
                    for key in important_metrics:
                        new_dict["kalman/" + kalman_bucket + "_" + key] = loss_dict[key][batch_idx_for_kalman_diff]
            loss_dict.update(new_dict)

            new_dict = {}
            agent_types = [1,2,3]
            agent_type_dict = {1: "vehicle", 2: "pedestrian", 3: "bicycle"}
            for type in agent_types:
                batch_idx_for_type = np.where(inputs['center_objects_type'] == type)[0]
                if len(batch_idx_for_type) > 0:
                    for key in important_metrics:
                        new_dict["agent_types"+'/'+agent_type_dict[type]+"_"+key] = loss_dict[key][batch_idx_for_type]
            # merge new_dict with log_dict
            loss_dict.update(new_dict)

        # Take mean for each key but store original length before (useful for aggregation)
        size_dict = {key: len(value) for key, value in loss_dict.items()}
        loss_dict = {key: np.mean(value) for key, value in loss_dict.items()}

        for k, v in loss_dict.items():
            self.log(status+"/" + k, v, on_step=False, on_epoch=True, sync_dist=True, batch_size=size_dict[k])
        return