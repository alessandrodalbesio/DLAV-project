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

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * self.output_dim + ['von_mises'] * self.output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * self.output_dim + ['von_mises'] * self.output_head,
                                       reduction='none')

        self.Brier = Brier(max_guesses=6)
        self.minADE = minADE(max_guesses=6)
        self.minAHE = minAHE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)
        self.minFHE = minFHE(max_guesses=6)
        self.MR = MR(max_guesses=6)
        

        self.test_predictions = dict()

    def forward(self, data, batch_idx):
        # data is already a HeteroData object in the original QCNet implementation, from the get function of the datasets/argoverse_v2_dataset.py file
        # this does not work: data = HeteroData(data)
        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)
        return pred


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
