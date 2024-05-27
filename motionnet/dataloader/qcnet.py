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
from typing import Callable, Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from motionnet.datasets.qcnet_dataset import QCNetDataset
from motionnet.utils.target_builder import TargetBuilder


class QCNetDataLoader(pl.LightningDataModule):

    def __init__(self,config) -> None:
        super(QCNetDataLoader, self).__init__()
        self.root = config.method.root
        self.train_batch_size = config.method.train_batch_size
        self.val_batch_size = config.method.val_batch_size
        self.test_batch_size = config.method.test_batch_size
        self.shuffle = config.method.get('shuffle', True)
        self.num_workers = config.method.get('num_workers', 1)
        self.pin_memory = config.method.get('pin_memory', True)
        self.persistent_workers = config.method.get('persistent_workers', True)
        self.train_raw_dir = config.method.get('train_raw_dir', None)
        self.val_raw_dir = config.method.get('val_raw_dir', None)
        self.test_raw_dir = config.method.get('test_raw_dir', None)
        self.train_processed_dir = config.method.get('train_processed_dir', None)
        self.val_processed_dir = config.method.get('val_processed_dir', None)
        self.test_processed_dir = config.method.get('test_processed_dir', None)
        self.train_transform = config.method.get('train_transform', TargetBuilder(50, 60))
        self.val_transform = config.method.get('val_transform', TargetBuilder(50, 60))
        self.test_transform = config.method.get('test_transform', None)

    def prepare_data(self) -> None:
        QCNetDataset(self.root, 'train', self.train_raw_dir, self.train_processed_dir, self.train_transform)
        QCNetDataset(self.root, 'val', self.val_raw_dir, self.val_processed_dir, self.val_transform)
        QCNetDataset(self.root, 'test', self.test_raw_dir, self.test_processed_dir, self.test_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = QCNetDataset(self.root, 'train', self.train_raw_dir, self.train_processed_dir,
                                                self.train_transform)
        self.val_dataset = QCNetDataset(self.root, 'val', self.val_raw_dir, self.val_processed_dir,
                                              self.val_transform)
        self.test_dataset = QCNetDataset(self.root, 'test', self.test_raw_dir, self.test_processed_dir,
                                               self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
