from motionnet.datasets import build_dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class PTRDataLoader(pl.LightningDataModule):

    def __init__(self,config,**kwargs):
        super(PTRDataLoader, self).__init__()
        self.cfg = config

    def setup(self, stage=None) -> None:
        # Build the datasets 
        self.train_dataset = build_dataset(self.cfg)
        self.val_dataset = build_dataset(self.cfg,val=True)

        # Define the train and validation batch sizes
        self.train_batch_size = max(self.cfg.method['train_batch_size'] // len(self.cfg.devices) // self.train_dataset.data_chunk_size,1)
        self.eval_batch_size = max(self.cfg.method['eval_batch_size'] // len(self.cfg.devices) // self.val_dataset.data_chunk_size,1)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.train_batch_size, 
            num_workers=self.cfg.load_num_workers,
            drop_last=False,
            collate_fn=self.train_dataset.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.eval_batch_size, 
            num_workers=self.cfg.load_num_workers, 
            shuffle=False, 
            drop_last=False,
            collate_fn=self.val_dataset.collate_fn
        )
