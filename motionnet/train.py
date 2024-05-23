import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from dataloader import build_dataloader
from utils.utils import set_seed
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):

    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)

    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg,val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices) // train_set.data_chunk_size,1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices) // val_set.data_chunk_size,1)

    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val/minADE6',    # Replace with your validation metric
        filename='{epoch}-{val/min_ade:.2f}',
        save_top_k=1,
        mode='min',            # 'min' for loss/error, 'max' for accuracy
    )

    call_backs.append(checkpoint_callback)

    # Define the build dataloader function
    train_loader = build_dataloader(train_set,train_batch_size,cfg)
    val_loader = build_dataloader(val_set,eval_batch_size,cfg,shuffle=False)

    # Define the trainer
    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger= None ,#if cfg.debug else WandbLogger(project="motionnet", name=cfg.exp_name),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto", #if cfg.debug else "ddp",
        callbacks=call_backs
    )

    if cfg.ckpt_path is not None:
        trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader,ckpt_path=cfg.ckpt_path)

if __name__ == '__main__':
    train()

