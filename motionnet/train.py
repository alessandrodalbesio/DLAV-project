import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import DictConfig, OmegaConf
from motionnet.dataloader import build_dataloader
from motionnet.checkpoints import get_checkpoint

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)

    call_backs = []

    checkpoint_callback = get_checkpoint(cfg)

    call_backs.append(checkpoint_callback)
    datamodule = build_dataloader(cfg)
    
    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger= None if cfg.debug else WandbLogger(project="motionnet", name=cfg.exp_name),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if cfg.debug else "ddp",
        callbacks=call_backs
    )

    if cfg.ckpt_path is not None:
        trainer.validate(model=model, dataloaders=datamodule, ckpt_path=cfg.ckpt_path)

    trainer.fit(model=model,datamodule=datamodule,ckpt_path=cfg.ckpt_path)

if __name__ == '__main__':
    train()

