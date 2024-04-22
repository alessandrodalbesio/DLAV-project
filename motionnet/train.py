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
import os

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):

    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    # Save the config file to a config folder
    save_path = os.path.join(os.getcwd(), "training_configs")
    os.makedirs(save_path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_path, f"{cfg.exp_name}.yaml"))

    # Build the model
    model = build_model(cfg)

    # Build the dataset
    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg,val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices) // train_set.data_chunk_size,1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices) // val_set.data_chunk_size,1)

    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val/brier_fde',    # Replace with your validation metric
        filename='{epoch}-{val/brier_fde:.2f}',
        save_top_k=1,
        mode='min',            # 'min' for loss/error, 'max' for accuracy
    )

    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
    collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
    collate_fn=train_set.collate_fn)

    if cfg.local_training:
        # Verify if cuda is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_on_CPU = device == torch.device("cpu")

        # Set the trainer
        trainer = pl.Trainer(
            max_epochs=cfg.method.max_epochs,
            logger= None if cfg.debug else WandbLogger(project="motionnet", name=cfg.exp_name),
            devices=1 if is_on_CPU else cfg.devices,
            gradient_clip_val=cfg.method.grad_clip_norm,
            accelerator="cpu" if is_on_CPU else "gpu",
            profiler="simple",
            strategy="auto",
            callbacks=call_backs
        )
    else:
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
        trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader,ckpt_path=cfg.ckpt_path)

if __name__ == '__main__':
    train()

