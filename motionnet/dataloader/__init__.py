from .ptr import PTRDataLoader
from .qcnet import QCNetDataLoader

__all__ = {
    'ptr': PTRDataLoader,
    'qcnet': QCNetDataLoader,
}

def build_dataloader(dataset,batch_size, cfg, shuffle=None):
    dataloader = __all__[cfg.method.model_name](
        dataset,batch_size,cfg,shuffle
    )
    return dataloader
