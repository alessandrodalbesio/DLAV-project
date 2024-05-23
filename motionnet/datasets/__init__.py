from .ptr import PTRDataset
from .qcnet import QCNetDataset

__all__ = {
    'ptr': PTRDataset,
    'qcnet': QCNetDataset,
}

def build_dataset(config,val=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val
    )
    return dataset
