from .ptr_checkpoint import PTRCheckpoint
from .qcnet_checkpoint import QCNetCheckpoint

__all__ = {
    'ptr': PTRCheckpoint,
    'qcnet': QCNetCheckpoint
}

def get_checkpoint(config):
    return __all__[config.method.model_name]()
