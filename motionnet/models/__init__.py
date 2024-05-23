from motionnet.models.ptr.ptr import PTR
from motionnet.models.qcnet.qcnet import QCNet

__all__ = {
    'ptr': PTR,
    'qcnet': QCNet,
}


def build_model(config):

    model = __all__[config.method.model_name](
        config=config
    )

    return model
