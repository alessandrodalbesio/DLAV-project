from pytorch_lightning.callbacks import ModelCheckpoint

class QCNetCheckpoint(ModelCheckpoint):
    def __init__(self):
        super().__init__(
            monitor='val_minFDE',
            filename='{epoch}-{val/min_ade:.2f}',
            save_top_k=1,
            mode='min'
        )