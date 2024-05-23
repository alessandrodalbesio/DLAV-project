from torch_geometric.loader import DataLoader

class QCNetDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, cfg, shuffle=None, **kwargs):
        super(QCNetDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=cfg.load_num_workers,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=True,
        )
        self.cfg = cfg
