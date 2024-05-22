from torch_geometric.loader import DataLoader

class QCNetDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, cfg, shuffle=None):
        super(QCNetDataLoader, self).__init__(dataset, batch_size, shuffle)
        self.cfg = cfg