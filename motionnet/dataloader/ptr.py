from torch.utils.data import DataLoader

class PTRDataLoader(DataLoader):
    def __init__(self,dataset,batch_size, cfg, shuffle=None):
        super(PTRDataLoader, self).__init__(
            dataset, 
            batch_size=batch_size, 
            num_workers=cfg.load_num_workers, 
            drop_last=False,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle
        )