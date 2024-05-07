from .base_dataset import BaseDataset



class PTRDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)

    def collate_fn(self, data_list):
        # Call the collate_fn of the base class
        batch_dict = super().collate_fn(data_list)

        # Do data augmentation on batch_dict
        if self.is_validation == False:
            # TODO: Data Augmentation
            pass

        # Return the augmented batch_dict
        return batch_dict
