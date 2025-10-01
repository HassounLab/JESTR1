from torch.utils.data.dataloader import DataLoader
from massspecgym.data.data_module import MassSpecDataModule
from jestr.data.datasets import ContrastiveDataset
from functools import partial
from massspecgym.models.base import Stage

class TestDataModule(MassSpecDataModule):
    def __init__(
            self,
            collate_fn,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.collate_fn = collate_fn

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        if stage == "test":
            self.test_dataset = self.dataset
        else:
            raise Exception("Data module supports test set only")

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        return None
    
    def val_dataset(self):
        return None

class ContrastiveDataModule(MassSpecDataModule):
    def __init__(
            self,
            collate_fn,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.collate_fn = collate_fn
        self.regularization_flag = False
             
    def train_dataloader(self):
        self.train_contrastive_dataset = ContrastiveDataset(self.train_dataset)

        return DataLoader(self.train_contrastive_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers,
                          drop_last=False,
                          collate_fn=partial(self.collate_fn, stage=Stage.TRAIN),
                          )

    def val_dataloader(self):
        self.val_contrastive_dataset = ContrastiveDataset(self.val_dataset)

        return DataLoader(self.val_contrastive_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers,
                          drop_last=False,
                          collate_fn=partial(self.collate_fn, stage=Stage.VAL))

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
        )
