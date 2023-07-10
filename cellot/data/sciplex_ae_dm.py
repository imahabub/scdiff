import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cellot.data.cell import load_cell_data, AnnDataDataset
from pathlib import Path
import torch
from cellot.data.utils import load_ae_cell_data, load_ae

class CellDataModule(pl.LightningDataModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.kwargs = kwargs
        self.seed = config.SEED
        self.val_size = config.VAL_SIZE

        
    def get_ae(self, path):
        
        ae = load_ae(self.config, device='cpu', restore=path, input_dim=1000)
        return ae

    def prepare_data(self):
        # If your datasets are downloadable you would write code here to download them
        pass

    def setup(self, stage=None):
        # Load data
        torch.manual_seed(self.seed)

        ae = self.get_ae(self.config.AE_PATH)
        self.data = load_ae_cell_data(self.config, return_as="dataset", ae=ae, encode_latents=True)

        # Create datasets
        full_train_dataset = self.data.train
        full_train_size = len(full_train_dataset)
        val_size = int(self.val_size * full_train_size)
        train_size = full_train_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
        self.test_dataset = self.data.test # assuming 'test' key is available

    def train_dataloader(self):
        # DataLoader for the training set
        return DataLoader(self.train_dataset, 
                          batch_size=self.config.dataloader.batch_size, 
                          shuffle=True, 
                          num_workers=self.config.dataloader.num_workers
                        )

    def val_dataloader(self):
        # DataLoader for the validation set
        return DataLoader(self.val_dataset, batch_size=self.config.dataloader.batch_size, 
                                  num_workers=self.config.dataloader.num_workers)

    def test_dataloader(self):
        # DataLoader for the test set
        return DataLoader(self.test_dataset, batch_size=self.config.dataloader.batch_size, 
                                  num_workers=self.config.dataloader.num_workers)
