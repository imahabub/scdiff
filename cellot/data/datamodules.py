import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cellot.data.cell import load_cell_data, AnnDataDataset
from pathlib import Path
import torch
from cellot.data.utils import load_ae_cell_data, load_ae
import scanpy as sc

class GenericDataModule(pl.LightningDataModule):
    
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.kwargs = kwargs
        self.seed = config.SEED
        self.val_fract = config.VAL_FRACT
        self.test_fract = config.TEST_FRACT
        
    def setup(self, stage='fit'):
        raise NotImplementedError
    
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

class CellDataModule(GenericDataModule):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        
    def get_ae(self, path):
        ae = load_ae(self.config, device='cpu', restore=path, input_dim=1000)
        return ae

    def prepare_data(self):
        # If your datasets are downloadable you would write code here to download them
        pass

    def setup(self, stage=None):
        # Load data
        torch.manual_seed(self.seed)

        if self.config.AE_PATH is not None:
            ae = self.get_ae(self.config.AE_PATH)
            self.data = load_ae_cell_data(self.config, return_as="dataset", ae=ae, encode_latents=True, split_on=['split'])
        else:
            self.data = load_ae_cell_data(self.config, return_as="dataset", split_on=['split'])

        # Create datasets
        full_train_dataset = self.data.train
        full_train_size = len(full_train_dataset)
        val_size = int(self.val_fract * full_train_size)
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

class scPerturbDataModule(GenericDataModule):
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.pca = config.data.pca
        
    def prepare_data(self):
        # If your datasets are downloadable you would write code here to download them
        # self.adata = sc.read_h5ad(self.config.data.path)
        # if self.pca:
        #     sc.tl.pca(self.adata, svd_solver='arpack')
        ...
        
    def setup(self, stage='fit'):
        torch.manual_seed(self.seed)
        
        self.adata = sc.read_h5ad(self.config.data.path)
        if self.pca:
            sc.tl.pca(self.adata, svd_solver='arpack')
        
        condition = 'perturbation'
        self.data = AnnDataDataset(adata=self.adata, obs=condition, categories=sorted(self.adata.obs[condition].cat.categories), pca=self.pca)

        test_size = int(self.test_fract * len(self.data))
        val_size = int(self.val_fract * len(self.data))
        train_size = len(self.data) - val_size - test_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.data, [train_size, val_size, test_size])

    