from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from hydra import compose
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from cellot.models.cond_score_module import CondScoreModule, Pred_X_0_Parameterization
from cellot.data.datamodules import GenericDataModule, CellDataModule, scPerturbDataModule
from cellot.train.utils import get_free_gpu
from utils import get_ckpt_path_from_run_id
import os, re, argparse

def main(cfg: DictConfig) -> None:
    # Prepare data
    if cfg.data.type == 'cell':
        data_module = CellDataModule(cfg)
    elif cfg.data.type == 'scPerturb':
        data_module = scPerturbDataModule(cfg)
    else:
        raise ValueError(f'Unknown data type {cfg.data.type}')
    
    # Train model
    train_model(cfg, data_module)
    
    
def init_model(cfg):
    
    if cfg.MODEL_CLASS == 'CondScoreModule':
        model_class = CondScoreModule
    elif cfg.MODEL_CLASS == 'Pred_X_0_Parameterization':
        model_class = Pred_X_0_Parameterization
    else:
        raise ValueError(f'Unknown model class {cfg.MODEL_CLASS}')
    
    # Load or initialize the model
    if cfg.WARM_START:
        ckpt_path = cfg.WARM_START_PATH if cfg.WANDB_RUN_ID is None else get_ckpt_path_from_run_id(cfg.WANDB_RUN_ID)
        model = model_class.load_from_checkpoint(checkpoint_path=ckpt_path, hparams=cfg)
        cfg.experiment.wandb_logger.name = cfg.experiment.wandb_logger.name + '_WS'
    else:
        model = model_class(cfg)
    
    return model

def train_model(cfg: DictConfig, data_module: GenericDataModule):
    
    model = init_model(cfg)

    # Define device strategy
    if cfg.trainer.strategy == 'auto':
        replica_id = int(get_free_gpu())
        trainer_devices = [replica_id]
    else: 
        trainer_devices = cfg.DEVICES
    
    # Set debug mode if required
    if cfg.DEBUG:
        os.environ["WANDB_MODE"] = "dryrun"

    # Initialize logger
    logger = WandbLogger(**cfg.experiment.wandb_logger)

    # Initialize and run the trainer
    trainer = Trainer(logger=logger, devices=trainer_devices, **cfg.trainer)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    # Register the config
    # cs = ConfigStore.instance()
    # cs.store(name="base", node=DictConfig) # Define your config schema here

    parser = argparse.ArgumentParser()
    parser.add_argument('--cn', type=str, default='base', help='Configuration name')
    parser.add_argument('--ao', nargs='*', default=[], help='Additional overrides')
    args = parser.parse_args()

    # Initialize Hydra
    with hydra.initialize(config_path="../../configs/diff"):
        # Get the config object
        cfg = compose(config_name=args.cn, overrides=args.ao)
        main(cfg)