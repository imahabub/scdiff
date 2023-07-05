from omegaconf import DictConfig
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from cellot.models.cond_score_module import CondScoreModule
from cellot.data.sciplex_ae_dm import CellDataModule
from cellot.train.utils import get_free_gpu

@hydra.main(config_path="../../configs/diff", config_name="base.yaml")
def main(cfg: DictConfig) -> None:
    replica_id = int(get_free_gpu())
    model = CondScoreModule(cfg)
    data_module = CellDataModule(cfg)

    trainer_devices = [replica_id] if cfg.experiment.dist_mode == 'single' else cfg.DEVICES

    logger = WandbLogger(save_dir='wandb_logs',
                         **cfg.experiment.wandb_logger)
    trainer = Trainer(logger=logger, devices=trainer_devices, **cfg.trainer)

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()