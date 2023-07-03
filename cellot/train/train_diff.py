from omegaconf import DictConfig
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from cellot.models.cond_score_module import CondScoreModule
from cellot.data.sciplex_ae_dm import CellDataModule

@hydra.main(config_path="../../configs/diff", config_name="base.yaml")
def main(cfg: DictConfig) -> None:
    model = CondScoreModule(cfg)
    data_module = CellDataModule(cfg)

    logger = TensorBoardLogger(save_dir='tb_logs')
    trainer = Trainer(logger=logger, **cfg.trainer)

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()