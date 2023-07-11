# %%
from omegaconf import DictConfig
import hydra, torch, numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from cellot.models.cond_score_module import CondScoreModule, CondScoreModuleV2
from cellot.data.sciplex_ae_dm import CellDataModule
from cellot.data.utils import load_ae_cell_data, load_ae, cast_dataset_to_loader
from cellot.train.utils import get_free_gpu
from cellot.utils.dev_utils import load_markers, compute_mmd_loss, get_ckpt_path_from_artifact_id
from tqdm import tqdm

gammas = np.logspace(1, -3, num=50)

def inference(lm, batch, lamb=4, dt=0.01, t_start=1.0, cond=True, ae=None):
    device = lm.device
    assert ae is not None, 'Must provide autoencoder'
    
    with torch.inference_mode():
        lm.eval()
        all_genes_x, y = batch
        latent_x = ae.eval().encode(all_genes_x)
        
        x_t, _ = lm.diffuser.forward_marginal(latent_x.detach().cpu().numpy(), t=t_start)
        
        for i, t in enumerate(np.arange(t_start, 0, -dt)):
            x_t = torch.tensor(x_t).float().to(lm.device)
            uncond_score = lm.score_network((x_t, (torch.ones_like(y) * lm.score_network.null_cond_idx).to(device)), t)
            if cond:
                cond_score = lm.score_network((x_t, y), t)
                pred_score = (1 + lamb) * cond_score - lamb * uncond_score
            else:
                pred_score = uncond_score
            
            x_t = lm.diffuser.reverse(x_t=x_t.detach().cpu().numpy(), score_t=pred_score.detach().cpu().numpy(), t=t, dt=lm.dt, center=False)
        
        x_0 = torch.tensor(x_t, dtype=torch.float).to(lm.device)
        
        recon = ae.eval().decode(x_0)
        return recon

@hydra.main(config_path="../configs/diff", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    replica_id = int(get_free_gpu())
    device = f'cuda:{replica_id}'
    
    if cfg.MODEL_CLASS == 'CondScoreModule':
        model_class = CondScoreModule
    elif cfg.MODEL_CLASS == 'CondScoreModuleV2':
        model_class = CondScoreModuleV2
    
    if cfg.WARM_START:
        model = model_class.load_from_checkpoint(checkpoint_path=cfg.WARM_START_PATH, hparams=cfg)
        cfg.experiment.wandb_logger.name = cfg.experiment.wandb_logger.name + '_WS'
    
    else:
        model = model_class(cfg)
    model = model.to(device)
    autoencoder = load_ae(cfg, device='cpu', restore=cfg.AE_PATH, input_dim=1000).to(device)

    if cfg.TARGET != 'all':
        n_markers = cfg.N_MARKERS
        sel_mg, gene_idxs = load_markers(cfg)
    else:
        n_markers = 1000
        gene_idxs = np.arange(1000)
    
    datasets = load_ae_cell_data(cfg, return_as='dataset', split_on=["split", "transport"])

    loader = cast_dataset_to_loader(datasets, batch_size=cfg.dataloader.batch_size, shuffle=False, drop_last=False)
    
    source = datasets.test.source.adata.X
    target = datasets.test.target.adata.X
    source[:, gene_idxs[:n_markers]].shape
    target[:, gene_idxs[:n_markers]].shape

    gts = []
    recons = []
    mmds = []
    
    lamb = cfg.infer.lamb
    dt = cfg.infer.dt
    t_start = cfg.infer.t_start
    cond = cfg.infer.cond
    sel_target = target[:, gene_idxs[:n_markers]]

    for batch in tqdm(loader.test.source):
        batch = [x.to(device) for x in batch]
        gts.append(batch)
        recon = inference(model, batch, ae=autoencoder, lamb=lamb, dt=dt, t_start=t_start, cond=cond)
        recons.append(recon)
        mmds.append(compute_mmd_loss(recon[:, gene_idxs[:n_markers]].detach().cpu().numpy(), sel_target, gammas))

    all_recon = torch.cat(recons, dim=0)
    all_recon.shape
    
    mmd_agg = compute_mmd_loss(all_recon[:, gene_idxs[:n_markers]].detach().cpu().numpy(), sel_target, gammas)
    mean_mmd = np.mean(mmds)
    
    print(f'MMD: {mmd_agg:.4f}, Mean MMD: {mean_mmd:.4f}') #TODO: Set up basic logging here.
    
if __name__ == "__main__":
    main()