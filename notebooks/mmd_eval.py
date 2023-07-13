# %%
from omegaconf import DictConfig
import hydra
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from cellot.models.cond_score_module import CondScoreModule, CondScoreModuleV2
from cellot.data.sciplex_ae_dm import CellDataModule
from cellot.data.utils import load_ae_cell_data, load_ae, cast_dataset_to_loader
from cellot.train.utils import get_free_gpu
from cellot.utils.dev_utils import compute_mmd_loss, get_ckpt_path_from_artifact_id, get_target_cond_idx
from tqdm import tqdm

from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime
import os

gammas = np.logspace(1, -3, num=50)

def inference(lm, batch, lamb=4, dt=0.01, t_start=1.0, cond=True, ae=None, target=None):
    device = lm.device
    assert ae is not None, 'Must provide autoencoder'
    assert target is not None or not cond, 'Must provide target'
    
    with torch.inference_mode():
        lm.eval()
        all_genes_x, y_batch = batch
        
        if cond:
            y = torch.ones_like(y_batch) * get_target_cond_idx(target)
        else:
            y = y_batch
        
        y = y.to(device)
        
        latent_x = ae.eval().encode(all_genes_x)
        latent_iden_recon = ae.eval().decode(latent_x)

        x_t, _ = lm.diffuser.forward_marginal(latent_x.detach().cpu().numpy(), t=t_start)
        
        for i, t in enumerate(np.arange(t_start, 0, -dt)):
            x_t = torch.tensor(x_t).float().to(device)
            uncond_score = lm.score_network((x_t, (torch.ones_like(y) * lm.score_network.null_cond_idx).to(device)), t)
            if cond:
                cond_score = lm.score_network((x_t, y), t)
                pred_score = (1 + lamb) * cond_score - lamb * uncond_score
            else:
                pred_score = uncond_score
            
            x_t = lm.diffuser.reverse(x_t=x_t.detach().cpu().numpy(), score_t=pred_score.detach().cpu().numpy(), t=t, dt=lm.dt, center=False)
        
        x_0 = torch.tensor(x_t, dtype=torch.float).to(lm.device)
        
        recon = ae.eval().decode(x_0)
        return recon, latent_x, x_0, latent_iden_recon
    
from cellot.data.cell import read_single_anndata
def load_markers(config, n_genes=50, gene_pool=None):
    data = read_single_anndata(config, path=None)
    key = f'marker_genes-{config.data.condition}-rank'

    # rebuttal preprocessing stored marker genes using
    # a generic marker_genes-condition-rank key
    # instead of e.g. marker_genes-drug-rank
    # let's just patch that here:
    if key not in data.varm:
        key = 'marker_genes-condition-rank'
        print('WARNING: using generic condition marker genes')
        
    # Filter marker genes using gene_pool before sorting and selecting top genes
    if gene_pool is not None:
        # Make sure gene_pool is a set for the intersection operation
        gene_pool = set(gene_pool)
        potential_mgs = set(data.varm[key].index)
        valid_genes = potential_mgs.intersection(gene_pool)
    else:
        valid_genes = data.varm[key].index

    sel_mg = (
        data.varm[key].loc[valid_genes][config.data.target]
        .sort_values()
        .index
    )[:n_genes]
    
    marker_gene_indices = [i for i, gene in enumerate(data.var_names) if gene in sel_mg]
    marker_gene_names = [gene for gene in data.var_names if gene in sel_mg]

    return sel_mg, marker_gene_indices, marker_gene_names

def save_df(df, directory, name):

    # Get the current date and time
    now = datetime.now()

    # Format it as a string. This will give you "YYYY-MM-DD_HH-MM-SS"
    formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create your filename. I'm assuming that cfg.eval_name is a string.
    filename = f"{name}_{formatted_now}.csv"

    # Create the full file path
    full_path = os.path.join(directory, filename)

    # Save the DataFrame to a csv file
    df.to_csv(full_path, index=False)

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
    
    gts = []
    recons = []
    uncond_recons = []
    lxs = []
    target_lxs = []
    x_0s = []
    latent_identities = []
    uncond_x_0s = []

    
    lamb = cfg.infer.lamb
    dt = cfg.infer.dt
    t_start = cfg.infer.t_start
    cond = cfg.infer.cond

    for batch in tqdm(loader.test.source):
        batch = [x.to(device) for x in batch]
        gts.append(batch)
        recon, latent_x, x_0, latent_iden_recon = inference(model, batch, ae=autoencoder, lamb=lamb, dt=dt, t_start=t_start, cond=cond, target=cfg.TARGET)
        uncond_recon, _, uncond_x_0, _ = inference(model, batch, ae=autoencoder, lamb=lamb, dt=dt, t_start=t_start, cond=False, target=cfg.TARGET)
        recons.append(recon)
        uncond_recons.append(uncond_recon)
        lxs.append(latent_x)
        x_0s.append(x_0)
        latent_identities.append(latent_iden_recon)
        uncond_x_0s.append(uncond_x_0)
        
    for batch in tqdm(loader.test.target):
        autoencoder.eval()
        batch = [x.to(device) for x in batch]
        target_lx = autoencoder.encode(batch[0])
        target_lxs.append(target_lx)

    all_gts = torch.cat([x[0] for x in gts], dim=0)
    all_recon = torch.cat(recons, dim=0)
    all_uncond_recons = torch.cat(uncond_recons, dim=0)
    all_lxs = torch.cat(lxs, dim=0)
    all_target_lxs = torch.cat(target_lxs, dim=0)
    all_x0s = torch.cat(x_0s, dim=0)    
    all_latent_identities = torch.cat(latent_identities, dim=0)

    sel_target = target[:, gene_idxs[:n_markers]]
    sel_recon = all_recon[:, gene_idxs[:n_markers]].detach().cpu().numpy()
    mmd_agg = compute_mmd_loss(sel_recon, sel_target, gammas)
    
    all_uncond_recon = torch.cat(uncond_recons, dim=0)
    sel_uncond_recon = all_uncond_recon[:, gene_idxs[:n_markers]].detach().cpu().numpy()
    uncond_mmd_agg = compute_mmd_loss(sel_uncond_recon, sel_target, gammas)
    
    # # mean_mmd = np.mean(mmds)
    # print(f'MMD: {mmd_agg:.4f}') #TODO: Set up basic logging here.
    # print(f'Uncond MMD: {uncond_mmd_agg:.4f}')
    # # print(f'MMD: {mmd_agg:.4f}, Mean MMD: {mean_mmd:.4f}') #TODO: Set up basic logging here.
    
    # Compute Euclidean distances
    dist_source_conditional = distance.cdist(all_source_lxs.detach().cpu().numpy(), all_x0s.detach().cpu().numpy(), 'euclidean')
    dist_source_unconditional = distance.cdist(all_source_lxs.detach().cpu().numpy(), all_uncond_x0s.detach().cpu().numpy(), 'euclidean')

    dist_target_conditional = distance.cdist(all_target_lxs.detach().cpu().numpy(), all_x0s.detach().cpu().numpy(), 'euclidean')
    dist_target_unconditional = distance.cdist(all_target_lxs.detach().cpu().numpy(), all_uncond_x0s.detach().cpu().numpy(), 'euclidean')

    # Compute Cosine similarities
    similarity_source_conditional = cosine_similarity(all_source_lxs.detach().cpu().numpy(), all_x0s.detach().cpu().numpy())
    similarity_source_unconditional = cosine_similarity(all_source_lxs.detach().cpu().numpy(), all_uncond_x0s.detach().cpu().numpy())

    similarity_target_conditional = cosine_similarity(all_target_lxs.detach().cpu().numpy(), all_x0s.detach().cpu().numpy())
    similarity_target_unconditional = cosine_similarity(all_target_lxs.detach().cpu().numpy(), all_uncond_x0s.detach().cpu().numpy())

    source_target_cdist = distance.cdist(all_source_lxs.detach().cpu().numpy(), all_target_lxs.detach().cpu().numpy(), 'euclidean')
    source_target_cdist.mean()

    source_target = cosine_similarity(all_source_lxs.detach().cpu().numpy(), all_target_lxs.detach().cpu().numpy())
    
    # Create a dictionary to hold all the data
    data = {
        'dist_source_conditional': dist_source_conditional.mean(),
        'dist_source_unconditional': dist_source_unconditional.mean(),
        'dist_target_conditional': dist_target_conditional.mean(),
        'dist_target_unconditional': dist_target_unconditional.mean(),
        'similarity_source_conditional': similarity_source_conditional.mean(),
        'similarity_source_unconditional': similarity_source_unconditional.mean(),
        'similarity_target_conditional': similarity_target_conditional.mean(),
        'similarity_target_unconditional': similarity_target_unconditional.mean(),
        'source_target_cdist': source_target_cdist.mean(),
        'source_target': source_target.mean(),
        'MMD': mmd_agg,
        'Uncond MMD': uncond_mmd_agg
    }

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(data, index=[0])
    save_df(df, cfg.directory, cfg.eval_name)
    
    
if __name__ == "__main__":
    main()