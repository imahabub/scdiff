# %%
import torch

# %%
import wandb
run = wandb.init()
artifact = run.use_artifact('protein-optimization/sc_diff/model-ylwt16su:v59', type='model')
artifact_dir = artifact.download()

# %%
artifact_dir

# %%
from cellot.models.cond_score_module import CondScoreModuleV2

# %%
ckpt_path = f'{artifact_dir}/model.ckpt'

# %%
YAML_STR = '''
DEBUG: False
TARGET: abexinostat
LATENT_DIM: 50
COND_CLASSES: 190
SEED: 42
AE_PATH: /Mounts/rbg-storage1/users/johnyang/cellot/results/sciplex3/full_ae
VAL_SIZE: 0.1
DEVICES: 1
WARM_START: False
WARM_START_PATH: null
MODEL_CLASS: CondScoreModule


diffuser:
  min_b: 0.01
  max_b: 1.0
  schedule: exponential
  score_scaling: var
  coordinate_scaling: 1.0
  latent_dim: ${LATENT_DIM}
  dt: 0.01
  min_t: 0

ae:
  name: scgen
  beta: 0.0
  dropout: 0.0
  hidden_units: [512, 512]
  latent_dim: 50

score_network:
  latent_dim: ${LATENT_DIM}
  cond_classes: ${COND_CLASSES}
  model_dim: 256   # Adjusted to 64
  n_layers: 6    # Adjusted to 12
  nhead: 8
  dim_feedforward: 2048
  dropout: 0.1
  ffn_hidden_dim: 1024
  extra_null_cond_embedding: False


data:
  type: cell
  source: control
  condition: drug
  path: /Mounts/rbg-storage1/users/johnyang/cellot/datasets/scrna-sciplex3/hvg.h5ad
  target: ${TARGET}

datasplit:
  groupby: drug   
  name: train_test
  test_size: 0.2
  random_state: 0
  
dataloader:
  batch_size: 256   # Adjusted to 256
  shuffle: true
  num_workers: 80
  
experiment:
  name: base
  mode: train
  num_loader_workers: 0
  port: 12319
  dist_mode: single
  use_wandb: True
  ckpt_path: null
  wandb_logger:
    project: sc_diff
    name: ${experiment.name}
    dir: /Mounts/rbg-storage1/users/johnyang/cellot/
    log_model: all
    tags: ['experimental']
  lr: 0.0001


trainer:
  accelerator: 'gpu'
  check_val_every_n_epoch: 50
  log_every_n_steps: 100
  num_sanity_val_steps: 1
  enable_progress_bar: True
  enable_checkpointing: True
  fast_dev_run: False
  profiler: simple
  max_epochs: 10000
  strategy: auto
  enable_model_summary: True
  overfit_batches: 0.0
  limit_train_batches: 1.0
  limit_val_batches: 1.0
  limit_predict_batches: 1.0
'''

# %%
from omegaconf import OmegaConf
config = OmegaConf.create(YAML_STR)

# %%
ckpt_path = f'{artifact_dir}/model.ckpt'

# %%
from cellot.train.utils import get_free_gpu
replica_id = int(get_free_gpu())

# %%
device = f'cuda:{replica_id}'

# %%
config

# %%
lm = CondScoreModuleV2.load_from_checkpoint(hparams=config, checkpoint_path=ckpt_path).to(device)
print('')

# %%
# %%
import cellot.models
from cellot.data.cell import load_cell_data
import torch
from cellot.models.ae import AutoEncoder
from pathlib import Path

def load_data(config, **kwargs):
    data_type = config.get("data.type", "cell")
    if data_type in ["cell", "cell-merged", "tupro-cohort"]:
        loadfxn = load_cell_data

    elif data_type == "toy":
        loadfxn = load_toy_data

    else:
        raise ValueError

    return loadfxn(config, **kwargs)


def load_model(config, device, restore=None, **kwargs):
    # def load_autoencoder_model(config, restore=None, **kwargs):
    
    def load_optimizer(config, params):
        kwargs = dict(config.get("optim", {}))
        assert kwargs.pop("optimizer", "Adam") == "Adam"
        optim = torch.optim.Adam(params, **kwargs)
        return optim


    def load_networks(config, **kwargs):
        kwargs = kwargs.copy()
        kwargs.update(dict(config.get("ae", {})))
        name = kwargs.pop("name")

        # if name == "scgen":
        model = AutoEncoder

        # elif name == "cae":
        #     model = ConditionalAutoEncoder
        # else:
        #     raise ValueError

        return model(**kwargs)
    
    model = load_networks(config, **kwargs)
    optim = load_optimizer(config, model.parameters())

    if restore is not None and Path(restore).exists():
        print('Loading model from checkpoint')
        ckpt = torch.load(restore, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optim_state"])
        # if config.model.name == "scgen" and "code_means" in ckpt:
        #     model.code_means = ckpt["code_means"]
            
    # logger.info(f'Model on device {next(model.parameters()).device}')

    return model, optim

def load(config, device, restore=None, include_model_kwargs=False, **kwargs):

    loader, model_kwargs = load_data(config, include_model_kwargs=True, **kwargs)

    model, opt = load_model(config, device, restore=restore, **model_kwargs)

    return model, opt, loader
# %% [markdown]
# ### Training

# %%
restore_path = '/Mounts/rbg-storage1/users/johnyang/cellot/saved_weights/ae/ae.pt'
ae = load_model(config, 'cuda', restore=restore_path, input_dim=1000)

# %%
ae[0].decode

# %%
import numpy as np

# %%
autoencoder = ae[0].to(device)

# %%
def inference(lm, batch, lamb=4, dt=0.01, t_start=1.0, cond=True):
    with torch.inference_mode():
        lm.eval()
        all_genes_x, y = batch
        latent_x = autoencoder.eval().encode(all_genes_x)
        
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
        
        recon = autoencoder.eval().decode(x_0)
        return recon
        
        

# %%
from cellot.data.cell import read_single_anndata
def load_markers():
    data = read_single_anndata(config, path=None)
    key = f'marker_genes-{config.data.condition}-rank'

    # rebuttal preprocessing stored marker genes using
    # a generic marker_genes-condition-rank key
    # instead of e.g. marker_genes-drug-rank
    # let's just patch that here:
    if key not in data.varm:
        key = 'marker_genes-condition-rank'
        print('WARNING: using generic condition marker genes')

    sel_mg = (
        data.varm[key][config.data.target]
        .sort_values()
        .index
    )
    marker_gene_indices = [i for i, gene in enumerate(data.var_names) if gene in sel_mg]

    return sel_mg, marker_gene_indices

sel_mg, gene_idxs = load_markers()
sel_mg

# %%
from cellot.data.utils import *

# %%
def DEV_load_ae_cell_data(
        config,
        data=None,
        split_on=None,
        return_as="loader",
        include_model_kwargs=False,
        pair_batch_on=None,
        ae=None,
        encode_latents=False,
        sel_mg=None,
        **kwargs
    ):
        assert ae is not None or not encode_latents, "ae must be provided"
        
        if isinstance(return_as, str):
            return_as = [return_as]

        assert set(return_as).issubset({"anndata", "dataset", "loader"})
        config.data.condition = config.data.get("condition", "drug")
        condition = config.data.condition
        
        data = read_single_anndata(config, **kwargs)
        
        inputs = torch.Tensor(
            data.X if not sparse.issparse(data.X) else data.X.todense()
        )

        if encode_latents:
            genes = data.var_names.to_list()
            data = anndata.AnnData(
                ae.eval().encode(inputs).detach().numpy(),
                obs=data.obs.copy(),
                uns=data.uns.copy(),
            )
            data.uns["genes"] = genes


        # cast to dense and check for nans
        if sparse.issparse(data.X):
            data.X = data.X.todense()
        assert not np.isnan(data.X).any()

        if sel_mg is not None:
            data = data[:, sel_mg]

        dataset_args = dict()
        model_kwargs = {}

        model_kwargs["input_dim"] = data.n_vars

        # if config.get("model.name") == "cae":
        condition_labels = sorted(data.obs[condition].cat.categories)
        model_kwargs["conditions"] = condition_labels
        dataset_args["obs"] = condition
        dataset_args["categories"] = condition_labels

        if "training" in config:
            pair_batch_on = config.training.get("pair_batch_on", pair_batch_on)

        # if split_on is None:
            # if config.model.name == "cellot":
            #     # datasets & dataloaders accessed as loader.train.source
        split_on = ["split", "transport"]
        if pair_batch_on is not None:
            split_on.append(pair_batch_on)

            # if (config.ae.name == "scgen" #or config.ae.name == "cae"
            #     #or config.ae.name == "popalign"):
            # split_on = ["split"]

            # else:
            #     raise ValueError

        if isinstance(split_on, str):
            split_on = [split_on]

        for key in split_on:
            assert key in data.obs.columns

        if len(split_on) > 0:
            splits = {
                (key if isinstance(key, str) else ".".join(key)): data[index]
                for key, index in data.obs[split_on].groupby(split_on).groups.items()
            }

            dataset = nest_dict(
                {
                    key: AnnDataDataset(val.copy(), **dataset_args)
                    for key, val in splits.items()
                },
                as_dot_dict=True,
            )
        else:
            dataset = AnnDataDataset(data.copy(), **dataset_args)

        if "loader" in return_as:
            kwargs = dict(config.dataloader)
            kwargs.setdefault("drop_last", True)
            loader = cast_dataset_to_loader(dataset, **kwargs)

        returns = list()
        for key in return_as:
            if key == "anndata":
                returns.append(data)

            elif key == "dataset":
                returns.append(dataset)

            elif key == "loader":
                returns.append(loader)

        if include_model_kwargs:
            returns.append(model_kwargs)

        if len(returns) == 1:
            return returns[0]

        # returns.append(data)

        return tuple(returns)

# %%
datasets = DEV_load_ae_cell_data(config, return_as='dataset')#, ae=autoencoder.cpu(), encode_latents=True)#, sel_mg=sel_mg)

# %%
loader = cast_dataset_to_loader(datasets, batch_size=256, shuffle=False, drop_last=False)
loader

# %%
source = datasets.test.source.adata.X

# %%
target = datasets.test.target.adata.X

# %%
source[:, gene_idxs[:50]].shape

# %%
target[:, gene_idxs[:50]].shape

# %%
from tqdm import tqdm

# %%
from cellot.losses.mmd import mmd_distance
import numpy as np

def compute_mmd_loss(lhs, rhs, gammas):
    return np.mean([mmd_distance(lhs, rhs, g) for g in gammas])

gammas = np.logspace(1, -3, num=50)

# %%
sel_target = target[:, gene_idxs[:50]]

# %%
gts = []
recons = []
for batch in tqdm(loader.test.source):
    batch = [x.to(device) for x in batch]
    gts.append(batch)
    recon = inference(lm, batch, lamb=4, dt=0.01, t_start=1.0, cond=True)
    recons.append(recon)

# %%
torch.where(gts[0][0][0, gene_idxs[:50]] > 0, 1, 0)

# %%
torch.where(recons[0][0, gene_idxs[:50]] > 0.1, 1, 0)

# %%
all_recon = torch.cat(recons, dim=0)
all_recon.shape

# %%
compute_mmd_loss(all_recon[:, gene_idxs[:50]].detach().cpu().numpy(), sel_target, gammas)

# %%
compute_mmd_loss(all_recon[:, gene_idxs[:50]].detach().cpu().numpy(), sel_target, gammas)

# %%
from cellot import losses
losses.compute_scalar_mmd(all_recon[:, gene_idxs[:50]].detach().cpu().numpy(), sel_target)

# %% [markdown]
# 


