# %%
# %%
from pathlib import Path

import torch
import numpy as np
import random
import pickle
from absl import logging
from absl.flags import FLAGS
from cellot import losses
from cellot.utils.loaders import load
from cellot.models.cellot import compute_loss_f, compute_loss_g, compute_w2_distance
from cellot.train.summary import Logger
from cellot.data.utils import cast_loader_to_iterator
from cellot.models.ae import compute_scgen_shift
from tqdm import trange

from cellot.models.ae import AutoEncoder

import omegaconf
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("data_logger")
logger.setLevel(logging.INFO)

# %%
DEBUG = False
TARGET = 'all' if not DEBUG else 'abexinostat'
LATENT_DIM = 50
COND_CLASSES = 189 if TARGET == 'all' else 2

from pathlib import Path
outdir_path = '/Mounts/rbg-storage1/users/johnyang/cellot/results/sciplex3/full_ae'
outdir = Path(outdir_path)

# %%
outdir.mkdir(exist_ok=True, parents=True)

cachedir = outdir / "cache"
cachedir.mkdir(exist_ok=True)

# %%

import torch
import GPUtil
import os

def get_free_gpu():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # Set environment variables for which GPUs to use.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    chosen_gpu = ''.join(
        [str(x) for x in GPUtil.getAvailable(order='memory')])
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen_gpu
    print(f"Using GPUs: {chosen_gpu}")
    return chosen_gpu

status = cachedir / "status"
status.write_text("running")

device = f'cuda:{get_free_gpu()}'


if DEBUG:
    n_iters = 250000
    batch_size = 256
else:
    n_iters = 250000
    batch_size = 256

yaml_str = f"""
model:
   name: scgen
   beta: 0.0
   dropout: 0.0
   hidden_units: [512, 512]
   latent_dim: 50

optim:
   lr: 0.001
   optimizer: Adam
   weight_decay: 1.0e-05

scheduler:
   gamma: 0.5
   step_size: 100000

training:
  cache_freq: 10000
  eval_freq: 2500
  logs_freq: 250
  n_iters: {n_iters}

data:
  type: cell
  source: control
  condition: drug
  path: /Mounts/rbg-storage1/users/johnyang/cellot/datasets/scrna-sciplex3/hvg.h5ad
  target: {TARGET}

datasplit:
    groupby: drug   
    name: train_test
    test_size: 0.2
    random_state: 0

dataloader:
    batch_size: {batch_size}
    shuffle: true
"""

config = omegaconf.OmegaConf.create(yaml_str)

# %%
import cellot.models
from cellot.data.cell import load_cell_data

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
        kwargs.update(dict(config.get("model", {})))
        name = kwargs.pop("name")

        if name == "scgen":
            model = AutoEncoder

        # elif name == "cae":
        #     model = ConditionalAutoEncoder
        else:
            raise ValueError

        return model(**kwargs)
    
    model = load_networks(config, **kwargs)
    optim = load_optimizer(config, model.parameters())

    if restore is not None and Path(restore).exists():
        print('Loading model from checkpoint')
        ckpt = torch.load(restore, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optim_state"])
        if config.model.name == "scgen" and "code_means" in ckpt:
            model.code_means = ckpt["code_means"]
            
    # logger.info(f'Model on device {next(model.parameters()).device}')

    return model, optim

def load(config, device, restore=None, include_model_kwargs=False, **kwargs):

    loader, model_kwargs = load_data(config, include_model_kwargs=True, **kwargs)

    model, opt = load_model(config, device, restore=restore, **model_kwargs)

    return model, opt, loader
# %% [markdown]
# ### Training

# %%
ae = load_model(config, 'cuda', restore=cachedir / "last.pt", input_dim=1000)

from cellot.data.cell import *

def load_cell_data(
    config,
    data=None,
    split_on=None,
    return_as="loader",
    include_model_kwargs=False,
    pair_batch_on=None,
    **kwargs
):

    if isinstance(return_as, str):
        return_as = [return_as]

    assert set(return_as).issubset({"anndata", "dataset", "loader"})
    config.data.condition = config.data.get("condition", "drug")
    condition = config.data.condition
    
    data = read_single_anndata(config, **kwargs)

    # if "ae_emb" in config.data:
        # load path to autoencoder
        # assert config.get("model.name", "cellot") == "cellot"
    # path_ae = Path(outdir_path)
    # model_kwargs = {"input_dim": data.n_vars}
    # config_ae = load_config('/Mounts/rbg-storage1/users/johnyang/cellot/configs/models/scgen.yaml')
    # ae_model, _ = load_autoencoder_model(
    #     config_ae, restore=path_ae / "cache/model.pt", **model_kwargs
    # )

    inputs = torch.Tensor(
        data.X if not sparse.issparse(data.X) else data.X.todense()
    )

    genes = data.var_names.to_list()
    data = anndata.AnnData(
        ae[0].eval().encode(inputs).detach().numpy(),
        obs=data.obs.copy(),
        uns=data.uns.copy(),
    )
    data.uns["genes"] = genes

    # cast to dense and check for nans
    if sparse.issparse(data.X):
        data.X = data.X.todense()
    assert not np.isnan(data.X).any()

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

    if split_on is None:
        if config.model.name == "cellot":
            # datasets & dataloaders accessed as loader.train.source
            split_on = ["split", "transport"]
            if pair_batch_on is not None:
                split_on.append(pair_batch_on)

        elif (config.model.name == "scgen" or config.model.name == "cae"
              or config.model.name == "popalign"):
            split_on = ["split"]

        else:
            raise ValueError

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

cond_datasets = load_cell_data(config, return_as="loader")

# %%
"""R^3 diffusion methods."""
import numpy as np
from scipy.special import gamma
import torch


class R3Diffuser:
    """VP-SDE diffuser class for translations."""

    def __init__(self, r3_conf):
        """
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self._r3_conf = r3_conf
        self.min_b = r3_conf.min_b
        self.max_b = r3_conf.max_b
        self.schedule = r3_conf.schedule
        self._score_scaling = r3_conf.score_scaling
        self.latent_dim = r3_conf.latent_dim

    def _scale(self, x):
        return x * self._r3_conf.coordinate_scaling

    def _unscale(self, x):
        return x / self._r3_conf.coordinate_scaling

    def b_t(self, t):
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        if self.schedule == 'linear': 
            return self.min_b + t*(self.max_b - self.min_b)
        elif self.schedule == 'cosine':
            return self.max_b + 0.5*(self.min_b - self.max_b)*(1 + np.cos(t*np.pi))
        elif self.schedule == 'exponential':
            sigma = t * np.log10(self.max_b) + (1 - t) * np.log10(self.min_b)
            return 10 ** sigma
        else:
            raise ValueError(f'Unknown schedule {self.schedule}')
    
    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return np.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1/2 * self.b_t(t) * x

    def sample_ref(self, n_samples: float=1):
        return np.random.normal(size=(n_samples, self.latent_dim))

    def marginal_b_t(self, t):
        if self.schedule == 'linear':
            return t*self.min_b + (1/2)*(t**2)*(self.max_b-self.min_b)
        elif self.schedule == 'exponential': 
            return (self.max_b**t * self.min_b**(1-t) - self.min_b) / (
                np.log(self.max_b) - np.log(self.min_b))
        else:
            raise ValueError(f'Unknown schedule {self.schedule}')

    def calc_trans_0(self, score_t, x_t, t, use_torch=True):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp if use_torch else np.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1/2*beta_t)

    def forward(self, x_t_1: np.ndarray, t: float, num_t: int):
        """Samples marginal p(x(t) | x(t-1)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1]. 

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_t_1 = self._scale(x_t_1)
        b_t = torch.tensor(self.marginal_b_t(t) / num_t).to(x_t_1.device)
        z_t_1 = torch.tensor(np.random.normal(size=x_t_1.shape)).to(x_t_1.device)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t
    
    def distribution(self, x_t, score_t, t, mask, dt):
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        std = g_t * np.sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu *= mask[..., None]
        return mu, std

    def forward_marginal(self, x_0: np.ndarray, t: float):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1]. 

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_0 = self._scale(x_0)
        x_t = np.random.normal(
            loc=np.exp(-1/2*self.marginal_b_t(t)) * x_0,
            scale=np.sqrt(1 - np.exp(-self.marginal_b_t(t)))
        )
        score_t = self.score(x_t, x_0, t)
        x_t = self._unscale(x_t)
        return x_t, score_t

    def score_scaling(self, t: float):
        if self._score_scaling == 'var':
            return 1 / self.conditional_var(t)
        elif self._score_scaling == 'std':
            return 1 / np.sqrt(self.conditional_var(t))
        elif self._score_scaling == 'expected_norm':
            return np.sqrt(2) / (gamma(1.5) * np.sqrt(self.conditional_var(t)))
        else:
            raise ValueError(f'Unrecognized scaling {self._score_scaling}')

    def reverse(
            self,
            *,
            x_t: np.ndarray,
            score_t: np.ndarray,
            t: float,
            dt: float,
            mask: np.ndarray=None,
            center: bool=True,
            ode: bool=False,
            noise_scale: float=1.0,
        ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        if ode:
            # Probability flow ODE
            perturb = (f_t - (1/2)*(g_t**2) * score_t) * dt
        else:
            # Usual stochastic dynamics
            z = noise_scale * np.random.normal(size=score_t.shape)
            perturb = (f_t - g_t**2 * score_t) * dt + g_t * np.sqrt(dt) * z

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = np.ones(x_t.shape[:-1])
        x_t_1 = x_t - perturb
        if center:
            com = np.sum(x_t_1, axis=-2) / np.sum(mask, axis=-1)[..., None]
            x_t_1 -= com[..., None, :]
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def conditional_var(self, t, use_torch=False):
        """Conditional variance of p(xt|x0).

        Var[x_t|x_0] = conditional_var(t)*I

        """
        if use_torch:
            return 1 - torch.exp(-self.marginal_b_t(t))
        return 1 - np.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t, use_torch=False, scale=False):
        if use_torch:
            exp_fn = torch.exp
        else:
            exp_fn = np.exp
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        return -(x_t - exp_fn(-1/2*self.marginal_b_t(t)) * x_0) / self.conditional_var(t, use_torch=use_torch)

# %%
from omegaconf import OmegaConf

r3_conf = OmegaConf.create({
    'min_b': 0.01,
    'max_b': 1.0,
    'schedule': 'linear',
    'score_scaling': 'var',
    'coordinate_scaling': 1.0,
    'latent_dim': LATENT_DIM,
})

# %%
diffuser = R3Diffuser(r3_conf)

# %%
import torch.nn as nn
import torch.nn.functional as F
import math
import functools as fn

# %%
model_dim = 64 #TODO:
num_layers = 2
nhead = 1
dim_feedforward = 128
dropout = 0.1 if not DEBUG else 0.0

class FeedForward(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=50):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class ScoreNetwork(nn.Module):
    def __init__(self):
        super(ScoreNetwork, self).__init__()
        
        self.latent_dim = LATENT_DIM
        self.model_dim = model_dim
        self.cond_classes = COND_CLASSES
        
        self.dropout = dropout
        print(f'Dropout is {self.dropout}')
        
        self.cond_embedding = nn.Embedding(COND_CLASSES, model_dim)
        self.embed_code_and_t = nn.Linear(LATENT_DIM + (2 * model_dim), model_dim)
        # self.trmr_layer = TransformerEncoderLayer(d_model=model_dim, nhead=8, dim_feedforward=2048, dropout=dropout)
        self.pred_score = FeedForward(input_dim=model_dim, hidden_dim=64, output_dim=LATENT_DIM)
        self.model = nn.ModuleList([self.embed_code_and_t, self.pred_score]) #*[self.trmr_layer for _ in range(num_layers)], self.pred_score])
        
        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self.model_dim,
            # max_positions=100
        )

    def forward(self, xy, t):
        x, y = xy
        device = x.device
        B, C = x.shape
        t_embed = torch.tile(self.timestep_embedder(torch.tensor([t]).to(device)), dims=[B, 1])
        y_embed = self.cond_embedding(y)
        x = torch.cat([x, t_embed, y_embed], dim=-1).to(device)
        for module in self.model[:-1]:  # iterate over all modules except the last one
            x = module(x)
        x = self.model[-1](x.squeeze(0))  # pass through the last module (FeedForward)
        return x

# %%
score_network = ScoreNetwork().to(device)

# %%
# sum(p.numel() for p in score_network.parameters())

# %%
optimizer = torch.optim.Adam(score_network.parameters(), lr=1e-4)

# %%

def tb(name):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'runs/{name}')
    return writer

def setup_run():
    # %%
    
    min_t = 0.0
    rng = np.random.default_rng()
    
    STEP = 0
    ticker = trange(STEP, n_iters, initial=STEP, total=n_iters)

    # %%
    iterator = cast_loader_to_iterator(cond_datasets, cycle_all=True)
    return iterator, ticker, rng, min_t


def train():
    iterator, ticker, rng, min_t = setup_run()
    writer = tb('7.3_all')

    def eval(step, dt=0.001):
        score_network.eval()
        mses = []
        with torch.no_grad():
            for xy in iterator.test:
                x, y = xy
                x_t, _ = diffuser.forward_marginal(x.numpy(), t=1.0)
                
                for i, t in enumerate(np.arange(1.0, 0, -dt)):
                    x_t = torch.tensor(x_t).float().to(device)
                    pred_score = score_network((x_t, y.to(device)), t)
                    
                    x_t = diffuser.reverse(x_t=x_t.detach().cpu().numpy(), score_t=pred_score.detach().cpu().numpy(), t=t, dt=dt, center=False)
                
                x_0 = x_t

                mse = torch.mean((x - x_0) ** 2)
                mses.append(mse.item())
            eval_mse = np.mean(mses)
            writer.add_scalar('MSE', eval_mse, global_step=step)
            return eval_mse
    
    eval_freq=1000
    for step in ticker:

        score_network.train()
            
        optimizer.zero_grad()
        
        t = rng.uniform(min_t, 1.0)
        
        x, y = next(iterator.train)
        
        x_t, gt_score_t = diffuser.forward_marginal(x.detach().cpu().numpy(), t=t)
        
        score_scaling = torch.tensor(diffuser.score_scaling(t)).to(device)
        gt_score_t = torch.tensor(gt_score_t).to(device)
        
        if np.random.random() > 0.5:
            pred_score_t = score_network((torch.tensor(x_t).float().to(device), y.to(device)), t)
        else:
            null_cond = torch.zeros_like(y)
            pred_score_t = score_network((torch.tensor(x_t).float().to(device), null_cond.to(device)), t)

        score_mse = (gt_score_t - pred_score_t)**2
        score_loss = torch.sum(
            score_mse / score_scaling[None, None]**2,
            dim=(-1, -2)
        ) #/ (loss_mask.sum(dim=-1) + 1e-10)    
        
        score_loss.backward()
        optimizer.step()

        if step % config.training.logs_freq == 0:
            # log to logger object
            # logger.log("train", loss=loss.item(), step=step, **comps)
            writer.add_scalar('Training loss', score_loss.item(), global_step=step)
            print(f'At step {step}, TRAINING loss is {score_loss.item()}')
            
        if step % eval_freq == 0:
            mean_mse = eval(step, dt=0.01)
            print(f'At step {step}, \n mse is {mean_mse}')

if __name__ == "__main__":
    train()

# %%
