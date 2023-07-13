# %%
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
TARGET = 'trametinib' #'all' if not DEBUG else 'trametinib'
LATENT_DIM = 50
COND_CLASSES = 189 if TARGET == 'all' else 2

# from pathlib import Path
# outdir_path = '/Mounts/rbg-storage1/users/johnyang/cellot/results/sciplex3/full_ae'
# outdir = Path(outdir_path)

# # %%
# outdir.mkdir(exist_ok=True, parents=True)

# cachedir = outdir / "cache"
# cachedir.mkdir(exist_ok=True)

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

# status = cachedir / "status"
# status.write_text("running")

device = f'cuda:{get_free_gpu()}'


# %%
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
restore_path = '/Mounts/rbg-storage1/users/johnyang/cellot/saved_weights/ae/ae.pt'
ae = load_model(config, 'cuda', restore=restore_path, input_dim=1000)


# %%

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
    # condition_labels = sorted(data.obs[condition].cat.categories)
    # model_kwargs["conditions"] = condition_labels
    # dataset_args["obs"] = condition
    # dataset_args["categories"] = condition_labels

    if "training" in config:
        pair_batch_on = config.training.get("pair_batch_on", pair_batch_on)

    # if split_on is None:
        # if config.model.name == "cellot":
            # datasets & dataloaders accessed as loader.train.source
    split_on = ["split", "transport"]
    if pair_batch_on is not None:
        split_on.append(pair_batch_on)

        # elif (config.model.name == "scgen" or config.model.name == "cae"
        #       or config.model.name == "popalign"):
        #     split_on = ["split"]

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

loader = load_cell_data(config, return_as="loader")

# %%
from cellot.data.utils import cast_loader_to_iterator
iterator = cast_loader_to_iterator(loader)

# %%
iterator.train.source

# %%
def load_lr_scheduler(optim, config):
    if "scheduler" not in config:
        return None

    return torch.optim.lr_scheduler.StepLR(optim, **config.scheduler)


def check_loss(*args):
    for arg in args:
        if torch.isnan(arg):
            raise ValueError


def load_item_from_save(path, key, default):
    path = Path(path)
    if not path.exists():
        return default

    ckpt = torch.load(path)
    if key not in ckpt:
        logging.warn(f"'{key}' not found in ckpt: {str(path)}")
        return default

    return ckpt[key]

# %%
# def train_cellot(outdir, config):
outdir_string = '/Mounts/rbg-storage1/users/johnyang/cellot/results/sciplex3/cellot_our_ae'
cellot_outdir = Path(outdir_string)

# %%
ot_config_str = f"""
model:
  name: cellot
  hidden_units: [64, 64, 64, 64]
  latent_dim: 50
  softplus_W_kernels: false

  g:
    fnorm_penalty: 1

  kernel_init_fxn:
    b: 0.1
    name: uniform

optim:
  optimizer: Adam
  lr: 0.0001
  beta1: 0.5
  beta2: 0.9
  weight_decay: 0

training:
  n_iters: 100000
  n_inner_iters: 10
  cache_freq: 1000
  eval_freq: 250
  logs_freq: 50
"""
ot_config = omegaconf.OmegaConf.create(ot_config_str)

# %%

def state_dict(f, g, opts, **kwargs):
    state = {
        "g_state": g.state_dict(),
        "f_state": f.state_dict(),
        "opt_g_state": opts.g.state_dict(),
        "opt_f_state": opts.f.state_dict(),
    }
    state.update(kwargs)

    return state

logger = Logger(cellot_outdir / "cache/scalars")
cachedir = cellot_outdir / "cache"

# %%
from pathlib import Path
import torch
from collections import namedtuple
from cellot.networks.icnns import ICNN

from absl import flags

FLAGS = flags.FLAGS

FGPair = namedtuple("FGPair", "f g")

def load_networks(config, **kwargs):
    def unpack_kernel_init_fxn(name="uniform", **kwargs):
        if name == "normal":

            def init(*args):
                return torch.nn.init.normal_(*args, **kwargs)

        elif name == "uniform":

            def init(*args):
                return torch.nn.init.uniform_(*args, **kwargs)

        else:
            raise ValueError

        return init

    kwargs.setdefault("hidden_units", [64] * 4)
    kwargs.update(dict(config.get("model", {})))

    # eg parameters specific to g are stored in config.model.g
    kwargs.pop("name")
    if "latent_dim" in kwargs:
        kwargs.pop("latent_dim")
    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})

    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
        **fkwargs.pop("kernel_init_fxn")
    )

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
        **gkwargs.pop("kernel_init_fxn")
    )

    f = ICNN(**fkwargs)
    g = ICNN(**gkwargs)

    if "verbose" in FLAGS and FLAGS.verbose:
        print(g)
        print(kwargs)

    return f, g


def load_opts(config, f, g):
    kwargs = dict(config.get("optim", {}))
    assert kwargs.pop("optimizer", "Adam") == "Adam"

    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})

    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["betas"] = (fkwargs.pop("beta1", 0.9), fkwargs.pop("beta2", 0.999))

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["betas"] = (gkwargs.pop("beta1", 0.9), gkwargs.pop("beta2", 0.999))

    opts = FGPair(
        f=torch.optim.Adam(f.parameters(), **fkwargs),
        g=torch.optim.Adam(g.parameters(), **gkwargs),
    )

    return opts

# %%
def load_cellot_model(config, restore=None, **kwargs):
    f, g = load_networks(config, **kwargs)
    f = f.to(device)
    g = g.to(device)
    
    opts = load_opts(config, f, g)

    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore, map_location='cpu')
        f.load_state_dict(ckpt["f_state"])
        opts.f.load_state_dict(ckpt["opt_f_state"])

        g.load_state_dict(ckpt["g_state"])
        opts.g.load_state_dict(ckpt["opt_g_state"])

    return (f, g), opts

# %%
(f, g), opts = load_cellot_model(config=ot_config, restore=cachedir / 'last.pt', input_dim=LATENT_DIM)
f, g = f.to(device), g.to(device)

# %%
loader.test.source.dataset.__len__()

# %%
from cellot.losses.mmd import mmd_distance
import numpy as np

def compute_mmd_loss(lhs, rhs, gammas):
    return np.mean([mmd_distance(lhs, rhs, g) for g in gammas])

gammas = np.logspace(1, -3, num=50)

# %%
autoencoder = ae[0].to(device)
autoencoder.eval()

# %%
iterator_train_target = iterator.train.target
iterator_train_source = iterator.train.source
iterator_test_target = iterator.test.target
iterator_test_source = iterator.test.source

def DEV_evaluate(target, source):
    source.requires_grad_(True)
    transport = g.transport(source).to(device)

    transport = transport.detach()
    with torch.no_grad():
        gl = compute_loss_g(f, g, source, transport).mean()
        fl = compute_loss_f(f, g, source, target, transport).mean()
        dist = compute_w2_distance(f, g, source, target, transport)
        mmd = losses.compute_scalar_mmd(
            target.detach().cpu().numpy(), transport.detach().cpu().numpy()
        )
        mmd_2 = compute_mmd_loss(target.detach().cpu().numpy(), transport.detach().cpu().numpy(), gammas)
        
        
        
        return mmd, mmd_2

mmds = []
mmd_2s = []
for _ in range(100):
    target = next(iterator_test_target).to(device) #TODO: Change to loader
    source = next(iterator_test_source).to(device)
    # print(target.shape, source.shape)
    mmd, mmd_2 = DEV_evaluate(target, source)
    mmds.append(mmd)
    mmd_2s.append(mmd_2)

# %%


# %%
np.mean(mmds)

# %%
mmd

# %%

# (f, g), opts, loader = load(ot_config, restore=cachedir / "last.pt")
# iterator = cast_loader_to_iterator(loader, cycle_all=True)

n_iters = ot_config.training.n_iters
# step = load_item_from_save(cachedir / "last.pt", "step", 0)

# minmmd = load_item_from_save(cachedir / "model.pt", "minmmd", np.inf)
# mmd = minmmd
step = 0
minmmd = np.inf
mmd = minmmd

def evaluate():
    target = next(iterator_test_target).to(device)
    source = next(iterator_test_source).to(device)
    source.requires_grad_(True)
    transport = g.transport(source).to(device)

    transport = transport.detach()
    with torch.no_grad():
        gl = compute_loss_g(f, g, source, transport).mean()
        fl = compute_loss_f(f, g, source, target, transport).mean()
        dist = compute_w2_distance(f, g, source, target, transport)
        mmd = losses.compute_scalar_mmd(
            target.detach().cpu().numpy(), transport.detach().cpu().numpy()
        )

    # log to logger object
    logger.log(
        "eval",
        gloss=gl.item(),
        floss=fl.item(),
        jloss=dist.item(),
        mmd=mmd,
        step=step,
    )
    check_loss(gl, gl, dist)

    return mmd

if 'pair_batch_on' in ot_config.training:
    keys = list(iterator.train.target.keys())
    test_keys = list(iterator.test.target.keys())
else:
    keys = None

ticker = trange(step, n_iters, initial=step, total=n_iters)
for step in ticker:
    if 'pair_batch_on' in ot_config.training:
        assert keys is not None
        key = random.choice(keys)
        iterator_train_target = iterator.train.target[key]
        iterator_train_source = iterator.train.source[key]
        try:
            iterator_test_target = iterator.test.target[key]
            iterator_test_source = iterator.test.source[key]
        # in the iid mode of the ood setting,
        # train and test keys are not necessarily the same ...
        except KeyError:
            test_key = random.choice(test_keys)
            iterator_test_target = iterator.test.target[test_key]
            iterator_test_source = iterator.test.source[test_key]

    else:
        iterator_train_target = iterator.train.target
        iterator_train_source = iterator.train.source
        iterator_test_target = iterator.test.target
        iterator_test_source = iterator.test.source
        
    target = next(iterator_train_target).to(device)
    
    for _ in range(ot_config.training.n_inner_iters):
        source = next(iterator_train_source).requires_grad_(True).to(device)

        opts.g.zero_grad()
        gl = compute_loss_g(f, g, source).mean()
        if not g.softplus_W_kernels and g.fnorm_penalty > 0:
            gl = gl + g.penalize_w()

        gl.backward()
        opts.g.step()

    source = next(iterator_train_source).requires_grad_(True).to(device)

    opts.f.zero_grad()
    fl = compute_loss_f(f, g, source, target).mean()
    fl.backward()
    opts.f.step()
    check_loss(gl, fl)
    f.clamp_w()

    if step % ot_config.training.logs_freq == 0:
        # log to logger object
        logger.log("train", gloss=gl.item(), floss=fl.item(), step=step)

    if step % ot_config.training.eval_freq == 0:
        mmd = evaluate()
        if mmd < minmmd:
            minmmd = mmd
            torch.save(
                state_dict(f, g, opts, step=step, minmmd=minmmd),
                cachedir / "model.pt",
            )

    if step % ot_config.training.cache_freq == 0:
        torch.save(state_dict(f, g, opts, step=step), cachedir / "last.pt")

        logger.flush()

torch.save(state_dict(f, g, opts, step=step), cachedir / "last.pt")

logger.flush()


# %%
# PATH = '/Mounts/rbg-storage1/users/johnyang/cellot/saved_weights/ae/ae.pt'
# torch.save({
#     'model_state': ae[0].state_dict(),
#     'optim_state': ae[1].state_dict(),
# }, PATH)

# %%



