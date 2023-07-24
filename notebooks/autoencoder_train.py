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

from cellot.models.ae import AutoEncoder, ConditionalAutoEncoder, VariationalAutoEncoder

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("data_logger")
logger.setLevel(logging.INFO)


import argparse
import ast

def get_args():
    parser = argparse.ArgumentParser()

    # Argument for model.hidden_units. Assumes a list of integers in the format [int, int, ...]
    parser.add_argument("--hidden_units", type=str, default="[512, 512]",
                        help="List of hidden units for the model. Should be a list in the format [int, int, ...]. Default is [512, 512].")

    # Argument for model.dropout
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout for the model. Default is 0.0.")

    # Argument for model.latent_dim
    parser.add_argument("--latent_dim", type=int, default=50,
                        help="Latent dimension for the model. Default is 50.")
    
    # Argument for output directory
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory path. Default is current directory.")
    
    parser.add_argument("--model", type=str, default="scgen",
                        help="Model name. Default is scgen.")
    
    args = parser.parse_args()

    # Convert hidden_units from string to list of integers
    args.hidden_units = ast.literal_eval(args.hidden_units)

    return args


# %%
TARGET = 'all'
DEBUG = False

# %%
import omegaconf

if DEBUG:
    n_iters = 250000
    batch_size = 256
else:
    n_iters = 250000
    batch_size = 256


# %% [markdown]
# ### Utils

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

        elif name == "cae":
            model = ConditionalAutoEncoder
        
        elif name == "vae":
            model = VariationalAutoEncoder
        
        else:
            raise ValueError

        return model(**kwargs)
    
    model = load_networks(config, **kwargs).to(device)
    optim = load_optimizer(config, model.parameters())

    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore)
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optim_state"])
        if config.model.name == "scgen" and "code_means" in ckpt:
            model.code_means = ckpt["code_means"]
            
    logger.info(f'Model on device {next(model.parameters()).device}')

    return model, optim


def load(config, device, restore=None, include_model_kwargs=False, **kwargs):

    loader, model_kwargs = load_data(config, include_model_kwargs=True, **kwargs)
    # dataset, _ = load_data(config, include_model_kwargs=True, return_as='dataset', **kwargs)


    model, opt = load_model(config, device, restore=restore, **model_kwargs)

    return model, opt, loader# , dataset

def train_auto_encoder(outdir, config, device):
    def state_dict(model, optim, **kwargs):
        state = {
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
        }

        if hasattr(model, "code_means"):
            state["code_means"] = model.code_means

        state.update(kwargs)

        return state

    def evaluate(vinputs):
        with torch.no_grad():
            loss, comps, _ = model(vinputs)
            loss = loss.mean()
            comps = {k: v.mean().item() for k, v in comps._asdict().items()}
            check_loss(loss)
            logger.log("eval", loss=loss.item(), step=step, **comps)
        return loss
    

    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    model, optim, loader = load(config, device, restore=cachedir / "last.pt")
    
    # print('Saving test and train dataset splits')
    # torch.save(dataset.test, cachedir / "test.pt")
    # torch.save(dataset.train, cachedir / "train.pt")
    # print('Done saving test and train dataset splits')

    iterator = cast_loader_to_iterator(loader, cycle_all=True)
    scheduler = load_lr_scheduler(optim, config)

    n_iters = config.training.n_iters
    step = load_item_from_save(cachedir / "last.pt", "step", 0)
    if scheduler is not None and step > 0:
        scheduler.last_epoch = step

    best_eval_loss = load_item_from_save(
        cachedir / "model.pt", "best_eval_loss", np.inf
    )

    eval_loss = best_eval_loss

    ticker = trange(step, n_iters, initial=step, total=n_iters)
    # model.to(device)
    
    if DEBUG:
        ex_batch = torch.load('/Mounts/rbg-storage1/users/johnyang/cellot/ex_batch_sciplex3.pt', map_location=device)
        print('MEMORIZING MODEEEE')
    else:
        print('NOT memorizing')
        
    for step in ticker:

        model.train()
        if DEBUG:
            inputs = ex_batch
        else:
            inputs = next(iterator.train)
            inputs = [x.to(device) for x in inputs]
            if config.model.name == "scgen":
                inputs = inputs[0]
        optim.zero_grad()
        loss, comps, _ = model(inputs)
        loss = loss.mean()
        comps = {k: v.mean().item() for k, v in comps._asdict().items()}
        loss.backward()
        optim.step()
        check_loss(loss)

        if step % config.training.logs_freq == 0:
            # log to logger object
            logger.log("train", loss=loss.item(), step=step, **comps)
            print(f'At step {step}, TRAINING loss is {loss.item()}')

        if step % config.training.eval_freq == 0:
            model.eval()
            if DEBUG:
                test_inputs = ex_batch
            else:
                test_inputs = next(iterator.test)
                test_inputs = [x.to(device) for x in test_inputs]
                if config.model.name == "scgen":
                    test_inputs = test_inputs[0]
                
            eval_loss = evaluate(test_inputs)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                sd = state_dict(model, optim, step=(step + 1), eval_loss=eval_loss)

                torch.save(sd, cachedir / "model.pt")
            print(f'At step {step}, TEST loss is {eval_loss.item()}')

        if step % config.training.cache_freq == 0:
            torch.save(state_dict(model, optim, step=(step + 1)), cachedir / "last.pt")

            logger.flush()

        if scheduler is not None:
            scheduler.step()

    # if config.model.name == "scgen" and config.get("compute_scgen_shift", True):
    #     labels = loader.train.dataset.adata.obs[config.data.condition]
    #     compute_scgen_shift(model, loader.train.dataset, labels=labels, device=device)

    torch.save(state_dict(model, optim, step=step), cachedir / "last.pt")

    logger.flush()


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

# %%
if __name__ == "__main__":
    device = get_free_gpu()
    args = get_args()
    # Use args.hidden_units, args.dropout, and args.latent_dim as necessary in your code
    print(f"model: {args.model}")
    print(f"hidden_units: {args.hidden_units}")
    print(f"dropout: {args.dropout}")
    print(f"latent_dim: {args.latent_dim}")

    yaml_str = f"""
    model:
        name: {args.model}
        beta: 0.0
        dropout: {args.dropout}
        hidden_units: {args.hidden_units}
        latent_dim: {args.latent_dim}

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
    from pathlib import Path
    outdir_path = args.outdir
    outdir = Path(outdir_path)

    # %%
    outdir.mkdir(exist_ok=True, parents=True)
    cachedir = outdir / "cache"
    cachedir.mkdir(exist_ok=True)

    train_auto_encoder(outdir, config, f'cuda:{device}')