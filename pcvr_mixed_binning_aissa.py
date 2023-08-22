import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from functools import reduce
import math
from einops import rearrange, repeat
from perceiver_pytorch import PerceiverIO
from torch import nn, einsum
import json
import glob
import argparse

# from cellgp.datasets import read_vocab
# from cellgp.tokenizer import Tokenizer
# from cellgp.utils import equal_area_bins, equal_width_bins
from tokenizer import Tokenizer
from utils import equal_area_bins, equal_width_bins
import h5py as h5
from functools import partial
import numpy as np

import pickle
from torch.utils.data import Dataset
import scanpy as sc
from scipy import sparse
import pandas as pd

from cellot.models.cond_score_module import CondScoreModule
from cellot.models.score_network import get_timestep_embedding
from omegaconf import DictConfig, OmegaConf
from functools import wraps


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


DEFAULT_ENCODING = "utf-8"

def read_vocab(path):
    with open(path, "r", encoding=DEFAULT_ENCODING) as inf:
        return {gene: i for i, gene in enumerate(json.load(inf))}
    
def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class Encoder(pl.LightningModule):
    def __init__(
        self,
        mask_prob,
        mask_ignore_token_ids,
        mask_token_id,
        emb_dim,
        logits_dim,
        depth,
        num_latents,
        latent_dim,
        cross_heads,
        latent_heads,
        cross_dim_head,
        latent_dim_head,
        weight_tie_layers,
        seq_dropout_prob,
        max_len: int,
        emb_size: int,
    ):
        super().__init__()
        self.mask_ignore_token_ids = set(mask_ignore_token_ids)
        self.mask_prob = mask_prob
        # TODO: clean up later
        self.mask_token_id = mask_token_id
        self.emb_dim = emb_dim

        self.model = PerceiverIO(
            dim=emb_dim,  # dimension of sequence to be encoded
            queries_dim=emb_dim,  # dimension of decoder queries
            logits_dim=logits_dim,  # dimension of final logits
            depth=depth,  # depth of net
            num_latents=num_latents,  # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=latent_dim,  # latent dimension
            cross_heads=cross_heads,  # number of heads for cross attention. paper said 1
            latent_heads=latent_heads,  # number of heads for latent self attention, 8
            cross_dim_head=cross_dim_head,  # number of dimensions per cross attention head
            latent_dim_head=latent_dim_head,  # number of dimensions per latent self attention head
            weight_tie_layers=weight_tie_layers,  # whether to weight tie layers (optional, as indicated in the diagram)
            seq_dropout_prob=seq_dropout_prob,  # fraction of the tokens from the input sequence to dropout (structured dropout, for saving compute and regularizing effects)
        )

        self.seq_dropout_prob = seq_dropout_prob

        self.queries = torch.randn(max_len, self.emb_dim)  # latent_dim
        self.emb = nn.Embedding(emb_size, self.emb_dim)
        # self.pos_emb = nn.Embedding(max_len, self.emb_dim)  # +1 for exp

    def _prob_mask_like(self, t, prob):
        return torch.zeros_like(t).float().uniform_(0, 1) < prob

    def _get_mask_subset_with_prob(self, t, mask, prob):
        batch, seq_len, device = *mask.shape, mask.device
        max_masked = math.ceil(prob * seq_len)

        num_tokens = mask.sum(dim=-1, keepdim=True)
        mask_excess = mask.cumsum(dim=-1) > (num_tokens * prob).ceil()
        mask_excess = mask_excess[:, :max_masked]

        rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
        _, sampled_indices = rand.topk(max_masked, dim=-1)
        sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

        new_mask = torch.zeros((batch, seq_len + 1), device=device)
        new_mask.scatter_(-1, sampled_indices, 1)
        return new_mask[:, 1:].bool()

    def forward(self, gid, eq_width_bin, eq_area_bin, pad_mask):
        # if masked training
        if self.training and self.mask_prob > 0:
            mask = self._get_mask_subset_with_prob(gid, pad_mask, self.mask_prob)
            gid = gid.masked_fill(mask, self.mask_token_id)
            eq_width_bin = eq_width_bin.masked_fill(mask, self.mask_token_id)
            eq_area_bin = eq_area_bin.masked_fill(mask, self.mask_token_id)

        x = self.emb(gid)
        x += self.emb(eq_width_bin)
        x += self.emb(eq_area_bin)

        # n, device = x.shape[1], x.device
        # pos_emb = self.pos_emb(torch.arange(n, device=device))
        # pos_emb = rearrange(pos_emb, "n d -> () n d")
        # x = x + pos_emb

        z = self.model(x, pad_mask)
        return x, z


class Decoder(pl.LightningModule):
    def __init__(
        self, emb_dim, logits_dim, latent_dim, cross_heads, cross_dim_head, decoder_ff
    ):
        super().__init__()

        self.decoder_cross_attn = PreNorm(
            emb_dim,
            Attention(emb_dim, latent_dim, heads=cross_heads, dim_head=cross_dim_head),
            context_dim=latent_dim,
        )
        self.decoder_ff = PreNorm(emb_dim, FeedForward(emb_dim)) if decoder_ff else None
        self.to_logits = (
            nn.Linear(emb_dim, logits_dim) if exists(logits_dim) else nn.Identity()
        )

    def forward(self, x, z):
        latents = self.decoder_cross_attn(x, context=z)
        latents = latents + self.decoder_ff(latents)
        return self.to_logits(latents)
    
class PcvrLatentScoreNetwork(nn.Module):
    def __init__(self, cfg: DictConfig, weight_tie_layers=False):
        super(PcvrLatentScoreNetwork, self).__init__()
        
        self.embed_dim = cfg.embed_dim
        self.cond_classes = cfg.cond_classes
        latent_dim = cfg.latent_dim
        # dim = cfg.dim
        cross_heads = cfg.cross_heads
        cross_dim_head = cfg.cross_dim_head
        heads = cfg.heads
        latent_dim_head = cfg.latent_dim_head
        depth = cfg.depth
        self.use_hidden_attn = cfg.use_hidden_attn
        
        self.cond_embedding = nn.Embedding(self.cond_classes + 1, self.embed_dim)
        self.null_cond_idx = self.cond_classes
        
        self.timestep_embedder = partial(
            get_timestep_embedding,
            embedding_dim=self.embed_dim,
        )

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, self.embed_dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=self.embed_dim)
        get_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, latent_dim, heads=heads, dim_head=latent_dim_head))
        get_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))

        get_cross_attn, get_attn, get_ff = map(cache_fn, (get_cross_attn, get_attn, get_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        if self.use_hidden_attn:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    get_cross_attn(**cache_args),
                    get_attn(**cache_args),
                    get_ff(**cache_args)
                ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    get_cross_attn(**cache_args),
                    get_ff(**cache_args)
                ]))

    def forward(self, z_t, y, t):
        '''
        Cross attend between z_t and y plus t.
        Attend bw z_t and z_t.
        Feedforward.
        
        - z_t: latent state noised to time t
        - y: condition tensor
        - t: time scalar
        '''
        device = z_t.device
        y_plus_t = self.cond_embedding(y) + self.timestep_embedder(torch.tensor([t]).to(device)) #NOTE: [B, N, K] + [B, K] => [B, N, K]
        hidden = z_t
        if self.use_hidden_attn:
            for cross_attn, attn, ff in self.layers:
                hidden = cross_attn(hidden, context=y_plus_t) + hidden
                hidden = attn(hidden) + hidden
                hidden = ff(hidden) + hidden
        else:
            for cross_attn, ff in self.layers:
                hidden = cross_attn(hidden, context=y_plus_t) + hidden
                hidden = ff(hidden) + hidden
        return hidden

class PcvrLatentScoreModule(CondScoreModule):  
    
    def __init__(self, hparams: DictConfig, cellgp):
        super().__init__(hparams)
        self.score_network = PcvrLatentScoreNetwork(hparams.latent_pcvr_score_network)
        self.encoder = cellgp.encoder

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
                
        self.mask_ignore_token_ids = hparams.mask_ignore_token_ids
        self.loss_type = hparams.latent_pcvr_score_network.loss_type
        
    def _mask_with_tokens(self, t, token_ids):
        init_no_mask = torch.full_like(t, False, dtype=torch.bool)
        mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
        return mask
    
    def forward(self, z_t_y, t):
        z_t, y = z_t_y #NOTE: legacy score_network takes (x_t, y), t
        return self.score_network(z_t, y, t)
    
    def encode(self, batch):
        self.encoder.eval()
        gid, equal_width_bins, equal_area_bins, y = batch
        
        no_mask = self._mask_with_tokens(gid, self.mask_ignore_token_ids)
        pad_mask = ~no_mask
        x_emb, z = self.encoder(gid, equal_width_bins, equal_area_bins, pad_mask)
        return x_emb, z, y
        
    def training_step(self, batch, batch_idx):
        x_emb, z, y = self.encode(batch)
        
        z_0 = z
        t = np.random.uniform(self.min_t, 1.0) 

        z_t, gt_score_t = self.diffuser.forward_marginal(z.detach().cpu().numpy(), t=t)

        if np.random.random() > 0.5:
            pred_z_0 = self((torch.tensor(z_t).float().to(self.device), y.to(self.device)), t)
        else:
            null_cond = torch.ones_like(y) * self.score_network.null_cond_idx
            pred_z_0 = self((torch.tensor(z_t).float().to(self.device), null_cond.to(self.device)), t)
            
        if self.loss_type == 'MSE':
            loss = torch.mean((z_0 - pred_z_0)**2)
            self.log('train/x_0_mse', loss.item(), sync_dist=True)
        elif self.loss_type == 'pred_x_0_sm':
            score_scaling = torch.tensor(self.diffuser.score_scaling(t)).to(self.device)
            gt_score_t = torch.tensor(gt_score_t).to(self.device)
            z_t = torch.tensor(z_t).float().to(self.device)
            pred_score_t = self.diffuser.score(x_t=z_t, x_0=pred_z_0, t=t, use_torch=True)
            score_mse = (gt_score_t - pred_score_t)**2
            loss = torch.sum(
                score_mse / score_scaling[None, None]**2,
                dim=(-1, -2)
            )
            loss = torch.mean(loss)
            self.log('train/score_mse_loss', loss.item(), sync_dist=True)
        else:
            raise ValueError(f'Unknown loss type {self.loss_type}')
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_emb, z, y = self.encode(batch)
        
        # x_t_fwd = z
        # for fwd_t in np.arange(0 + self.dt, 1.0, self.dt):
        #     x_t_fwd = torch.tensor(x_t_fwd).float().to(self.device)
        #     pred_x_0 = self((x_t_fwd, y), fwd_t)
        #     fwd_cond_score = self.diffuser.score(x_t=x_t_fwd, x_0=pred_x_0, t=fwd_t, use_torch=True)
        #     x_t_fwd = self.forward_ODE_step(x_t_fwd, fwd_cond_score, fwd_t, self.dt)
    
        x_t_rvs = np.random.normal(size=z.shape) 
        #NOTE: Take Peter's suggestion abt checking whether the prior aligns with the noised distribution into account.
        
        for rvs_t in np.arange(1.0, 0, -self.dt):
            x_t_rvs = torch.tensor(x_t_rvs).float().to(self.device)
            pred_x_0 = self((x_t_rvs, y), rvs_t)
            rvs_cond_score = self.diffuser.score(x_t=x_t_rvs, x_0=pred_x_0, t=rvs_t, use_torch=True)
            x_t_rvs = self.reverse_ODE_step(x_t_rvs, rvs_cond_score, rvs_t, self.dt)
        
        z_0 = torch.tensor(x_t_rvs).to(self.device)

        mse = torch.mean((z - z_0) ** 2)
        self.log('val/mse', mse.item(), sync_dist=True)
        return mse

class CellGP(pl.LightningModule):
    def __init__(
        self,
        nbins,
        max_len,
        genes,
        extra_tokens,
        emb_size,
        lr=1e-4,
        mask_prob=0.15,
        mask_ignore_token_ids=[0],
        mask_token_id=1,
        emb_dim=256,
        logits_dim_enc=None,
        logits_dim_dec=1,
        depth=6,
        num_latents=256,
        latent_dim=256,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,
        seq_dropout_prob=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.extra_tokens = extra_tokens
        self.nbins = nbins
        self.genes = genes

        self.encoder = Encoder(
            mask_prob,
            mask_ignore_token_ids,
            mask_token_id,
            emb_dim,
            logits_dim_enc,
            depth,
            num_latents,
            latent_dim,
            cross_heads,
            latent_heads,
            cross_dim_head,
            latent_dim_head,
            weight_tie_layers,
            seq_dropout_prob,
            max_len=max_len,
            emb_size=emb_size,
        )

        self.decoder = Decoder(
            emb_dim,
            logits_dim=nbins,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            cross_dim_head=cross_dim_head,
            decoder_ff=True,
        )

        for p in self.parameters():
            p.requires_grad_()

        self.criterion = nn.CrossEntropyLoss(
            reduction="sum"
        )  # divide by the appropriate
        self.mask_ignore_token_ids = set(mask_ignore_token_ids)

    def forward(self, gid, equal_width_bins, equal_area_bins, pad_mask):
        x_emb, z = self.encoder(gid, equal_width_bins, equal_area_bins, pad_mask)
        out = self.decoder(x_emb, z)
        return out

    def _mask_with_tokens(self, t, token_ids):
        init_no_mask = torch.full_like(t, False, dtype=torch.bool)
        mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
        return mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)

    def _shared_eval_step(self, batch, batch_idx):
        gid, equal_width_bins, equal_area_bins, _ = batch #NOTE: No condition token in pretrained CellGP
        no_mask = self._mask_with_tokens(gid, self.mask_ignore_token_ids)
        pad_mask = ~no_mask

        bin_hat = self(gid, equal_width_bins, equal_area_bins, pad_mask)

        no_reduc = self.criterion(
            rearrange(bin_hat, "b g c -> b c g"),
            equal_area_bins - len(self.genes) - self.extra_tokens - self.nbins,
        )
        non_zero_elements = pad_mask.sum()
        loss = no_reduc / non_zero_elements

        return loss


def fixed_bin_collate(batch, extra_tokens, max_gid, genes, n_bins):
    """
    Collate function for dataloader providing binning on a per cell basis
    
    - batch: tuple of (gid, counts, med, cond_token)
    
    """

    n_batch = len(batch)

    # these are indices in the original vocabulary that we want for the fixed representation
    genes_t = torch.tensor(genes)

    # this array allows us to remap vocab integers to integers between 0-1
    index = torch.zeros((n_batch, max_gid + 1), dtype=torch.float)
    condition = torch.zeros((n_batch, 1), dtype=torch.int)

    for i, (gid, counts, med, cond_token) in enumerate(batch):
        # print('gid', gid, 'counts', counts, 'med', med, 'cond_token', cond_token)
        counts_t = torch.tensor(counts)
        # non_zero_gid = torch.tensor(gid, dtype=torch.long)[counts_t > 0]
        index[i, gid] = torch.log1p(counts_t * med / counts_t.sum())
        condition[i] = cond_token
    
    fixed_exp = index[:, genes_t]
    fixed_gid = (
        torch.vstack(
            [torch.arange(0, len(genes), dtype=torch.int64) for _ in range(n_batch)]
        )
        + extra_tokens
    )

    fixed_eq_width_bin = equal_width_bins(fixed_exp, n_bins) + len(genes) + extra_tokens
    fixed_eq_area_bin = (
        equal_area_bins(fixed_exp, n_bins) + len(genes) + extra_tokens + n_bins
    )

    return fixed_gid, fixed_eq_width_bin, fixed_eq_area_bin, condition


def test_collate():
    batch = [[list(range(10)), list(range(10)), 5], [[1, 2, 3], [1, 2, 3], 5]]
    fg, ewb, eab, _ = fixed_bin_collate(batch, 2, 10, [3, 4, 5, 6, 7], n_bins=2)
    print(fg, ewb, eab)
    breakpoint()


class SCDataset(Dataset):
    def __init__(self, path):
        raise NotImplementedError
        self.hf = h5.File(path)

    def __len__(self):
        return len(self.hf["gid"])

    def __getitem__(self, idx):
        return self.hf["gid"][idx], self.hf["exp"][idx], self.hf["med"][idx]


class DM(pl.LightningDataModule):
    def __init__(
        self,
        path,
        vocab,
        subset_genes,
        nbins,
        train_percentage=0.85,
        val_percentage=0.10,
        test_percentage=0.05,
        batch_size=16,
        num_workers=16,
        timeout=5,
        collate_fn=None,
    ):
        super().__init__()
        assert collate_fn is not None
        self.path = path

        self.nbins = nbins
        self.vocab = vocab
        self.collate_fn = collate_fn
        self.train_percentage = train_percentage
        self.val_percentage = val_percentage
        self.test_percentage = test_percentage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.timeout = timeout

    def setup(self, stage: str = "default"):
        self.dset = SCDataset(self.path)
        self.dset_train, self.dset_val, self.dset_test = random_split(
            self.dset,
            [self.train_percentage, self.val_percentage, self.test_percentage],
        )

    def train_dataloader(self):
        return DataLoader(
            self.dset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            timeout=self.timeout,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        """
        Returns a DataLoader that loads the validation dataset.
        """
        return DataLoader(
            self.dset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            timeout=self.timeout,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        """
        Returns a DataLoader that loads the holdout (test) dataset.
        """
        return DataLoader(
            self.dset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            timeout=self.timeout,
            collate_fn=self.collate_fn,
        )

    def teardown(self, stage: str):
        pass
    
class PcvrAnnDataDataset(Dataset):
    def __init__(
        self, adata, obs=None, categories=None, include_index=False, dim_red=None, pca=False,
    ):
        self.adata = adata
        self.adata.X = self.adata.X.astype(np.float32)
        self.obs = obs
        self.categories = categories
        self.include_index = include_index
        self.pca = pca
        self.gid = np.array(self.adata.var['gid'].values, dtype=np.int64)
        self.med = self.adata.obs['med'].values[0] #constant across all cells.

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        '''
        Returns a tuple of (gid, counts, median, condition) for Pcvr model.
        - gid: integer array of gene_ids that have non-zero count values from original vocab dict
        - counts: raw non-zero count data from adata.X
        - median: median of counts across all cells
        - condition: integer representing the condition of the cell
        '''
        
        if self.pca:
            X = self.adata.obsm['X_pca'][idx]
        else:
            X = self.adata.X[idx]
        
        if isinstance(X, sparse.csc_matrix) or isinstance(X, sparse.csr_matrix):
            X = X.toarray()
            
        assert isinstance(X, np.ndarray), f"Expected np.ndarray, got {type(X)}"
        
        if len(X.shape) > 1:
            X = X.squeeze()
                    
        non_zero_gid = self.gid[X > 0]
        non_zero_X = X[X > 0]

        if self.obs is not None:
            category_value = self.adata.obs[self.obs].iloc[idx]
            if pd.isna(category_value):
                return non_zero_gid, non_zero_X, self.med, len(self.categories) + 1 #NOTE: We use +2 because we reserve +1 for the null condition index.
            
            meta = self.categories.index(category_value)
            value = non_zero_gid, non_zero_X, self.med, int(meta)
        else:
            raise ValueError("Must provide obs column to use condition.")

        if self.include_index:
            raise ValueError("include_index should not be called for Pcvr.")
            return self.adata.obs_names[idx], value

        return value


class AissaDM(DM):
    
    def __init__(
        self,
        path,
        vocab,
        subset_genes,
        nbins,
        train_percentage=0.85,
        val_percentage=0.10,
        test_percentage=0.05,
        batch_size=16,
        num_workers=16,
        timeout=5,
        collate_fn=None,
        count_threshold=50,
        symbol_to_ensg_dict=None,
    ):
        super().__init__(
            path,
            vocab,
            subset_genes,
            nbins,
            train_percentage,
            val_percentage,
            test_percentage,
            batch_size,
            num_workers,
            timeout,
            collate_fn,
        )
        assert symbol_to_ensg_dict is not None
        
        self.symbol_to_ensg_dict = symbol_to_ensg_dict
        self.count_threshold = count_threshold
        

    def setup(self, stage: str = "default"):
        adata = sc.read_h5ad(self.path)
        
        def process_adata(adata):
            '''
            Filter out cells with low counts and map gene symbols to ensg ids.
            '''
        
            adata = adata[adata.obs['ncounts'] > self.count_threshold]
            adata.var['ensg'] = adata.var.index.map(self.symbol_to_ensg_dict) 
            
            #Map ensg to vocab
            vocab_pre_dot = {k.split('.')[0]: v for k, v in self.vocab.items()}
            adata = adata[:, adata.var['ensg'].isna() == False]
            adata.var['gid'] = adata.var['ensg'].map(vocab_pre_dot)
            adata = adata[:, adata.var['gid'].isna() == False]
            adata.obs['med'] = adata.obs['ncounts'].median()
        
            return adata
        
        adata = process_adata(adata)

        self.dset = PcvrAnnDataDataset(adata, 'perturbation', categories=sorted(adata.obs['perturbation'].cat.categories), pca=False)

        # self.dset = SCDataset(self.path)
        self.dset_train, self.dset_val, self.dset_test = random_split(
            self.dset,
            [self.train_percentage, self.val_percentage, self.test_percentage],
        )
    
def main(args):
    # Configuration
    cfg = {
        "latent_pcvr_score_network": {
            "base": args.base,
            "embed_dim": args.embed_dim,
            "cond_classes": args.cond_classes,
            "latent_dim": args.latent_dim,
            "cross_heads": args.cross_heads,
            "cross_dim_head": args.cross_dim_head,
            "heads": args.heads,
            "latent_dim_head": args.latent_dim_head,
            "depth": args.depth,
            "loss_type": args.loss_type,
            "use_hidden_attn": args.use_hidden_attn,
        },
        "mask_ignore_token_ids": [0],
        "diffuser": {
            "min_b": 0.01,
            "max_b": 1.0,
            "schedule": "exponential",
            "score_scaling": "var",
            "coordinate_scaling": 1.0,
            "latent_dim": args.latent_dim,
            "dt": 0.01,
            "min_t": 0
        },
        "experiment": {
            "lr": 1e-4
        }
    }
    cfg = OmegaConf.create(cfg)
    
    devices = [int(device) for device in args.devices.split(',')]
    
    original_vocab = read_vocab("/data/rsg/nlp/ujp/cellgp/vocab.json")
    with open("/data/rsg/nlp/ujp/cellgp/run_encoder/ccle_vocab_genes.json", "r") as inf:
        ccle_vocab_genes = sorted(json.load(inf))
    BINS = 64
    # tokenizer = Tokenizer()
    extra_tokens = 2

    # test_collate()
    extra_tokens = 2
    
    cell_gp_checkpoint_path = '/data/rsg/nlp/ujp/cellgp/cellgp/models/perceiverio/CellGP_Encoder/lo8jpygf/checkpoints/epoch=2-step=83589.ckpt'

    cellgp = CellGP.load_from_checkpoint(
        cell_gp_checkpoint_path,
        map_location='cpu',
        nbins=BINS,
        max_len=len(ccle_vocab_genes),
        genes=ccle_vocab_genes,
        extra_tokens=extra_tokens,
        emb_size=len(ccle_vocab_genes) + extra_tokens + 2 * BINS,
        lr=1e-4,
        mask_prob=0.5,
        mask_ignore_token_ids=[0],
        mask_token_id=1,
        emb_dim=256,
        logits_dim_enc=None,
        logits_dim_dec=BINS,
        depth=8,
        num_latents=256,
        latent_dim=256,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=256,
        latent_dim_head=256,
        weight_tie_layers=False,
        seq_dropout_prob=0.1,
    )
    
    lm = PcvrLatentScoreModule(cfg, cellgp)

    to_ensg_path = '/data/rsg/nlp/ujp/cellgp/datasets/nano_hugo_to_ensg.pkl'
    with open(to_ensg_path, 'rb') as f:
        to_ensg = pickle.load(f)

    dm = AissaDM(
        "/Mounts/rbg-storage1/users/johnyang/cellot/datasets/AissaBenevolenskaya2021.h5ad",
        vocab=original_vocab,
        subset_genes=ccle_vocab_genes,
        nbins=BINS,
        train_percentage=0.85,
        val_percentage=0.10,
        test_percentage=0.05,
        batch_size=48, 
        num_workers=64, 
        timeout=10,
        collate_fn=partial(
            fixed_bin_collate,
            extra_tokens=extra_tokens,
            max_gid=max(original_vocab.values()),
            genes=ccle_vocab_genes,
            n_bins=BINS,
        ),
        symbol_to_ensg_dict=to_ensg,
        count_threshold=50,
    )
    # import os
    # os.environ["WANDB_MODE"] = "dryrun" 

    wandb_logger = WandbLogger(log_model="all", project="sc_diff", name=args.name)
    cback = ModelCheckpoint(monitor="val/mse", mode="min", save_top_k=1)
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp", #_find_unused_parameters_true",
        devices=devices,
        precision=16,
        max_epochs=-1, #TODO: Change
        logger=wandb_logger,
        callbacks=[cback],
        check_val_every_n_epoch=25,
        # overfit_batches=1, 
        log_every_n_steps=10, 
        # fast_dev_run=True, 
    )
    trainer.fit(lm, dm)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Configuration arguments
    parser.add_argument('--devices', type=str, required=True, help='Comma-separated list of devices')
    parser.add_argument('--name', type=str, required=True, help='Name of experiment.')
    parser.add_argument('--use_hidden_attn', type=bool, default=True, help='Whether to use hidden attention in the latent score network.')
    parser.add_argument('--base', type=bool, default=False, help='Base configuration value.')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--cond_classes', type=int, default=4, help='Number of condition classes.')
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension.')
    parser.add_argument('--cross_heads', type=int, default=4, help='Number of cross heads.')
    parser.add_argument('--cross_dim_head', type=int, default=32, help='Cross dimension head.')
    parser.add_argument('--heads', type=int, default=4, help='Number of heads.')
    parser.add_argument('--latent_dim_head', type=int, default=32, help='Latent dimension head.')
    parser.add_argument('--depth', type=int, default=3, help='Depth.')
    parser.add_argument('--loss_type', type=str, default='MSE', choices=['MSE', 'pred_x_0_sm'], help='Type of loss function.')

    args = parser.parse_args()
    main(args)