# %%
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

import h5py as h5
from functools import partial
import numpy as np


# %%


VOCAB_SIZE = 60873

MAX_LEN = 18976

DEFAULT_ENCODING = "utf-8"

DEFAULT_BINS = 10


# %%


def read_vocab(path):
    with open(path, "r", encoding=DEFAULT_ENCODING) as inf:
        return {gene: i for i, gene in enumerate(json.load(inf))}


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


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


# Main classes #####################################################################################
####################################################################################################


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
        nbins: int = DEFAULT_BINS,
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

        self.queries = torch.randn(MAX_LEN, self.emb_dim)  # latent_dim
        self.emb = nn.Embedding(VOCAB_SIZE + nbins + 2, self.emb_dim)
        self.pos_emb = nn.Embedding(MAX_LEN, self.emb_dim)  # +1 for exp

    def forward(self, gid, bin_t, pad_mask):
        x = self.emb(gid)
        x += self.emb(bin_t)

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


class CellGP(pl.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        mask_prob=0.15,
        mask_ignore_token_ids=[0],
        mask_token_id=1,
        pad_token_id=0,
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
        nbins: int = DEFAULT_BINS,
        tokenizer=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        assert tokenizer is not None

        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer

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
            nbins=nbins,
        )

        self.decoder = Decoder(
            emb_dim,
            logits_dim=logits_dim_dec,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            cross_dim_head=cross_dim_head,
            decoder_ff=True,
        )

        # self.decoder = MLPDecoder(
        #    emb_dim,
        # )

        for p in self.parameters():
            p.requires_grad_()

        self.criterion = nn.CrossEntropyLoss()
        self.mask_ignore_token_ids = set(mask_ignore_token_ids)

    def forward(self, gid, bin_t, pad_mask):
        x_emb, z = self.encoder(gid, bin_t, pad_mask)
        bin_hat = self.decoder(x_emb, z)
        return bin_hat

    def _mask_with_tokens(self, t, token_ids):
        init_no_mask = torch.full_like(t, False, dtype=torch.bool)
        mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
        return mask

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)

    def _shared_eval_step(self, batch, batch_idx):
        gid_orig, bin_orig, mask_zero_batch = batch
        no_mask = self._mask_with_tokens(gid_orig, self.mask_ignore_token_ids) #True cannot be masked, False can be masked.
        pad_mask = ~no_mask #False cannot be masked, True can be masked.
        gid_max = gid_orig.max()
        
        # if masked training
        if self.mask_prob > 0:
            no_zero_mask = no_mask | mask_zero_batch #True cannot be masked, False can be masked.
            true_can_mask = ~no_zero_mask #False cannot be masked, True can be masked.
            
            mask = self._get_mask_subset_with_prob(gid_orig, true_can_mask, self.mask_prob)
            gid_t = gid_orig.masked_fill(mask, self.mask_token_id)
            bin_t = bin_orig.masked_fill(mask, self.mask_token_id)

        bin_hat = self(gid_t, bin_t, pad_mask)

        bin_hat_batch = torch.cat(
            [bin_hat[i, mask[i, :]] for i in range(bin_hat.shape[0])]
        )
        bin_orig_batch = torch.cat(
            [bin_orig[i, mask[i, :]] - gid_max - 1 for i in range(bin_orig.shape[0])],
        )
        loss = self.criterion(bin_hat_batch, bin_orig_batch)
        return loss


def fixed_bin_collate(batch, vocab, genes, bins):
    """
    Collate function for dataloader providing binning on a per cell basis
    """
    assert bins > 0
    bins = (
        bins - 2
    )  # searchsorted produces an extra bin on the left or right, and zero genes should get bin 0
    extra_tokens = 2

    # this array allows us to remap vocab integers to integers between 0-1
    index = torch.full((max(vocab.values()) + 1,), -1, dtype=torch.int64)
    index[genes] = torch.arange(0, len(genes), dtype=torch.int64)

    gid_batch = torch.zeros((len(batch), len(genes)), dtype=torch.int64)
    bins_batch = torch.full(
        (len(batch), len(genes)), len(genes) + extra_tokens, dtype=torch.int64
    )
    mask_zero_batch = torch.zeros((len(batch), len(genes)), dtype=torch.bool)

    # import ipdb
    # ipdb.set_trace()
    
    for i, (gid, exp, med) in enumerate(batch):
        if len(gid) == 0:
            continue
        try:
            gid_t = torch.tensor(gid, dtype=torch.int64)
            gid_batch[i] = index[genes] + extra_tokens
            exp_t = torch.tensor(exp)
            
            genes_in_gid = set([x.item() for x in gid_t[index[gid_t] != -1]])
            genes_not_in_gid = list(set(genes).difference(genes_in_gid))
            mask_zero_batch[i, index[genes_not_in_gid]] = True
            
            exp_t = exp_t[index[gid_t] != -1]
            exp_t = torch.log1p(exp_t * med / exp_t.sum())
            bin_e = torch.linspace(exp_t.min(), exp_t.max(), bins)
            bins_batch[i, index[gid_t[index[gid_t] != -1]]] = (
                torch.searchsorted(bin_e, exp_t, side="right")
                + len(genes)
                + extra_tokens
                + 1
            )
        except:
            continue
    return gid_batch, bins_batch, mask_zero_batch


class SCDataset(Dataset):
    def __init__(self, path):
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
    ):
        super().__init__()
        self.path = path

        self.nbins = nbins
        self.vocab = vocab
        self.collate_fn = partial(
            fixed_bin_collate,
            vocab=self.vocab,
            genes=subset_genes,
            bins=self.nbins,
        )
        self.train_percentage = train_percentage
        self.val_percentage = val_percentage
        self.test_percentage = test_percentage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.timeout = timeout

    def setup(self, stage: str = "default"):
        self.dset = SCDataset(self.path)
        
        val_size = int(self.val_percentage * len(self.dset))
        test_size = int(self.test_percentage * len(self.dset))
        train_size = len(self.dset) - val_size - test_size
        
        self.dset_train, self.dset_val, self.dset_test = random_split(
            self.dset,
            [ train_size, val_size, test_size ],
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




# %%
def test_collate():
    vocab = {v: k for k, v in enumerate("abcdefghij")}
    genes = [1, 2, 3]
    bins = 3
    print(vocab)
   
   
    gid = [2, 3, 4, 5]
    exp = [1, 1, 1, 1]
    med = 5
    batch = [[gid, exp, med]]
    print(fixed_bin_collate(batch, vocab, genes, bins))

# %%
tokenizer = read_vocab("vocab_ccle.json")
VOCAB_SIZE = len(tokenizer)
BINS = 64

# %%
with open("notebooks/ccle_vocab_genes.json", "r") as inf:
    ccle_vocab_genes = sorted(json.load(inf))
MAX_LEN = len(ccle_vocab_genes)

# %%
original_path = '/storage/ujp/processed_raw.h5'

dm = DM(
    original_path,
    vocab=tokenizer,
    subset_genes=ccle_vocab_genes,
    nbins=BINS,
    train_percentage=0.85,
    val_percentage=0.10,
    test_percentage=0.05,
    batch_size=16,
    num_workers=16,
    timeout=5,
)

# %%
test_collate()

# # %%
import os
os.environ["WANDB_MODE"] = "dryrun"

# %%

model = CellGP(
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
    nbins=BINS,
    tokenizer=tokenizer,
)

# trainer = pl.Trainer(
#     accelerator="cpu",
#     precision=32,
#     max_epochs=1,
# )

wandb_logger = WandbLogger(log_model="all", project="CellGP_Encoder", name="pretrained_mask_zero")
cback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
trainer = pl.Trainer(
    accelerator="gpu",
    strategy="auto",
    devices=[2],
    precision=16,
    max_epochs=11,
    logger=wandb_logger,
    callbacks=[cback],
    # fast_dev_run=True,
)
trainer.fit(model, dm)



