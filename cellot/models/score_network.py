import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import functools as fn

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

from omegaconf import DictConfig

class ScoreNetwork(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ScoreNetwork, self).__init__()

        self.latent_dim = cfg.latent_dim
        self.model_dim = cfg.model_dim
        self.cond_classes = cfg.cond_classes
        self.dropout = cfg.dropout
        self.n_layers = cfg.n_layers
        self.nhead = cfg.nhead
        self.dim_feedforward = cfg.dim_feedforward
        self.ffn_hidden_dim = cfg.ffn_hidden_dim

        print(f'Dropout is {self.dropout}')

        self.cond_embedding = nn.Embedding(self.cond_classes, self.model_dim)
        self.embed_code_and_t = nn.Linear(self.latent_dim + (2 * self.model_dim), self.model_dim)
        self.ffn = FeedForward(input_dim=self.model_dim, hidden_dim=cfg.ffn_hidden_dim, output_dim=self.latent_dim)

        transformer_layers = [nn.TransformerEncoderLayer(d_model=self.model_dim, 
                                                        nhead=self.nhead, 
                                                        dim_feedforward=self.dim_feedforward, 
                                                        dropout=self.dropout) 
                            for _ in range(self.n_layers)]
        self.model = nn.ModuleList([self.embed_code_and_t, *transformer_layers, self.ffn])

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self.model_dim,
        )

    def forward(self, xy, t):
        x, y = xy
        device = x.device
        B, C = x.shape
        t_embed = torch.tile(self.timestep_embedder(torch.tensor([t]).to(device)), dims=[B, 1])
        y_embed = self.cond_embedding(y).to(device)
        x = torch.cat([x, t_embed, y_embed], dim=-1).to(device)
        for module in self.model[:-1]:  # iterate over all modules except the last one
            x = module(x)
        x = self.model[-1](x.squeeze(0))  # pass through the last module (FeedForward)
        return x
