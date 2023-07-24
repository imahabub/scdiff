import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import functools as fn

class FeedForward(nn.Module):
    def __init__(self, input_dim, n_layers, model_dim, final_layers, output_dim):
        super(FeedForward, self).__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for _ in range(n_layers):
            self.layers.append(nn.Linear(prev_dim, model_dim))
            prev_dim = model_dim
        
        if isinstance(final_layers, list):
            for hidden_dim in final_layers:
                self.layers.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim
        elif isinstance(final_layers, int):
            self.layers.append(nn.Linear(prev_dim, final_layers))
            prev_dim = final_layers
        else:
            raise ValueError(f"final_layers must be list or int, got {type(final_layers)}")
            
        self.layers.append(nn.Linear(prev_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
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

        # self.latent_dim = cfg.latent_dim
        self.input_dim = cfg.input_dim
        self.embed_data_dim = cfg.embed_data_dim
        self.model_dim = cfg.model_dim
        self.n_layers = cfg.n_layers
        self.final_layers = cfg.final_layers
        
        self.cond_classes = cfg.cond_classes
        self.extra_null_cond_embedding = cfg.extra_null_cond_embedding
        self.dropout = cfg.dropout
        # self.nhead = cfg.nhead
        # self.dim_feedforward = cfg.dim_feedforward
        # self.ffn_hidden_dims = cfg.ffn_hidden_dims

        print(f'Dropout is {self.dropout}')

        # if self.extra_null_cond_embedding:
        self.cond_embedding = nn.Embedding(self.cond_classes + 1, self.model_dim)
        self.null_cond_idx = self.cond_classes
        # else:
        #     self.cond_embedding = nn.Embedding(self.cond_classes, self.model_dim)
        #     self.null_cond_idx = 0
        
        self.embed_data = nn.Linear(self.input_dim, self.embed_data_dim)
        self.x_y_t_embed = nn.Linear(self.embed_data_dim + (2 * self.model_dim), self.model_dim)
        self.ffn = FeedForward(input_dim=self.model_dim, model_dim=self.model_dim, n_layers=self.n_layers, final_layers=self.final_layers, output_dim=self.input_dim)

        # transformer_layers = [nn.TransformerEncoderLayer(d_model=self.model_dim, 
        #                                                 nhead=self.nhead, 
        #                                                 dim_feedforward=self.dim_feedforward, 
        #                                                 dropout=self.dropout) 
        #                     for _ in range(self.n_layers)]
        # self.model = nn.ModuleList([self.x_y_t_embed, *transformer_layers, self.ffn])
        self.model = nn.ModuleList([self.x_y_t_embed, self.ffn])

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
        x_embed = self.embed_data(x).to(device)
        hidden = torch.cat([x_embed, t_embed, y_embed], dim=-1).to(device)
        for module in self.model[:-1]:  # iterate over all modules except the last one
            hidden = module(hidden)
        hidden = self.model[-1](hidden.squeeze(0))  # pass through the last module (FeedForward)
        return hidden
