import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
from cellot.models.score_network import ScoreNetwork
from cellot.models.latent_diffuser import LatentDiffuser

class CondScoreModule(pl.LightningModule):
    
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.cfg = hparams
        self.score_network = self.init_network(hparams.score_network) # Initialize your network here
        self.writer = SummaryWriter(log_dir='tb_logs')
        self.diffuser = self.init_diffuser(hparams.diffuser) # Initialize your diffuser here
        self.dt = hparams.diffuser.dt
        self.min_t = hparams.diffuser.min_t
        
    def forward(self, xy, t):
        # Your forward method
        return self.score_network(xy, t)
    
    def init_network(self, hparams):
        return ScoreNetwork(hparams)
    
    def init_diffuser(self, hparams):
        return LatentDiffuser(hparams)

    def training_step(self, batch, batch_idx):
        x, y = batch
        t = np.random.uniform(self.min_t, 1.0)

        x_t, gt_score_t = self.diffuser.forward_marginal(x.detach().cpu().numpy(), t=t)

        score_scaling = torch.tensor(self.diffuser.score_scaling(t)).to(self.device)
        gt_score_t = torch.tensor(gt_score_t).to(self.device)

        if np.random.random() > 0.5:
            pred_score_t = self((torch.tensor(x_t).float().to(self.device), y.to(self.device)), t)
        else:
            null_cond = torch.ones_like(y) * self.score_network.null_cond_idx
            pred_score_t = self((torch.tensor(x_t).float().to(self.device), null_cond.to(self.device)), t)

        score_mse = (gt_score_t - pred_score_t)**2
        score_loss = torch.sum(
            score_mse / score_scaling[None, None]**2,
            dim=(-1, -2)
        ) 

        self.log('train/score_mse_loss', score_loss.item())
        return score_loss

    def configure_optimizers(self):
        # Your optimizer configuration
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.experiment.lr)
        return optimizer
    
    def forward_ODE_step(self, x_t, score_t, t, dt):
        return self.diffuser.ode(x_t=x_t.detach().cpu().numpy(), score_t=score_t.detach().cpu().numpy(), t=t, dt=dt)
    
    def reverse_ODE_step(self, x_t, score_t, t, dt):
        return self.diffuser.ode(x_t=x_t.detach().cpu().numpy(), score_t=score_t.detach().cpu().numpy(), t=t, dt=-dt)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        x_t_fwd = x
        for fwd_t in np.arange(0, 1.0, self.dt):
            x_t_fwd = torch.tensor(x_t_fwd).float().to(self.device)
            fwd_cond_score = self.score_network((x_t_fwd, y), fwd_t)
            x_t_fwd = self.forward_ODE_step(x_t_fwd, fwd_cond_score, fwd_t, self.dt)
        
        x_t_rvs = x_t_fwd
        for rvs_t in np.arange(1.0, 0, -self.dt):
            x_t_rvs = torch.tensor(x_t_rvs).float().to(self.device)
            cond_score = self.score_network((x_t_rvs, y), rvs_t)
            pred_score = cond_score
            x_t_rvs = self.reverse_ODE_step(x_t_rvs, pred_score, rvs_t, self.dt)
        
        x_0 = torch.tensor(x_t_rvs).to(self.device)

        mse = torch.mean((x - x_0) ** 2)
        self.log('val/mse', mse.item())
        return mse
    
class Pred_X_0_Parameterization(CondScoreModule):
    
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        t = np.random.uniform(self.min_t, 1.0)

        x_t, gt_score_t = self.diffuser.forward_marginal(x.detach().cpu().numpy(), t=t)

        score_scaling = torch.tensor(self.diffuser.score_scaling(t)).to(self.device)
        gt_score_t = torch.tensor(gt_score_t).to(self.device)

        if np.random.random() > 0.5:
            pred_x_0 = self((torch.tensor(x_t).float().to(self.device), y.to(self.device)), t)
        else:
            null_cond = torch.ones_like(y) * self.score_network.null_cond_idx
            pred_x_0 = self((torch.tensor(x_t).float().to(self.device), null_cond.to(self.device)), t)
            
        x_t_torch = torch.tensor(x_t).float().to(self.device)
        pred_score_t = self.diffuser.score(x_t=x_t_torch, x_0=pred_x_0, t=t, use_torch=True)

        score_mse = (gt_score_t - pred_score_t)**2
        score_loss = torch.sum(
            score_mse / score_scaling[None, None]**2,
            dim=(-1, -2)
        ) 

        self.log('train/score_mse_loss', score_loss.item())
        self.log('train/x_0_mse', torch.mean((x - pred_x_0) ** 2).item())
        return score_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        x_t_fwd = x
        for fwd_t in np.arange(0 + self.dt, 1.0, self.dt):
            x_t_fwd = torch.tensor(x_t_fwd).float().to(self.device)
            pred_x_0 = self.score_network((x_t_fwd, y), fwd_t)
            fwd_cond_score = self.diffuser.score(x_t=x_t_fwd, x_0=pred_x_0, t=fwd_t, use_torch=True)
            x_t_fwd = self.forward_ODE_step(x_t_fwd, fwd_cond_score, fwd_t, self.dt)
        
        x_t_rvs = x_t_fwd
        for rvs_t in np.arange(1.0, 0, -self.dt):
            x_t_rvs = torch.tensor(x_t_rvs).float().to(self.device)
            pred_x_0 = self.score_network((x_t_rvs, y), rvs_t)
            rvs_cond_score = self.diffuser.score(x_t=x_t_rvs, x_0=pred_x_0, t=rvs_t, use_torch=True)
            x_t_rvs = self.reverse_ODE_step(x_t_rvs, rvs_cond_score, rvs_t, self.dt)
        
        x_0 = torch.tensor(x_t_rvs).to(self.device)

        mse = torch.mean((x - x_0) ** 2)
        self.log('val/mse', mse.item())
        return mse