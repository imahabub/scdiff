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
            null_cond = torch.zeros_like(y)
            pred_score_t = self((torch.tensor(x_t).float().to(self.device), null_cond.to(self.device)), t)

        score_mse = (gt_score_t - pred_score_t)**2
        score_loss = torch.sum(
            score_mse / score_scaling[None, None]**2,
            dim=(-1, -2)
        ) 

        self.log('Training loss', score_loss.item())
        return score_loss

    def configure_optimizers(self):
        # Your optimizer configuration
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.experiment.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        # Your evaluation code
        # mses = []
        x, y = batch
        x_t, _ = self.diffuser.forward_marginal(x.detach().cpu().numpy(), t=1.0)
        
        for i, t in enumerate(np.arange(1.0, 0, -self.dt)):
            x_t = torch.tensor(x_t).float().to(self.device)
            pred_score = self.score_network((x_t, y), t)
            
            x_t = self.diffuser.reverse(x_t=x_t.detach().cpu().numpy(), score_t=pred_score.detach().cpu().numpy(), t=t, dt=self.dt, center=False)
        
        x_0 = torch.tensor(x_t).to(self.device)

        mse = torch.mean((x - x_0) ** 2)
        # mses.append(mse.item())
        # eval_mse = np.mean(mses)
        # writer.add_scalar('MSE', eval_mse, global_step=step)
        return mse
