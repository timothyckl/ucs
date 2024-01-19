import os
import math
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as L
from pl_bolts.models.self_supervised.resnets import resnet50, resnet101
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
from torchmetrics.classification import Accuracy

from torch.optim import Adam
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay

import numpy as np


class Projection(nn.Module):
  def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim

    self.model = nn.Sequential(
        nn.Linear(self.input_dim, self.hidden_dim, bias=True),
        nn.BatchNorm1d(self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim, self.output_dim, bias=False)
    )

  def forward(self, x):
    x = self.model(x)
    return F.normalize(x, dim=1)

class SimCLR(L.LightningModule):
    def __init__(
            self, 
            arch: str = 'resnet50', 
            optimizer: str = 'lars',
            warmup_epochs: int = 10, 
            learning_rate: float = 1e-4, 
            thrust_coef: float = 1e-3, 
            weight_decay: float = 1e-6, 
            temperature: int = 0.5 ,
            coreset_select: bool = False
        ):
        super().__init__()
        self.save_hyperparameters()
        
        if arch == 'resnet50':
            self.encoder = resnet50(pretrained=False, return_all_feature_maps=False)
        elif arch == 'resnet101':
            self.encoder = resnet101(pretrained=False, return_all_feature_maps=False)
        else:
            raise NotImplementedError('Model not implemented.')

        self.projection = Projection()

        # follow imagenet adjustments
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def setup(self, stage):
        batch_size = self.trainer.datamodule.hparams.batch_size
        
        if stage == 'fit':
            self.n_train_samples = len(self.trainer.datamodule.train_dataloader()) * batch_size
            global_batch_size = self.trainer.world_size * batch_size
            self.train_iters_per_epoch = self.n_train_samples // global_batch_size

    def exclude_from_weight_decay(self, named_params, weight_decay, skip_list=["bias", "bn"]):
        params, excluded_params = [], []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

            return [
                {'params': params, 'weight_decay': weight_decay},
                {'params': excluded_params, 'weight_decay': 0.0}
            ]

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
            return [optimizer]
    
        elif self.hparams.optimizer == 'lars':
            # exclude certain parameters
            parameters = self.exclude_from_weight_decay(
                named_params=self.named_parameters(),
                weight_decay=self.hparams.weight_decay
            )

            optimizer = LARS(
                parameters,
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
                trust_coefficient=self.hparams.thrust_coef
            )

            warmup_steps = self.train_iters_per_epoch * self.hparams.warmup_epochs
            total_steps = self.train_iters_per_epoch * self.trainer.max_epochs

            linear_warmup_cosine_decay = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True)
            )

            scheduler = {
                'scheduler': linear_warmup_cosine_decay,
                'interval': 'step',
                'frequency': 1
            }

            return [optimizer], [scheduler]
        
        else:
            raise NotImplementedError("Optimizer not implemented.")

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1, out_2], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        return -torch.log(pos / (neg + eps)).mean()

    def shared_step(self, batch, _):
        (d_i, m_i), (d_j, m_j), _, batch_idx = batch

        # (b, 1, 512, 512) -> (b, 2048)
        # image features
        h_i = self.encoder(d_i)[-1]
        h_j = self.encoder(d_j)[-1]

        # mask features
        h_mi = self.encoder(m_i)[-1]
        h_mj = self.encoder(m_j)[-1]

        # handles error at validation step
        # epoch_cossim only exists during training step
        try:
            if self.hparams.coreset_select:
                batch_sim = F.cosine_similarity(h_i, h_j, dim=1, eps=1e-8)  # image representations only
                batch_sim_cpu = batch_sim.detach().cpu().numpy()

                start = batch_idx[0]
                end = batch_idx[-1] + 1
                self.epoch_cossim[start:end] += batch_sim_cpu
        except:
            pass

        # (b, 2048) -> (b, 128)
        # image projection
        z_i = self.projection(h_i)
        z_j = self.projection(h_j)

        # mask projection
        z_mi = self.projection(h_mi)
        z_mj = self.projection(h_mj)

        loss_1 = self.nt_xent_loss(z_i, z_j, self.hparams.temperature)
        loss_2 = self.nt_xent_loss(z_mi, z_mj, self.hparams.temperature)
        loss = loss_1 + loss_2
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def on_train_start(self):
        if self.hparams.coreset_select:
            # self.cossim_history = np.empty((self.trainer.max_epochs, self.n_train_samples))
            self.cossim_history = []

    def on_train_epoch_start(self):
        if self.hparams.coreset_select:
            self.epoch_cossim = np.zeros(shape=(self.n_train_samples))

    def on_train_epoch_end(self):
        # keep epoch's cossim history
        if self.hparams.coreset_select:
            self.epoch_cossim = self.epoch_cossim[self.epoch_cossim != 0]
            self.cossim_history.append(self.epoch_cossim)
        
    def on_train_end(self):
        # average accumulated cossim
        if self.hparams.coreset_select:
            self.cossim_history = np.array(self.cossim_history)
            avg_cossim = np.sum(self.cossim_history, axis=0) / self.trainer.max_epochs
            
            os.makedirs('./cossim_outputs', exist_ok=True)
            np.save('./cossim_outputs/cossim_avg.npy', avg_cossim)
            np.save('./cossim_outputs/cossim_history.npy', self.cossim_history)

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        # self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("hp_metric", loss, on_step=False, on_epoch=True)  
        return loss