import math
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as L
from .modules.resnets import get_resnet

from torch.optim import Adam
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay

class SimCLRv2(L.LightningModule):
    def __init__(
            self, 
            arch: str = 'resnet50', 
            optimizer: str = 'lars',
            warmup_epochs: int = 10, 
            learning_rate: float = 1e-4, 
            thrust_coef: float = 1e-3, 
            weight_decay: float = 1e-6, 
            temperature: int = 0.5 
        ):
        super().__init__()
        self.save_hyperparameters()

        if arch == 'resnet50':
            self.encoder, self.projection = get_resnet(50)
        elif arch == 'resnet101':
            self.encoder, self.projection = get_resnet(101)
        elif arch == 'resnet152':
            self.encoder, self.projection = get_resnet(152)
        elif arch.encoder == 'resnet200':
            self.encoder, self.projection = get_resnet(200)
        else:
            raise NotImplementedError('Model not implemented.')        

    def setup(self, stage):
        batch_size = self.trainer.datamodule.hparams.batch_size
        
        if stage == 'fit':
            n_train_samples = len(self.trainer.datamodule.train_dataloader()) * batch_size
            global_batch_size = self.trainer.world_size * batch_size
            self.train_iters_per_epoch = n_train_samples // global_batch_size

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

    def shared_step(self, batch, batch_idx):
        # img_1 and img_2 are augmentations of a single image
        (d_i, m_i), (d_j, m_j), _ = batch

        # (b, 1, 512, 512) -> (b, 2048)
        # image features
        h_i = self.encoder(d_i)
        h_j = self.encoder(d_j)

        # mask features
        h_mi = self.encoder(m_i)
        h_mj = self.encoder(m_j)

        # calc cossim here (only image)
        # sim = F.cosine_similarity(h_i, h_j, dim=1, eps=1e-8)

        # accumulate sim scores


        # (b, 2048) -> (b, 128)
        # image non-linear projection
        z_i = self.projection(h_i)
        z_j = self.projection(h_j)
        print(z_i)
        print(z_j)
        print()

        # mask non-linear projection
        z_mi = self.projection(h_mi)
        z_mj = self.projection(h_mj)
        # print(z_mi)
        # print(z_mj)
        # print()

        loss_1 = self.nt_xent_loss(z_i, z_j, self.hparams.temperature)
        loss_2 = self.nt_xent_loss(z_mi, z_mj, self.hparams.temperature)
        loss = loss_1 + loss_2
        
        # print(f'loss_1: {loss_1} + loss_2: {loss_2} = {loss}')
        # raise Exception('stop')

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    # def test_step(self):
    #     pass

    # def forward(self, ):
    #     pass