from model.SimCLR import SimCLR
from model.SimCLRv2 import SimCLRv2
from cifar import CIFAR10DataModule

import torch
from torch import set_float32_matmul_precision
from torch.utils.data import Subset, DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from json import loads
from pathlib import Path
from warnings import filterwarnings


filterwarnings("ignore")
set_float32_matmul_precision('medium')

CORESET_SELECT = True


if __name__ == '__main__':
    f = open("best_params.json", 'r')
    best_params = loads(f.read())
    
    # =========================
    # 1. Setup dataset
    # =========================
    dm = CIFAR10DataModule(data_dir="", batch_size=32,)
    dm.setup()

    del best_params['crop_size']
    del best_params['batch_size']

    # =========================
    # 2. Setup model
    # =========================
    model = SimCLR(
        **best_params,
        coreset_select=CORESET_SELECT
    )

    # =========================
    # 3. Fit trainer
    # =========================
    logger = TensorBoardLogger("tb_logs", name="CoresetSelect", default_hp_metric=True)

    trainer = Trainer(
       max_epochs=300, 
       logger=logger,
       enable_progress_bar=True, 
       gpus=1,
       precision='64',
       limit_test_batches=0
    )

    trainer.fit(model, dm)

    # =========================
    # 4. Get coreset
    # =========================
    loader, cossim_avg, cossim_hist, sorted_indices = dm.get_coreset()

    # save sorted dataloader, order is maintained even after saving
    torch.save(loader, 'coreset_dataloader.pth')
