from model.SimCLR import SimCLR
from model.SimCLRv2 import SimCLRv2
from dataset import KMTDataModule

from json import dump
from pathlib import Path
from flash import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from optuna import create_study
from optuna.trial import Trial
from optuna.pruners import MedianPruner, PatientPruner

# https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html
from optuna.visualization.matplotlib import plot_optimization_history, \
    plot_parallel_coordinate, plot_parallel_coordinate, plot_param_importances

import matplotlib.pyplot as plt

from torch import set_float32_matmul_precision
from warnings import filterwarnings

filterwarnings("ignore")
set_float32_matmul_precision('high')
plt.rcParams["axes.grid"] = False

MAX_TRIALS = 50
EPOCH_PER_TRAIL = 100


def objective(trial: Trial):
    # try larger batch size and smaller crop size
    crop_size = trial.suggest_categorical(name="crop_size", choices=[(2**i, 2**i) for i in range(4, 7)])
    batch_size = trial.suggest_categorical(name="batch_size", choices=[2**i for i in range(5, 8)])
    arch = trial.suggest_categorical(name="arch", choices=['resnet50'])

    # crop_size = trial.suggest_categorical(name="crop_size", choices=[(2**i, 2**i) for i in range(4, 8)])  # [(16, 16), (32, 32), (64, 64), (128, 128)]
    # batch_size = trial.suggest_categorical(name="batch_size", choices=[2**i for i in range(4, 7)])  # [16, 32, 64]
    # arch = trial.suggest_categorical(name="arch", choices=['resnet50', 'resnet101'])

    optimizer = trial.suggest_categorical(name="optimizer", choices=['adam'])
    # optimizer = trial.suggest_categorical(name="optimizer", choices=['lars', 'adam'])
    learning_rate = trial.suggest_uniform(name="learning_rate", low=1e-5, high=1e-3)  # 0.00001 to 0.001
    temperature = trial.suggest_categorical(name="temperature", choices=[i / 10 for i in range(1, 10)])
    
    # if optimizer == 'lars':
    #     weight_decay = trial.suggest_uniform(name="weight_decay", low=1e-8, high=1e-5) 
    #     thrust_coef = trial.suggest_uniform(name="thrust_coef", low=1e-5, high=1e-2)
    # else:
    #     weight_decay = None
    #     thrust_coef = None

    weight_decay = None
    thrust_coef = None

    dm = KMTDataModule(data_dir=Path('./KMT-data'), crop_size=crop_size, batch_size=batch_size)

    model = SimCLR(
        arch=arch,
        optimizer=optimizer,
        learning_rate=learning_rate,
        thrust_coef=thrust_coef,
        weight_decay=weight_decay,
        temperature=temperature,
        warmup_epochs=0
    )

    callbacks = [
        ModelCheckpoint(
            dirpath='./checkpoints', 
            filename='{epoch}-{val_loss:.2f}', 
            monitor='val_loss',
        ), 
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(refresh_rate=1)
    ]

    logger = TensorBoardLogger("tb_logs", name="SimCLR", default_hp_metric=True)

    trainer = Trainer(
        max_epochs=EPOCH_PER_TRAIL,
        callbacks=callbacks, 
        logger=logger,
        enable_progress_bar=True, 
        gpus=1,
        precision='64'
    )

    trainer.fit(model, dm)

    return trainer.callback_metrics['val_loss'].item()

def save_plots(study, plot_dir: Path):
    import os 
    os.makedirs(plot_dir, exist_ok=True)

    # plot optimzation history
    plot_optimization_history(study, target_name='Objective Value (val_loss)')
    plt.tight_layout()
    plt.savefig(plot_dir / 'optimization_history.png')
    plt.close()
    
    # plot high-dimensional parameter relationships.
    plot_parallel_coordinate(study, target_name='Objective Value (val_loss)')
    plt.tight_layout()
    plt.savefig(plot_dir / 'param_relationships.png')
    plt.close()

    # # plot parameter importance
    # plot_param_importances(study)
    # plt.tight_layout()
    # plt.savefig(plot_dir / 'param_importance.png')
    # plt.close()

    print('plots saved!')

if __name__ == '__main__':
    # =========================
    # Hyperparameter Search
    # =========================
    pruner = PatientPruner(
        MedianPruner(
            n_startup_trials=5, 
            n_warmup_steps=0,
            interval_steps=1, 
            n_min_trials=1
        ), 
        patience=3
    )

    study = create_study(
        study_name='SimCLR Hyper-Parameter Search',
        direction='minimize',
        pruner=pruner
    )
    study.optimize(objective, n_trials=MAX_TRIALS)

    save_plots(study, Path('./plots'))

    # export best params
    out_file = open("best_params.json", "w")
    dump(study.best_params, out_file, indent=4)
    out_file.close()