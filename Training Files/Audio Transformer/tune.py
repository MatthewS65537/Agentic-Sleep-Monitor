import pytorch_lightning as pl
import torch
from modules.AudioTransformer import *
from modules.Dataset import *
from torch.utils.data import DataLoader
import optuna

# Load data with weights_only=True to address the FutureWarning
MFCC_Tensors = torch.load("./data/MFCC_Tensors.pt")
Label_Tensors = torch.load("./data/Label_Tensors.pt")
MFCC_Val_Tensors = torch.load("./data/MFCC_Val_Tensors.pt")
Label_Val_Tensors = torch.load("./data/Label_Val_Tensors.pt")

train_split = int(MFCC_Tensors.shape[0] * 0.8)
X_train = MFCC_Tensors[:train_split]
y_train = Label_Tensors[:train_split]
X_dev = MFCC_Tensors[train_split:]
y_dev = Label_Tensors[train_split:]
X_test = MFCC_Val_Tensors
y_test = Label_Val_Tensors

def objective(trial: optuna.trial.Trial) -> float:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hparams = {
        "grokking": "None",
        "learning_rate": trial.suggest_float('learning_rate', 5e-5, 1e-3, log=True),
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": trial.suggest_float('scheduler_factor', 0.1, 0.8),
        "scheduler_patience": trial.suggest_int('scheduler_patience', 2, 10),
        "min_lr": trial.suggest_float('min_lr', 1e-7, 1e-5, log=True),
        "num_classes": 5,
        "model_name": "AudioTransformer",
        "precision" : "16-mixed",
        "weight_decay" : trial.suggest_float('weight_decay', 1e-5, 0.1, log=True),
        "beta_1" : trial.suggest_float('beta_1', 0.7, 0.9),
        "beta_2" : trial.suggest_float('beta_2', 0.8, 0.999),
        "bsz" : 256
    }

    model = AudioTransformer(hparams)

    bsz = hparams["bsz"]
    train_dataset = MFCCDataset(X_train, y_train)
    dev_dataset = MFCCDataset(X_dev, y_dev)
    test_dataset = MFCCDataset(X_test, y_test)
    data_module = AudioDataModule(train_dataset, dev_dataset, test_dataset, batch_size=bsz)

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        enable_progress_bar=False,
        max_epochs=20,
        log_every_n_steps=1,
        precision=hparams['precision'],
    )
    trainer.fit(model, datamodule=data_module)
    
    return trainer.callback_metrics["test/accuracy"].item()

study = optuna.create_study(direction="maximize", storage="sqlite:///db.sqlite3")
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))