import pytorch_lightning as pl
import torch
from modules.VisionTransformer import *
from modules.Dataset import *
from torch.utils.data import DataLoader
import optuna

# Load data with weights_only=True to address the FutureWarning
X_train = torch.load("./data/images_train.pt", weights_only=True)
y_train = torch.load("./data/labels_train.pt", weights_only=True)
X_test = torch.load("./data/images_test.pt", weights_only=True)
y_test = torch.load("./data/labels_test.pt", weights_only=True)

def objective(trial: optuna.trial.Trial) -> float:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hparams = {
        'num_classes': 4,
        'in_channels': 1,
        'patch_size': 8,
        'img_size': (96, 56),
        'd_model': 512,
        'num_layers': 6,
        'nhead': 16,
        'dim_feedforward' : 1024,
        'dropout': 0.25,
        'precision': '16-mixed',
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
        'scheduler' : 'Exponential Decay',
        'scheduler_step_size': trial.suggest_int('scheduler_step_size', 1, 20),
        'scheduler_gamma': trial.suggest_float('scheduler_gamma', 0.1, 0.99),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'beta_1': trial.suggest_float('beta_1', 0.8, 0.99),
        'beta_2': trial.suggest_float('beta_2', 0.85, 0.999),
        'bsz' : 4, # 16, technically
        'grokking': 'None',
    }

    model = ViTLightning(hparams)

    train_loader = DataLoader(ImageDataset(X_train, y_train), batch_size=hparams['bsz'], shuffle=True)
    val_loader = DataLoader(ImageDataset(X_test, y_test), batch_size=hparams['bsz'], shuffle=False)

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        enable_progress_bar=False,
        max_epochs=100,
        precision=hparams['precision'],
    )
    trainer.fit(model, train_loader, val_loader)
    
    return trainer.callback_metrics["val/accuracy"].item()

study = optuna.create_study(direction="maximize", storage="sqlite:///db.sqlite3")
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))