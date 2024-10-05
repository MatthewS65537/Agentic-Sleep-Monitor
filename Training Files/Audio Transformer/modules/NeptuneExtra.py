# We don't need this, I think.

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning import Trainer

class UploadLastCheckpointCallback(Callback):
    def __init__(self, artifact_name="best_model"):
        self.artifact_name = artifact_name

    def on_train_end(self, trainer, pl_module):
        if not hasattr(trainer.checkpoint_callback, 'best_model_path'):
            print("Warning: No best model path found. No checkpoint will be uploaded.")
            return

        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            try:
                trainer.logger.experiment[self.artifact_name].upload_files(best_model_path)
                print(f"Successfully uploaded the best model checkpoint: {best_model_path}")
            except Exception as e:
                print(f"Error uploading checkpoint: {e}")
        else:
            print("No best model checkpoint found to upload.")