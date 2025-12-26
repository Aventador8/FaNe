import logging
import os
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only



def setup_logger(log_file: str = "training.log"):
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)


    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger



class LogLossToFileCallback(Callback):
    def __init__(self, log_file: str = "training.log"):
        super().__init__()
        self.logger = setup_logger(log_file)


    @rank_zero_only
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs, batch, batch_idx):


        if isinstance(outputs, dict):
            log_message = ", ".join([f"{k}: {v.item():.4f}" if torch.is_tensor(v) else f"{k}: {v:.4f}" for k, v in outputs.items()])
            self.logger.info(f"Batch {batch_idx}: {log_message}")


