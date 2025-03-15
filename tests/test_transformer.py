from ..light_transformer import LightningTransformer, WMTDataset

import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path

def main():
    pl.seed_everything(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get vocabulary size
    transformer = LightningTransformer(vocab_size=17546150, ninp=200, nhead=2, nhid=200, nlayers=3)
    transformer = transformer.to(device)
    """
    The Trainer automatically calls the appropriate methods in your LightningModule. 
    When trainer.fit(model) is called, PyTorch Lightning will:

        Call dataloader methods:
        It invokes train_dataloader() and val_dataloader() to get your training and validation data.

        Training loop:
        It automatically calls training_step() for each batch during training.

        Validation loop:
        It automatically calls validation_step() for each batch and then validation_epoch_end() at the end of the validation epoch.

        Optimizer configuration:
        It calls configure_optimizers() to set up the optimizers.
    """

    trainer = Trainer(
        max_epochs=10,
        accelerator=device,  # use 'cuda' if available; otherwise, use 'cpu'
        devices=1,          # number of GPUs, if available
        log_every_n_steps=100,
        logger=CSVLogger("csv_logs", name="transformer") # save logs to csv file in ogs/transformer/ directory.
    )
    trainer.fit(transformer)

if __name__ == "__main__":
    main()
