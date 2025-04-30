from light_transformer import LightningTransformer, WMTDataset, MOST_COMMON_WORDS

import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path
import pdb

torch.cuda.empty_cache()

def main():
    pl.seed_everything(42)

    # get vocabulary size
    # train_dataset = WMTDataset(Path("../data/news.2024.en.train.preprocessed.txt"))
#    vocab_size = 1143
#    pdb.set_trace()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transformer = LightningTransformer(vocab_size=MOST_COMMON_WORDS+1, ninp=50, nhead=2, nhid=50, nlayers=2, device=device)
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
        accelerator="gpu",  # use 'gpu' if available; otherwise, use 'cpu'
        devices=1,          # number of GPUs, if available
        log_every_n_steps=100,
        logger=CSVLogger("csv_logs", name="transformer_v1") # save logs to csv file in ogs/transformer/ directory.
    )
    trainer.fit(transformer)
    """
    transformer = LightningTransformer.load_from_checkpoint(
       "/home/ubuntu/inhibition/tests/csv_logs/transformer_full/version_20/checkpoints/epoch=4-step=925260.ckpt",\
       vocab_size=10001, ninp=100, nhead=2, nhid=100, nlayers=2)
    transformer.eval()
    """

    prompt = "The government announced today that"
    print("Generated text:", transformer.generate(prompt, 50, temperature=3.0, device=device))

if __name__ == "__main__":
    main()
