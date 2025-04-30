from light_transformer_s import LightningTransformer, WMTDataset

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
#    LightningTransformer(vocab_size=601, ninp=200, nhead=2, nhid=200, nlayers=2).val_dataloader()
    transformer = LightningTransformer(vocab_size=10001, ninp=200, nhead=2, nhid=200, nlayers=2)
#    transformer = transformer.to(device)
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

#    trainer = Trainer(
#        max_epochs=50,
#        accelerator="gpu",  # use 'gpu' if available; otherwise, use 'cpu'
#        devices=1,          # number of GPUs, if available
#        log_every_n_steps=100,
#        logger=CSVLogger("csv_logs", name="transformer_small") # save logs to csv file in ogs/transformer/ directory.
#    )
#    trainer.fit(transformer)

    transformer = LightningTransformer.load_from_checkpoint(
        "/home/ubuntu/inhibition/tests/csv_logs/transformer_small/version_28/checkpoints/epoch=49-step=1350.ckpt",
        vocab_size=10001, 
        ninp=200, 
        nhead=2, 
        nhid=200, 
        nlayers=2)
    transformer.eval()

    prompt = "I will have breakfast today and go "
    print("Generated text:", transformer.generate(prompt, 50, 1.5))
if __name__ == "__main__":
    main()
