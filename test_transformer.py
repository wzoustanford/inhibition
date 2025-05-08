from light_transformer import LightningTransformer, WMTDataset, MOST_COMMON_WORDS

import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path
import pdb

torch.cuda.empty_cache()

def main(
        use_moe, 
        num_experts, 
        K, 
        use_glu, 
        run_epochs, 
):
    pl.seed_everything(42)

    # get vocabulary size
    # train_dataset = WMTDataset(Path("../data/news.2024.en.train.preprocessed.txt"))
    # vocab_size = 1143
    # pdb.set_trace()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    moec = {
        'use_moe': use_moe,
        'num_experts': num_experts,
        'K': K,
        'use_glu': use_glu,
        'run_epochs': run_epochs,
    }
    transformer = LightningTransformer(
        vocab_size=MOST_COMMON_WORDS+1, 
        ninp=50, 
        nhead=2, 
        nhid=50, 
        nlayers=2, 
        device=device, 
        moe_inh_config = moec,
    )
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
        max_epochs=run_epochs,
        accelerator="gpu",  # use 'gpu' if available; otherwise, use 'cpu'
        devices=1,          # number of GPUs, if available
        log_every_n_steps=100,
        logger=CSVLogger("csv_logs", name="transformer_moe_v1") # save logs to csv file in ogs/transformer/ directory.
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
    print(f"exp: epochs: {moec['run_epochs']}, use_moe: {moec['use_moe']}, num_experts:{moec['num_experts']}, K: {moec['K']}, use_glu: {moec['use_glu']}")

if __name__ == "__main__":
    use_moe = True 
    run_epochs_l = [3]
    num_experts = 5
    K = 3
    use_glu_l = [False, True] 
    for run_epochs in run_epochs_l: 
        for use_glu in use_glu_l:
            main(
                use_moe, 
                num_experts, 
                K, 
                use_glu, 
                run_epochs, 
            )

    use_moe = False 
    run_epochs_l = [3]
    num_experts = 1
    K = 1
    use_glu = False 
    for run_epochs in run_epochs_l: 
        main(
            use_moe, 
            num_experts, 
            K, 
            use_glu, 
            run_epochs, 
        )
