"""Demo of a simple transformer language model.

Code is adapted from the PyTorch examples at
https://github.com/pytorch/examples/blob/main/word_language_model

"""

import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn.modules import MultiheadAttention
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningModule

_REQUESTS_AVAILABLE = RequirementCache("requests")


if hasattr(MultiheadAttention, "_reset_parameters") and not hasattr(MultiheadAttention, "reset_parameters"):
    # See https://github.com/pytorch/pytorch/issues/107909
    MultiheadAttention.reset_parameters = MultiheadAttention._reset_parameters


class Transformer(nn.Module):
    """
    A Transformer model for sequence-to-sequence tasks.

    Args:
        vocab_size (int): Size of the vocabulary. Default is 33278.
        ninp (int): Dimensionality of the input embeddings. Default is 200.
        nhead (int): Number of attention heads. Default is 2.
        nhid (int): Dimensionality of the feedforward network model. Default is 200.
        nlayers (int): Number of layers in the encoder and decoder. Default is 2.
        dropout (float): Dropout probability. Default is 0.2.

    Attributes:
        pos_encoder (PositionalEncoding): Positional encoding layer.
        embedding (nn.Embedding): Embedding layer for input tokens.
        transformer (nn.Transformer): Transformer model.
        decoder (nn.Linear): Linear layer to map transformer output to vocabulary size.
        ninp (int): Dimensionality of the input embeddings.
        vocab_size (int): Size of the vocabulary.
        src_mask (Optional[Tensor]): Source mask for the transformer.

    Methods:
        forward(inputs: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
            Forward pass of the transformer model.
            Args:
                inputs (Tensor): Input tensor of shape (batch_size, seq_len).
                target (Tensor): Target tensor of shape (batch_size, seq_len).
                mask (Optional[Tensor]): Mask tensor for the target sequence. Default is None.
            Returns:
                Tensor: Output tensor of shape (batch_size * seq_len, vocab_size).
    """
    def __init__(
        self,
        vocab_size: int = 33278,  # default for WikiText2
        ninp: int = 200,
        nhead: int = 2,
        nhid: int = 200,
        nlayers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.embedding = nn.Embedding(vocab_size, ninp)
        self.transformer = nn.Transformer(
            d_model=ninp,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.Linear(ninp, vocab_size)

        self.ninp = ninp
        self.vocab_size = vocab_size
        self.src_mask = None

    def forward(self, inputs: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        _, t = inputs.shape

        # we assume target is already shifted w.r.t. inputs
        if mask is None:
            mask = torch.tril(torch.ones(t, t, device=inputs.device)) == 1
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)

        src = self.pos_encoder(self.embedding(inputs) * math.sqrt(self.ninp))
        target = self.pos_encoder(self.embedding(target) * math.sqrt(self.ninp))
        output = self.transformer(src, target, tgt_mask=mask)
        output = self.decoder(output)
        output = F.log_softmax(output, dim=-1)
        # output of the model represnts the probability of the next token (of all vocab_size tokens)
        output = output.view(-1, self.vocab_size)
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for adding positional information to input tensors.

    Args:
        dim (int): The dimension of the input tensor.
        dropout (float, optional): Dropout rate. Default is 0.1.
        max_len (int, optional): Maximum length of the input sequence. Default is 5000.

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        dim (int): Dimension of the input tensor.
        max_len (int): Maximum length of the input sequence.
        pe (Optional[Tensor]): Positional encoding tensor.

    Methods:
        forward(x: Tensor) -> Tensor:
            Adds positional encoding to the input tensor and applies dropout.

        _init_pos_encoding(device: torch.device) -> Tensor:
            Initializes the positional encoding tensor.
    """
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.max_len = max_len
        self.pe: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        if self.pe is None:
            # 1) can't use buffer, see https://github.com/pytorch/pytorch/issues/68407
            # 2) can't use parameter becauses pe gets sliced and DDP requires all params to participate in forward
            # TODO: Could make this a `nn.Parameter` with `requires_grad=False`
            self.pe = self._init_pos_encoding(device=x.device)

        x = x + self.pe[:, x.size(1)]
        return self.dropout(x)

    def _init_pos_encoding(self, device: torch.device) -> Tensor:
        pe = torch.zeros(self.max_len, self.dim, device=device)
        position = torch.arange(0, self.max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device).float() * (-math.log(10000.0) / self.dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe


class WMTDataset(Dataset):
    """
    A PyTorch Dataset class for processing and batching text data from a file.

    Attributes:
        data (List[int]): Tokenized text data.
        dictionary (Dict[str, int]): Mapping from tokens to their corresponding indices.
        block_size (int): The size of each data block.

    Methods:
        vocab_size() -> int:
            Returns the size of the vocabulary.
        
        __len__() -> int:
            Returns the number of data blocks in the dataset.
        
        __getitem__(index: int) -> tuple[Tensor, Tensor]:
            Returns a tuple containing the input and target tensors for the given index.
    """
    def __init__(self, file_path: Path, block_size: int = 35) -> None:
        # Assume the file is a plain text file containing English text
        self.data, self.dictionary = tokenize(file_path)
        self.block_size = block_size

    @property
    def vocab_size(self) -> int:
        return len(self.dictionary)

    def __len__(self) -> int:
        return len(self.data) // self.block_size - 1

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = index * self.block_size
        end = start + self.block_size
        inputs = self.data[start:end]
        target = self.data[start+1:end+1]
        return inputs, target



class Dictionary:
    def __init__(self) -> None:
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.idx2word)


def tokenize(path: Path) -> tuple[Tensor, Dictionary]:
    """
    Tokenizes the text file at the given path and returns a tuple containing the tokenized tensor and the dictionary.

    Args:
        path (Path): The path to the text file to be tokenized.

    Returns:
        tuple[Tensor, Dictionary]: A tuple containing:
            - Tensor: A tensor of tokenized words from the text file.
            - Dictionary: A dictionary object containing the mapping of words to indices.

    Raises:
        AssertionError: If the file at the given path does not exist.
    """
    dictionary = Dictionary()

    assert os.path.exists(path)
    # Add words to the dictionary
    with open(path, encoding="utf8") as f:
        for line in f:
            words = line.split() + ["<eos>"]
            for word in words:
                dictionary.add_word(word)

    # Tokenize file content
    with open(path, encoding="utf8") as f:
        idss: list[Tensor] = []
        for line in f:
            words = line.split() + ["<eos>"]
            ids: list[int] = []
            for word in words:
                ids.append(dictionary.word2idx[word])
            idss.append(torch.tensor(ids).type(torch.int64))

    return torch.cat(idss), dictionary


class LightningTransformer(LightningModule):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        return self.model(inputs, target)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, target = batch
        # pass the outputs and targets to the model
        output = self(inputs, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, target = batch
        output = self(inputs, target)
        loss = F.nll_loss(output, target.view(-1))
        # Log the raw validation loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs: list[Tensor]) -> None:
        avg_loss = torch.stack(outputs).mean()
        # Perplexity is the exponentiation of the average loss
        perplexity = torch.exp(avg_loss)
        self.log("val_perplexity", perplexity, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

    def train_dataloader(self) -> DataLoader:
        train_dataset = WMTDataset(Path("./data/news.2024.en.train.txt"))
        return DataLoader(train_dataset, batch_size=32, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        val_dataset = WMTDataset(Path("./data/news.2024.en.valid.txt"))
        return DataLoader(val_dataset, batch_size=32)
