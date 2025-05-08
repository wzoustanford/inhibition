"""Demo of a simple transformer language model.

Code is adapted from the PyTorch examples at
https://github.com/pytorch/examples/blob/main/word_language_model

"""

import math
import os
from pathlib import Path
from typing import Optional

import torch, pdb, random
import torch.nn as nn
import torch.nn.functional as F
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn.modules import MultiheadAttention
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningModule
from moe_transformer import MOETransformerDecoderLayer

_REQUESTS_AVAILABLE = RequirementCache("requests")
MOST_COMMON_WORDS = 15000


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
        vocab_size: int = MOST_COMMON_WORDS + 1,  # default for WikiText2
        ninp: int = 200,
        nhead: int = 2,
        nhid: int = 200,
        nlayers: int = 2,
        dropout: float = 0.2,
        device: torch.device = None,
        moe_inh_config: dict = {},
    ) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.embedding = nn.Embedding(vocab_size, ninp)
        if moe_inh_config['use_moe']:
            custom_decoder_layer = MOETransformerDecoderLayer(
                d_model=ninp, 
                nhead=nhead,
                dim_feedforward=nhid,
                dropout=dropout,
                batch_first=True,
                device=device,
                moe_inh_config=moe_inh_config,
            )
            custom_decoder = nn.TransformerDecoder(decoder_layer=custom_decoder_layer, num_layers=nlayers)
        else: 
            custom_decoder = None 
        
        self.transformer = nn.Transformer(
            d_model=ninp,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            dim_feedforward=nhid,
            dropout=dropout,
            custom_decoder=custom_decoder,
            batch_first=True,
        )
        self.decoder = nn.Linear(ninp, vocab_size)

        self.ninp = ninp
        self.vocab_size = vocab_size
        self.src_mask = None

    def forward(self, inputs: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        _, t = target.shape

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

from torch.utils.data import IterableDataset, get_worker_info
from itertools import islice
import torch
from pathlib import Path

class WMTIterableDataset(IterableDataset):
    def __init__(self, file_path: Path, vocab, dictionary, block_size: int = 10) -> None:
        """
        Args:
            file_path (Path): Path to the text file.
            vocab: A vocabulary object (e.g., a dict-like structure) to check for known words.
            dictionary: A Dictionary instance with a word2idx attribute.
            block_size (int): Number of tokens per block.
        """
        self.file_path = file_path
        self.vocab = vocab
        self.dictionary = dictionary
        self.block_size = block_size

    def _line_iterator(self):
        # Open the file and yield one line at a time.
        with open(self.file_path, encoding="utf8") as f:
            for line in f:
                yield line

    def __iter__(self):
        # For multi-worker setup: each worker processes a slice of the file.
        worker_info = get_worker_info()
        if worker_info is None:
            file_iter = self._line_iterator()
        else:
            # Split the file lines among workers using islice
            file_iter = islice(self._line_iterator(), worker_info.id, None, worker_info.num_workers)
        
        buffer = []  # Accumulates token indices
        for line in file_iter:
            # Tokenize: add start and end tokens, split on whitespace, and replace unknown words.
            words = ["<S>"] + line.strip().split() + ["</S>"]
            words = [word if word in self.dictionary.word2idx else "<UNK>" for word in words]
            # Convert words to indices using the dictionary.
            indices = [self.dictionary.word2idx[word] for word in words if word in self.dictionary.word2idx]
            buffer.extend(indices)
            # Yield blocks once we have enough tokens.
            while len(buffer) >= self.block_size + 1:
                #inputs = buffer[:self.block_size]
                inputs = buffer[:1]
                targets = buffer[1:self.block_size + 1]
                yield torch.tensor(inputs, dtype=torch.int64), torch.tensor(targets, dtype=torch.int64)
                # Remove the tokens that were yielded.
                buffer = buffer[self.block_size:]

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
    def __init__(self, file_path: Path, vocab, dictionary, block_size: int = 10) -> None:
        # Assume the file is a plain text file containing English text
        self.data, self.dictionary = tokenize(file_path, vocab, dictionary)
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

class ShuffleWMTIterableDataset(IterableDataset):
    def __init__(self, file_path: Path, vocab, dictionary, block_size: int = 10, shuffle: bool = False, shuffle_buffer_size: int = 1000) -> None:
        """
        Args:
            file_path (Path): Path to the text file.
            vocab: A vocabulary object (e.g., a dict-like structure) to check for known words.
            dictionary: A Dictionary instance with a word2idx attribute.
            block_size (int): Number of tokens per block.
            shuffle (bool): Whether to shuffle the dataset.
            shuffle_buffer_size (int): Size of the shuffle buffer.
        """
        self.file_path = file_path
        self.vocab = vocab
        self.dictionary = dictionary
        self.block_size = block_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size

    def _line_iterator(self):
        with open(self.file_path, encoding="utf8") as f:
            for line in f:
                yield line

    def _block_iterator(self, file_iter):
        buffer = []
        for line in file_iter:
            words = ["<S>"] + line.strip().split() + ["</S>"]
            words = [word if word in self.dictionary.word2idx else "<UNK>" for word in words]
            indices = [self.dictionary.word2idx[word] for word in words if word in self.dictionary.word2idx]
            buffer.extend(indices)

            while len(buffer) >= self.block_size + 1:
                inputs = buffer[:self.block_size]
                targets = buffer[1:self.block_size + 1]
                yield torch.tensor(inputs, dtype=torch.int64), torch.tensor(targets, dtype=torch.int64)
                buffer = buffer[self.block_size:]

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            file_iter = self._line_iterator()
        else:
            file_iter = islice(self._line_iterator(), worker_info.id, None, worker_info.num_workers)

        block_iter = self._block_iterator(file_iter)

        if self.shuffle:
            # Implement a shuffle buffer
            shuffle_buffer = []
            try:
                # Fill the shuffle buffer initially
                for _ in range(self.shuffle_buffer_size):
                    shuffle_buffer.append(next(block_iter))
            except StopIteration:
                pass  # Handle smaller datasets

            random.shuffle(shuffle_buffer)

            while shuffle_buffer:
                yield shuffle_buffer.pop()
                try:
                    shuffle_buffer.append(next(block_iter))
                    if len(shuffle_buffer) % self.shuffle_buffer_size == 0:
                        random.shuffle(shuffle_buffer)
                except StopIteration:
                    continue
        else:
            yield from block_iter

class Dictionary:
    """
    A dictionary class for mapping words to indices.
    """
    def __init__(self) -> None:
        self.word2idx: dict[str, int] = {} # Mapping from words to indices
        self.idx2word: list[str] = [] # A list of tokenized words in the order they were added

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.idx2word)

class Vocabulary:
    """
    A dictionary class for mapping words to count their occurrences.
    """
    def __init__(self):
        self.word2count = {}
    
    def add_word(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
    def most_common(self, n):
        return dict(sorted(self.word2count.items(), key=lambda x: x[1], reverse=True)[:n])


def tokenize(path: Path, vocab, dictionary):
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
    # dictionary = Dictionary()

    # assert os.path.exists(path)
    # # Add words to the dictionary
    # with open(path, encoding="utf8") as f:
    #     for line in f:
    #         words = ["<S>"] + line.split(" ") + ["</S>"]
    #         for word in words:
    #             dictionary.add_word(word)

    # # Save dictionary_count top 40000 to a file
    # with open("dictionary_count.txt", "w") as f:
    #     # Sort the dictionary by count
    #     vocab_count.word2count = dict(sorted(vocab_count.word2count.items(), key=lambda item: item[1], reverse=True))

    #     for i in range(30000):
    #         f.write(f"{list(vocab_count.word2count.keys())[i]}: {list(vocab_count.word2count.values())[i]}\n")

    # Tokenize file content
    
    with open(path, encoding="utf8") as f:
        idss: list[Tensor] = []
        for line in f:
            words = ["<S>"] + line.strip().split(" ") + ["</S>"]
            # Replace words not in the vocab_count with <UNK>
            words = [word if word in vocab else "<UNK>" for word in words]
            ids: list[int] = []
            for word in words:
                ids.append(dictionary.word2idx[word]) # Get the index of the word
            idss.append(torch.tensor(ids).type(torch.int64)) # Convert the list of indices to a tensor

    return torch.cat(idss), dictionary


class LightningTransformer(LightningModule):
    def __init__(
            self, 
            vocab_size: int, 
            ninp: int, 
            nhead: int, 
            nhid: int, 
            nlayers: int, 
            device: str,
            moe_inh_config: dict,
        ) -> None:
        super().__init__()
        self.vocab = Vocabulary()
        self.dictionary = Dictionary()
        self.build_vocab(Path("/home/ubuntu/inhibition/data/news.2024.small.txt"))
        #self.build_vocab(Path("/home/ubuntu/inhibition/data/news.2024.en.train.preprocessed.txt"))
        self.keep_dict()
        self.validation_step_outputs = []
        self.model = Transformer(
            vocab_size=vocab_size, 
            ninp=ninp, 
            nhead=nhead, 
            nhid=nhid, 
            nlayers=nlayers,
            device=device,
            moe_inh_config=moe_inh_config,
        )

    def build_vocab(self, path: Path) -> None:
        with open(path, encoding="utf8") as f:
            for line in f:
                words = ["<S>"] + line.strip().split(" ") + ["</S>"]
                for word in words:
                    self.vocab.add_word(word)
                    self.dictionary.add_word(word)

    def keep_dict(self) -> None:
        # top_words is the dict of your 50k most frequent words
        top_words_set = set(self.vocab.most_common(MOST_COMMON_WORDS).keys())

        # Create fresh containers
        new_word2idx = {}
        new_idx2word = []

        for word in self.dictionary.idx2word:
            if word in top_words_set:
                new_word2idx[word] = len(new_idx2word)
                new_idx2word.append(word)

        # Make sure <UNK> is included and has a valid index
        if "<UNK>" not in new_word2idx:
            new_word2idx["<UNK>"] = len(new_idx2word)
            new_idx2word.append("<UNK>")

        # Overwrite the old dictionary with newly indexed data
        self.dictionary.word2idx = new_word2idx
        self.dictionary.idx2word = new_idx2word
    


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
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        loss = self.validation_step_outputs[-1]
        #torch.stack(self.validation_step_outputs).mean()
        # Perplexity is the exponentiation of the average loss
        perplexity = torch.exp(loss)
        self.log("val_perplexity", perplexity, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
        #torch.optim.SGD(self.model.parameters(), lr=0.1)

    def train_dataloader(self) -> DataLoader:
        #train_dataset = WMTIterableDataset(Path("/home/ubuntu/inhibition/data/news.2024.en.train.preprocessed.txt"), self.vocab.most_common(MOST_COMMON_WORDS), self.dictionary)
        #train_dataset = ShuffleWMTIterableDataset(Path("/home/ubuntu/inhibition/data/news.2024.en.train.preprocessed.txt"), self.vocab.most_common(MOST_COMMON_WORDS), self.dictionary, shuffle=True, shuffle_buffer_size=1000)
        train_dataset = ShuffleWMTIterableDataset(Path("/home/ubuntu/inhibition/data/news.2024.small.txt"), self.vocab.most_common(MOST_COMMON_WORDS), self.dictionary, shuffle=True, shuffle_buffer_size=1000)
        return DataLoader(train_dataset, batch_size=256)

    def val_dataloader(self) -> DataLoader:
        #val_dataset = WMTIterableDataset(Path("/home/ubuntu/inhibition/data/news.2024.en.valid.preprocessed.txt"), self.vocab.most_common(MOST_COMMON_WORDS), self.dictionary)
        #val_dataset = ShuffleWMTIterableDataset(Path("/home/ubuntu/inhibition/data/news.2024.en.valid.preprocessed.txt"), self.vocab.most_common(MOST_COMMON_WORDS), self.dictionary, shuffle=False)
        val_dataset = ShuffleWMTIterableDataset(Path("/home/ubuntu/inhibition/data/news.2024.small.valid.txt"), self.vocab.most_common(MOST_COMMON_WORDS), self.dictionary, shuffle=False)
        return DataLoader(val_dataset, batch_size=5120)

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 1.0, device: str = 'cuda'):
        self.eval()
        self.to(device)
        words = ["<S>"] + prompt.strip().lower().split()
        # if work in vocab, get the index, else use index of <UNK>
        indices = [self.dictionary.word2idx.get(word, self.dictionary.word2idx["<UNK>"]) for word in words]
        #inputs = torch.tensor(indices[:-1]).unsqueeze(0).to(device)  # [1, seq_len]
        inputs = torch.tensor(indices[:1]).unsqueeze(0).to(device)  # [1, seq_len]
        # Decoder input alwasy starts with <S> token
        #generated_indices = inputs.tolist()[0] # start from <S> token
        generated_indices = indices[1:]
        for _ in range(max_tokens):
            generated_indices_tensor = torch.tensor(generated_indices).unsqueeze(0).to(device)          
            outputs = self.model(inputs, generated_indices_tensor)
            
            logits = outputs[-1] / temperature
            probabilities = torch.softmax(logits, dim=-1)
            
            next_token_idx = torch.multinomial(probabilities, num_samples=1, replacement=True).item()
            generated_indices.append(next_token_idx)

            # Stop generation if end token is produced
            if self.dictionary.idx2word[next_token_idx] == "</S>":
                break

        generated_words = [self.dictionary.idx2word[idx] for idx in generated_indices]

        # Skip the initial <S> token when joining
        return ' '.join(generated_words[1:])
