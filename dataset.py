# modified code from my RNN homework

from typing import Union, List, Tuple, Optional
import os
import torch
from torch.utils.data import Dataset
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor


class TextDataset(Dataset):

    def __init__(
        self,
        en_train_data_file: str,
        de_train_data_file: str,
        en_val_data_file: str,
        de_val_data_file: str,
        train: bool = True,
        sp_model_prefix: str = None,
        en_vocab_size: int = 6000,
        de_vocab_size: int = 6000,
        normalization_rule_name: str = "nmt_nfkc_cf",
        en_model_type: str = "word",
        de_model_type: str = "bpe",
        max_length: int = 64,
        data_ratio: Optional[float] = None,
    ):
        """
        Dataset with texts, supporting BPE tokenizer
        :param data_file: txt file containing texts
        :param train: whether to use train or validation split
        :param sp_model_prefix: path prefix to save tokenizer model
        :param vocab_size: sentencepiece tokenizer vocabulary size
        :param normalization_rule_name: sentencepiece tokenizer normalization rule
        :param model_type: sentencepiece tokenizer model type
        :param max_length: maximal length of text in tokens
        """

        if not os.path.isfile(sp_model_prefix + "-en.model"):
            # train tokenizer if not trained yet
            SentencePieceTrainer.train(
                input=en_train_data_file,
                vocab_size=en_vocab_size,
                model_type=en_model_type,
                model_prefix=sp_model_prefix + "-en",
                normalization_rule_name=normalization_rule_name,
                pad_id=3,
            )
        if not os.path.isfile(sp_model_prefix + "-de.model"):
            # train tokenizer if not trained yet
            SentencePieceTrainer.train(
                input=de_train_data_file,
                vocab_size=de_vocab_size,
                model_type=de_model_type,
                model_prefix=sp_model_prefix + "-de",
                normalization_rule_name=normalization_rule_name,
                pad_id=3,
            )
        # load tokenizer from file
        self.de_sp_model = SentencePieceProcessor(
            model_file=sp_model_prefix + "-de.model"
        )
        self.en_sp_model = SentencePieceProcessor(
            model_file=sp_model_prefix + "-en.model"
        )

        with open(
            en_train_data_file,
            encoding="utf-8",
        ) as file:
            en_train_texts = file.readlines()
            if data_ratio:
                en_train_texts = en_train_texts[: int(len(en_train_texts) * data_ratio)]

        with open(
            en_val_data_file,
            encoding="utf-8",
        ) as file:
            en_val_texts = file.readlines()

        with open(
            de_train_data_file,
            encoding="utf-8",
        ) as file:
            de_train_texts = file.readlines()
            if data_ratio:
                de_train_texts = de_train_texts[: int(len(de_train_texts) * data_ratio)]

        with open(
            de_val_data_file,
            encoding="utf-8",
        ) as file:
            de_val_texts = file.readlines()

        assert len(en_train_texts) == len(de_train_texts)
        assert len(en_val_texts) == len(de_val_texts)

        self.en_texts = en_train_texts if train else en_val_texts
        self.en_indices = self.en_sp_model.encode(self.en_texts)

        self.de_texts = de_train_texts if train else de_val_texts
        self.de_indices = self.de_sp_model.encode(self.de_texts)

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = (
            self.en_sp_model.pad_id(),
            self.en_sp_model.unk_id(),
            self.en_sp_model.bos_id(),
            self.en_sp_model.eos_id(),
        )
        self.max_length = max_length
        self.en_vocab_size = self.en_sp_model.vocab_size()
        self.de_vocab_size = self.de_sp_model.vocab_size()

    def en_text2ids(
        self, texts: Union[str, List[str]]
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        return self.en_sp_model.encode(texts)

    def de_text2ids(
        self, texts: Union[str, List[str]]
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        return self.de_sp_model.encode(texts)

    def en_ids2text(
        self, ids: Union[torch.Tensor, List[int], List[List[int]]]
    ) -> Union[str, List[str]]:
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert (
                len(ids.shape) <= 2
            ), "Expected tensor of shape (length, ) or (batch_size, length)"
            ids = ids.cpu().tolist()

        return self.en_sp_model.decode(ids)

    def de_ids2text(
        self, ids: Union[torch.Tensor, List[int], List[List[int]]]
    ) -> Union[str, List[str]]:
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert (
                len(ids.shape) <= 2
            ), "Expected tensor of shape (length, ) or (batch_size, length)"
            ids = ids.cpu().tolist()

        return self.de_sp_model.decode(ids)

    def __len__(self):
        """
        Size of the dataset
        :return: number of texts in the dataset
        """
        return len(self.en_indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add specials to the index array and pad to maximal length
        :param item: text id
        :return: (en_indices, en_length), (de_indices, de_length). Encoded text indices and its actual length (including BOS and EOS specials)
        """
        en_indices = [self.bos_id] + self.en_indices[item] + [self.eos_id]
        en_indices = torch.tensor(
            en_indices + [self.pad_id] * (self.max_length - len(en_indices))
        )[: self.max_length]

        de_indices = [self.bos_id] + self.de_indices[item] + [self.eos_id]
        de_indices = torch.tensor(
            de_indices + [self.pad_id] * (self.max_length - len(de_indices))
        )[: self.max_length]

        return de_indices, en_indices
