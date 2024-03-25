# https://pytorch.org/tutorials/beginner/translation_transformer.html

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import generate_square_subsequent_mask
from dataset import TextDataset


# function to generate output sequence using greedy algorithm
def batch_greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    memory = memory.to(device)
    ys = torch.ones(src.shape[0], 1).fill_(start_symbol).type(torch.long).to(device)
    for _ in range(max_len - 1):

        tgt_mask = (
            generate_square_subsequent_mask(ys.shape[0], device).type(torch.bool)
        ).to(device)
        print(memory.shape)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_words = torch.max(prob, dim=2)

        ys = torch.cat([ys, next_words], dim=1)
    return ys


def greedy_decode(model, src, src_mask, max_len, start_symbol, eos_id, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for _ in range(max_len - 1):
        memory = memory.to(device)
        tgt_mask = (
            generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)
        ).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == eos_id:
            break
    return ys


# actual function to translate input sentence into target language
def translate(
    model: torch.nn.Module, src_sentence: str, train_set, bos_id, eos_id, device
):
    model.eval()
    src = torch.tensor(train_set.de_text2ids(src_sentence)).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,
        src,
        src_mask,
        max_len=num_tokens + 5,
        start_symbol=bos_id,
        eos_id=eos_id,
        device=device,
    ).flatten()
    return (
        "".join(train_set.en_ids2text(tgt_tokens))
        .replace("<bos>", "")
        .replace("<eos>", "")
    )


def translate_file(
    model: torch.nn.Module,
    file_path: str,
    train_set,
    batch_size,
    device,
):
    translate_set = TextDataset(
        en_train_data_file="data/train.de-en.en",
        de_train_data_file="data/train.de-en.de",
        en_val_data_file="data/val.de-en.en",
        de_val_data_file=file_path,
        train=False,
        sp_model_prefix=f"word_{train_set.de_vocab_size}-{train_set.en_vocab_size}",
        en_vocab_size=train_set.en_vocab_size,
        de_vocab_size=train_set.de_vocab_size,
        data_ratio=1,
    )

    model.eval()
    ans = torch.Tensor()
    with torch.no_grad():

        translate_dataloader = DataLoader(
            translate_set,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
        )

        for src, _ in tqdm(translate_dataloader, desc="Translating for BLEU"):
            src = torch.transpose(src, 0, 1)

            num_tokens = src.shape[0]

            src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

            tgt_tokens = batch_greedy_decode(
                model,
                src,
                src_mask,
                max_len=num_tokens + 5,
                start_symbol=train_set.bos_id,
                device=device,
            ).flatten(start_dim=1)
            torch.cat(ans, tgt_tokens)

    return [
        "".join(train_set.en_ids2text(tokens)).replace("<bos>", "").replace("<eos>", "")
        for tokens in tgt_tokens
    ]
