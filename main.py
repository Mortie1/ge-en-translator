# https://pytorch.org/tutorials/beginner/translation_transformer.html

import torch
from torch import nn

import wandb

from tqdm import tqdm

from model import Seq2SeqTransformer
from dataset import TextDataset
from train import train
from inference import translate
from torch.optim.lr_scheduler import (
    LinearLR,
    SequentialLR,
    CosineAnnealingWarmRestarts,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

wandb.login()


epochs = 50
embed_size = 300
batch_size = 340
max_length = 81

src_vocab_size = 12000
tgt_vocab_size = 18000
n_heads = 10
ffn_hid_dim = 2048
num_encoder_layers = 7
num_decoder_layers = 3
dropout = 0.3

lr = 0.0005
betas = (0.9, 0.98)


run = wandb.init(
    # Set the project where this run will be logged
    project="bhw-2-translator",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "betas": betas,
        "epochs": epochs,
        "embed_size": embed_size,
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "n_heads": n_heads,
        "batch_size": batch_size,
        "ffn_hid_dim": ffn_hid_dim,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "max_length": max_length,
        "dropout": dropout,
        "architecture": "Transformer",
    },
)


train_set = TextDataset(
    en_train_data_file="./data/train.de-en.en",
    de_train_data_file="./data/train.de-en.de",
    en_val_data_file="./data/val.de-en.en",
    de_val_data_file="./data/val.de-en.de",
    train=True,
    sp_model_prefix=f"word_{src_vocab_size}-{tgt_vocab_size}",
    en_vocab_size=tgt_vocab_size,
    de_vocab_size=src_vocab_size,
)
valid_set = TextDataset(
    en_train_data_file="./data/train.de-en.en",
    de_train_data_file="./data/train.de-en.de",
    en_val_data_file="./data/val.de-en.en",
    de_val_data_file="./data/val.de-en.de",
    train=False,
    sp_model_prefix=f"word_{src_vocab_size}-{tgt_vocab_size}",
    en_vocab_size=tgt_vocab_size,
    de_vocab_size=src_vocab_size,
)

torch.manual_seed(0)


transformer = Seq2SeqTransformer(
    num_encoder_layers,
    num_decoder_layers,
    embed_size,
    n_heads,
    src_vocab_size,
    tgt_vocab_size,
    ffn_hid_dim,
    dropout=dropout,
)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)
optimizer = torch.optim.Adam(transformer.parameters(), lr=lr, betas=betas, eps=1e-9)
scheduler1 = LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=1500)
scheduler2 = CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)
scheduler = SequentialLR(
    optimizer, schedulers=[scheduler1, scheduler2], milestones=[1500]
)

# )
train(
    transformer,
    optimizer,
    train_set,
    valid_set,
    epochs,
    batch_size,
    device,
    run,
    scheduler=scheduler,
)

translated = []
with open("data/test1.de-en.de", "r") as f:
    for line in tqdm(f.readlines()):
        translated.append(
            translate(
                transformer, line, train_set, train_set.bos_id, train_set.eos_id, device
            )
        )

with open("data/test1.de-en.en", "w", encoding="utf-8") as f:
    for line in translated:
        f.write(line + "\n")
