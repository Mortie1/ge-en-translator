# https://pytorch.org/tutorials/beginner/translation_transformer.html
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler

from tqdm import tqdm

from utils import create_mask
from inference import translate
import sacrebleu


def train_epoch(
    model: Module,
    optimizer: Optimizer,
    train_dataset: Dataset,
    batch_size: int,
    device: torch.device,
    desc: str,
    scheduler: LRScheduler = None,
) -> float:

    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.pad_id)
    losses = 0
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    for src, tgt in tqdm(train_dataloader, desc=desc):

        src = torch.transpose(src.to(device), 0, 1)
        tgt = torch.transpose(tgt.to(device), 0, 1)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, device, train_dataset.pad_id
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        if scheduler:
            scheduler.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(
    model: Module,
    val_dataset: Dataset,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> float:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=val_dataset.pad_id)
    losses = 0

    with torch.no_grad():
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
        )

        for src, tgt in tqdm(val_dataloader, desc=desc):

            src = torch.transpose(src.to(device), 0, 1)
            tgt = torch.transpose(tgt.to(device), 0, 1)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input, device, val_dataset.pad_id
            )

            logits = model(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(list(val_dataloader))


def train(
    transformer: Module,
    optimizer: Optimizer,
    train_set: Dataset,
    valid_set: Dataset,
    epochs: int,
    batch_size: int,
    device: torch.device,
    run=None,
    scheduler: LRScheduler = None,
):
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            transformer,
            optimizer,
            train_set,
            batch_size,
            device,
            f"Training {epoch}/{epochs}",
            scheduler,
        )
        val_loss = evaluate(
            transformer, valid_set, batch_size, device, f"Validating {epoch}/{epochs}"
        )

        if epoch % 2 == 1:
            bleu_score = 0.0
            translated = []
            with open("data/val.de-en.de", "r") as f:
                for line in tqdm(f.readlines(), desc="Translating for BLEU"):
                    translated.append(
                        translate(
                            transformer,
                            line,
                            valid_set,
                            valid_set.bos_id,
                            valid_set.eos_id,
                            device,
                        )
                    )

            # results = translate_file(
            #     transformer, "data/val.de-en.de", train_set, 64, device
            # )

            with open("data/val.de-en.en", "r", encoding="utf-8") as f:
                refs = f.readlines()
                bleu_score = sacrebleu.corpus_bleu(translated, [refs]).score
        print(
            f"Train loss: {train_loss}    |   Val loss: {val_loss}    |   BLEU:   {bleu_score:.2f}"
        )
        if run:
            run.log(
                {"train loss": train_loss, "val loss": val_loss, "bleu": bleu_score}
            )
