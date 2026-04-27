import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_agnews(tokenizer, max_length=128, n_per_class=5000, seed=42):
    """
    AGNews: label 0 (World), 1 (Sports) → downstream task (client trains on these)
            label 2 (Business), 3 (Sci/Tech) → server auxiliary for reconstruction attack
    Returns train_ds, val_ds, aux_ds
    """
    raw = load_dataset("ag_news", split="train")

    main_texts, main_labels = [], []
    aux_texts = []

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for item in raw:
        lbl = item["label"]
        if lbl in (0, 1) and counts[lbl] < n_per_class:
            main_texts.append(item["text"])
            main_labels.append(lbl)
            counts[lbl] += 1
        elif lbl in (2, 3) and counts[lbl] < n_per_class:
            aux_texts.append(item["text"])
            counts[lbl] += 1
        if all(v >= n_per_class for v in counts.values()):
            break

    tr_texts, va_texts, tr_labels, va_labels = train_test_split(
        main_texts, main_labels, test_size=0.2, random_state=seed, stratify=main_labels
    )

    train_ds = AGNewsDataset(tr_texts, tr_labels, tokenizer, max_length)
    val_ds = AGNewsDataset(va_texts, va_labels, tokenizer, max_length)
    aux_ds = AGNewsDataset(aux_texts, [0] * len(aux_texts), tokenizer, max_length)
    return train_ds, val_ds, aux_ds


class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.texts = texts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.encodings["input_ids"][idx],
            self.encodings["attention_mask"][idx],
            self.labels[idx],
            self.texts[idx],
        )
