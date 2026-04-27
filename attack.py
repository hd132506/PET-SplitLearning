"""
Data Reconstruction Attack (DRA) - FORA-style (Section IV-B in paper)
Steps:
  1. Mimic client: same architecture as target client, trained on server's auxiliary data
  2. Smashed data distribution alignment via MK-MMD
  3. Train inversion model (InversionModel) on (mimic_smashed, original_text) pairs
  4. Evaluate reconstruction quality on real client's smashed data
"""
import torch
import torch.nn as nn
from tqdm import tqdm


# ─── MK-MMD ───────────────────────────────────────────────────────────────────

def _gaussian_kernel(x, y, sigma):
    diff = x.unsqueeze(1) - y.unsqueeze(0)        # [N, M, D]
    dist_sq = diff.pow(2).sum(dim=2)               # [N, M]
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def mk_mmd(source, target, sigmas=(0.5, 1.0, 2.0, 4.0, 8.0)):
    """Multi-Kernel MMD (Eq. 22, 23)"""
    source = source.reshape(source.size(0), -1)
    target = target.reshape(target.size(0), -1)

    loss = torch.tensor(0.0, device=source.device)
    for sigma in sigmas:
        kss = _gaussian_kernel(source, source, sigma).mean()
        ktt = _gaussian_kernel(target, target, sigma).mean()
        kst = _gaussian_kernel(source, target, sigma).mean()
        loss = loss + kss + ktt - 2 * kst
    return loss / len(sigmas)


# ─── Step 1-2: Align mimic client to target client's smashed data ──────────────

def align_mimic_client(mimic_client, real_smashed_buffer, aux_loader,
                        device, epochs=3, lr=1e-4):
    """
    mimic_client: same architecture as T5ClientModel (target)
    real_smashed_buffer: list of smashed tensors collected from real client during SL
    """
    opt = torch.optim.Adam(mimic_client.parameters(), lr=lr)
    mimic_client.train()

    real_tensor = torch.cat(real_smashed_buffer, dim=0).to(device)  # [N, L, D]
    real_mean = real_tensor.mean(dim=0, keepdim=True)               # [1, L, D]

    for epoch in range(epochs):
        total = 0
        for input_ids, attn_mask, _, _ in tqdm(aux_loader, desc=f"Mimic align ep{epoch+1}", leave=False):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            mimic_smashed = mimic_client(input_ids, attn_mask)  # [B, L, D]
            mimic_mean = mimic_smashed.mean(dim=0, keepdim=True)

            loss = mk_mmd(mimic_mean.view(1, -1), real_mean.view(1, -1))

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  MK-MMD ep{epoch+1}: {total/len(aux_loader):.6f}")


# ─── Step 3: Train inversion model ────────────────────────────────────────────

def train_inversion_model(inversion_model, mimic_client, aux_loader,
                           tokenizer, device, epochs=5, lr=1e-4, max_label_len=64):
    opt = torch.optim.Adam(inversion_model.parameters(), lr=lr)
    inversion_model.train()
    mimic_client.eval()

    for epoch in range(epochs):
        total = 0
        for input_ids, attn_mask, _, texts in tqdm(aux_loader, desc=f"Inversion ep{epoch+1}", leave=False):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            with torch.no_grad():
                smashed = mimic_client(input_ids, attn_mask)  # [B, L, D]

            label_enc = tokenizer(
                list(texts),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_label_len,
            )
            labels = label_enc["input_ids"].to(device)
            labels[labels == tokenizer.pad_token_id] = -100   # ignore padding in loss

            loss, _ = inversion_model(smashed, attn_mask, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  Inversion ep{epoch+1}: {total/len(aux_loader):.4f}")


# ─── Step 4: Collect real smashed data during training ────────────────────────

class SmashBuffer:
    """Accumulates smashed data from real client during SL training."""

    def __init__(self, max_batches=50):
        self.buffer = []
        self.max_batches = max_batches

    def push(self, smashed: torch.Tensor):
        if len(self.buffer) < self.max_batches:
            self.buffer.append(smashed.detach().cpu())

    def ready(self):
        return len(self.buffer) > 0
