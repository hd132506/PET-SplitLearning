import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ─── CRS: Supervised Contrastive Loss (InfoNCE style) ────────────────────────

def supervised_contrastive_loss(embeddings, labels, temperature=1.0):
    """
    Eq. 9 in paper. embeddings: [B, D] (pooled), labels: [B]
    """
    embeddings = F.normalize(embeddings, dim=1)
    sim = torch.matmul(embeddings, embeddings.T) / temperature  # [B, B]

    B = embeddings.size(0)
    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [B, B]
    mask_self = torch.eye(B, device=embeddings.device)
    mask_pos = mask_pos - mask_self  # exclude self

    # log softmax over all (except self)
    exp_sim = torch.exp(sim) * (1 - mask_self)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    n_pos = mask_pos.sum(dim=1)
    loss = -(mask_pos * log_prob).sum(dim=1) / (n_pos + 1e-8)
    return loss[n_pos > 0].mean()


# ─── Main Training Loop ────────────────────────────────────────────────────────

def train_one_epoch(client, server, loader, optimizer, device, use_crs=False,
                    alpha=0.1, temperature=1.0, smash_buf=None):
    client.train()
    server.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    for input_ids, attn_mask, labels, _ in tqdm(loader, desc="Train", leave=False):
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        smashed = client(input_ids, attn_mask)        # [B, L, D]

        if smash_buf is not None:
            smash_buf.push(smashed)

        logits = server(smashed, attn_mask)            # [B, C]
        loss_task = criterion(logits, labels)

        loss_trans = client.pet_reg_loss()
        if loss_trans.device != device:
            loss_trans = loss_trans.to(device)

        loss_crs = torch.tensor(0.0, device=device)
        if use_crs:
            pooled = smashed.mean(dim=1)               # [B, D]
            loss_crs = supervised_contrastive_loss(pooled, labels, temperature)

        loss = loss_task + loss_trans + alpha * loss_crs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss_task.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(client, server, loader, device):
    client.eval()
    server.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0, 0, 0

    for input_ids, attn_mask, labels, _ in loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        smashed = client(input_ids, attn_mask)
        logits = server(smashed, attn_mask)
        total_loss += criterion(logits, labels).item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total
