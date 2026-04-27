"""
run.py — Personal Embedding Transformation (PET) for Privacy-Preserving Split Learning
Paper: "Personal Embedding Transformation for Privacy-Preserving Split Learning on Textual Data"
       Kim et al., IEEE Access 2026

Usage:
    python run.py --method vanilla          # baseline (no defense)
    python run.py --method pet              # PET only (ablation)
    python run.py --method pet_crs          # PET + CRS (proposed, default)
    python run.py --method pet_crs --skip_attack  # skip reconstruction attack stage
"""

import argparse
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5Tokenizer

from models import T5ClientModel, T5ServerModel, InversionModel
from data import load_agnews
from trainer import train_one_epoch, evaluate
from attack import SmashBuffer, align_mimic_client, train_inversion_model
from evaluate import evaluate_reconstruction


# ─── Config ──────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["vanilla", "pet", "pet_crs"], default="pet_crs",
                   help="Defense method: vanilla | pet | pet_crs (proposed)")
    p.add_argument("--epochs", type=int, default=5,
                   help="Main task training epochs")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.1,
                   help="CRS loss weight (α in Eq. 10)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="CRS InfoNCE temperature τ")
    p.add_argument("--n_per_class", type=int, default=3000,
                   help="AGNews samples per class (reduce for quick test)")
    p.add_argument("--mimic_epochs", type=int, default=2,
                   help="MK-MMD alignment epochs for mimic client")
    p.add_argument("--inversion_epochs", type=int, default=3,
                   help="Inversion model training epochs")
    p.add_argument("--skip_attack", action="store_true",
                   help="Skip reconstruction attack evaluation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Method : {args.method.upper()}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("[1/4] Loading AGNews dataset …")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    train_ds, val_ds, aux_ds = load_agnews(
        tokenizer, max_length=128, n_per_class=args.n_per_class, seed=args.seed
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    aux_loader   = DataLoader(aux_ds,   batch_size=args.batch_size, shuffle=True,  num_workers=2)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Aux: {len(aux_ds)}")

    # ── 2. Models ─────────────────────────────────────────────────────────────
    print("\n[2/4] Building models …")
    use_pet = args.method in ("pet", "pet_crs")
    use_crs = args.method == "pet_crs"

    client = T5ClientModel(tokenizer, use_pet=use_pet, seed=args.seed).to(device)
    server = T5ServerModel(num_classes=2).to(device)
    print(f"  PET: {use_pet}  |  CRS: {use_crs}")
    print(f"  Client params: {sum(p.numel() for p in client.parameters()):,}")
    print(f"  Server params: {sum(p.numel() for p in server.parameters()):,}")

    optimizer = torch.optim.Adam(
        list(client.parameters()) + list(server.parameters()), lr=args.lr
    )

    # ── 3. Split Learning Training ────────────────────────────────────────────
    print(f"\n[3/4] Training ({args.epochs} epochs) …")

    smash_buf = SmashBuffer(max_batches=50)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            client, server, train_loader, optimizer, device,
            use_crs=use_crs, alpha=args.alpha, temperature=args.temperature,
            smash_buf=smash_buf,
        )
        val_loss, val_acc = evaluate(client, server, val_loader, device)
        best_acc = max(best_acc, val_acc)
        print(f"  Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}")

    print(f"\n  ★ Best Val Accuracy : {best_acc:.4f}")

    # ── 4. Reconstruction Attack ───────────────────────────────────────────────
    if args.skip_attack:
        print("\n[4/4] Skipping reconstruction attack (--skip_attack).")
        _print_summary(args.method, best_acc, None)
        return

    print("\n[4/4] Running reconstruction attack (FORA-style) …")

    # 4-a. Mimic client (same architecture, no PET — adversary doesn't know PET params)
    mimic_client = T5ClientModel(tokenizer, use_pet=False, seed=args.seed + 1).to(device)
    print("  Aligning mimic client via MK-MMD …")
    align_mimic_client(
        mimic_client, smash_buf.buffer, aux_loader,
        device=device, epochs=args.mimic_epochs,
    )

    # 4-b. Train inversion model
    inversion = InversionModel().to(device)
    print("  Training inversion (reconstruction) model …")
    train_inversion_model(
        inversion, mimic_client, aux_loader, tokenizer,
        device=device, epochs=args.inversion_epochs,
    )

    # 4-c. Evaluate reconstruction on real client smashed data (val set)
    print("  Evaluating reconstruction quality …")
    recon_metrics = evaluate_reconstruction(
        client, inversion, val_loader, tokenizer, device, n_samples=200
    )

    _print_summary(args.method, best_acc, recon_metrics)


def _print_summary(method, acc, recon):
    print(f"\n{'='*60}")
    print(f"  RESULTS  —  Method: {method.upper()}")
    print(f"{'='*60}")
    print(f"  Downstream Task Accuracy : {acc:.4f}")
    if recon:
        print(f"\n  Reconstruction Attack Metrics (lower = better privacy):")
        print(f"    Cosine Similarity : {recon['cosine']:.4f}")
        print(f"    BLEU              : {recon['bleu']:.4f}")
        print(f"    ROUGE-1           : {recon['rouge1']:.4f}")
        print(f"    ROUGE-2           : {recon['rouge2']:.4f}")
        print(f"    ROUGE-L           : {recon['rougeL']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
