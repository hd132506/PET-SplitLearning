"""
Evaluation metrics for reconstruction attack quality (Section IV-A-4 in paper):
- Cosine Similarity (CS)
- BLEU
- ROUGE-1, ROUGE-2, ROUGE-L
"""
import numpy as np
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer as rouge_lib
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import T5Tokenizer


_rouge = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
_smooth = SmoothingFunction().method1


def _embed_text(text, tokenizer, model, device, max_length=128):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_length, padding="max_length")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        emb = model.embed_tokens(input_ids).mean(dim=1)  # [1, D]
    return emb.squeeze(0)


def cosine_sim_texts(orig, recon, tokenizer, embed_model, device):
    """CS between embedding of original and reconstructed text."""
    e1 = _embed_text(orig, tokenizer, embed_model, device)
    e2 = _embed_text(recon, tokenizer, embed_model, device)
    return F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()


def bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens:
        return 0.0
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=_smooth)


def rouge(reference: str, hypothesis: str):
    scores = _rouge.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


@torch.no_grad()
def evaluate_reconstruction(client, inversion_model, loader, tokenizer,
                              device, n_samples=100):
    """
    Run reconstruction on n_samples from loader.
    Returns dict with mean CS, BLEU, ROUGE-1/2/L.
    """
    client.eval()
    inversion_model.eval()

    cs_scores, bleu_scores = [], []
    r1_scores, r2_scores, rL_scores = [], [], []

    count = 0
    for input_ids, attn_mask, _, orig_texts in loader:
        if count >= n_samples:
            break
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        smashed = client(input_ids, attn_mask)
        gen_ids = inversion_model.generate(smashed, attn_mask)
        recon_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for orig, recon in zip(orig_texts, recon_texts):
            # Cosine similarity via token embedding
            enc_o = tokenizer(orig, return_tensors="pt", truncation=True,
                              max_length=128, padding="max_length")
            enc_r = tokenizer(recon, return_tensors="pt", truncation=True,
                              max_length=128, padding="max_length")
            e_o = client.embed_tokens(enc_o["input_ids"].to(device)).mean(dim=1)
            e_r = client.embed_tokens(enc_r["input_ids"].to(device)).mean(dim=1)
            cs_scores.append(F.cosine_similarity(e_o, e_r).item())

            bleu_scores.append(bleu(orig, recon))
            r = rouge(orig, recon)
            r1_scores.append(r["rouge1"])
            r2_scores.append(r["rouge2"])
            rL_scores.append(r["rougeL"])
            count += 1
            if count >= n_samples:
                break

    return {
        "cosine": np.mean(cs_scores),
        "bleu": np.mean(bleu_scores),
        "rouge1": np.mean(r1_scores),
        "rouge2": np.mean(r2_scores),
        "rougeL": np.mean(rL_scores),
    }
