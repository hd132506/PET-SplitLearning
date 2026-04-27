import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5LayerNorm


class T5ClientModel(nn.Module):
    """
    Client-side model: T5 embedding layer only (cut_layer=0).
    Applies PET (Personal Embedding Transformation) if use_pet=True.
    """

    def __init__(self, tokenizer, d_model=512, use_pet=True, seed=None):
        super().__init__()
        self.d_model = d_model
        self.use_pet = use_pet

        base = T5ForConditionalGeneration.from_pretrained("t5-small")
        vocab_size = base.config.vocab_size
        # Randomly initialized embedding (client-local)
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        nn.init.xavier_uniform_(self.embed_tokens.weight)

        if use_pet:
            self.plus_mask, self.minus_mask = self._make_masks(d_model, 128, 128, seed)

    def _make_masks(self, d, n_plus, n_minus, seed):
        rng = random.Random(seed)
        idx = rng.sample(range(d), n_plus + n_minus)
        plus_mask = torch.zeros(d)
        minus_mask = torch.zeros(d)
        for i in idx[:n_plus]:
            plus_mask[i] = 1.0
        for i in idx[n_plus:]:
            minus_mask[i] = 1.0
        return plus_mask, minus_mask

    def forward(self, input_ids, attention_mask=None):
        h = self.embed_tokens(input_ids)  # [B, L, D]

        if self.use_pet:
            pm = self.plus_mask.to(h.device).view(1, 1, -1)
            mm = self.minus_mask.to(h.device).view(1, 1, -1)
            # Apply masks: amplify and suppress selected dims
            h = h * pm + h * mm + h * (1 - pm - mm)
            # Regularization signal exposed via plus/minus components
            self._plus_h = h * pm
            self._minus_h = h * mm
        return h  # smashed data [B, L, D]

    def pet_reg_loss(self, lambda_plus=1e-4, lambda_minus=1e-4):
        """Ltrans = -lambda+ * ||plus|| + lambda- * ||minus||  (Eq. 11, 12)"""
        if not self.use_pet or not hasattr(self, '_plus_h'):
            return torch.tensor(0.0)
        loss_amp = -lambda_plus * self._plus_h.norm(p=1)
        loss_sup = lambda_minus * self._minus_h.norm(p=1)
        return loss_amp + loss_sup


class T5ServerModel(nn.Module):
    """
    Server-side model: T5 encoder layers + classification head.
    The server has its own copy of T5 encoder blocks (from cut_layer=0 to end).
    """

    def __init__(self, num_classes=2):
        super().__init__()
        base = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.encoder_blocks = base.encoder.block           # all 6 encoder layers
        self.final_layer_norm = T5LayerNorm(base.config.d_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base.config.d_model, num_classes)

    def forward(self, hidden_states, attention_mask=None):
        # Extend attention mask to 4D for T5 blocks
        if attention_mask is not None:
            ext = (1.0 - attention_mask[:, None, None, :].float()) * -1e9
        else:
            ext = None

        for block in self.encoder_blocks:
            hidden_states = block(hidden_states, attention_mask=ext)[0]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        cls = hidden_states[:, 0, :]           # CLS token
        return self.classifier(cls)


class InversionModel(nn.Module):
    """
    Reconstruction attack model: seq2seq that maps smashed data → original text.
    Uses T5ForConditionalGeneration as backbone.
    """

    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")

    def forward(self, inputs_embeds, attention_mask, labels):
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out.loss, out.logits

    @torch.no_grad()
    def generate(self, inputs_embeds, attention_mask, max_new_tokens=64):
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
