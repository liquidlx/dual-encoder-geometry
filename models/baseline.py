"""
baseline.py -- Standard encoder-decoder transformer (single shared encoder space).

Instruction and context are concatenated with <sep> and encoded together
in the same vector space. This is the standard LLM architecture for code.

Architecture:
  [<bos> instruction <sep> context] -> Encoder -> memory
  [<bos> target tokens]             -> Decoder (cross-attn over memory)

Parameters: ~12M (default d_model=256)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class BaselineConfig:
    vocab_size: int = 8005
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 384       # max instruction + sep + context
    max_target_len: int = 128
    pad_id: int = 0
    sep_id: int = 4


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BaselineModel(nn.Module):
    """
    Standard encoder-decoder transformer.

    Instruction and context share a single embedding space.
    They are concatenated before encoding -- no architectural separation.

    This is the control condition for the dual encoder experiment.
    get_encoder_embeddings() extracts instruction/context regions from the
    joint memory for geometry analysis (Stage 1).
    """

    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config

        # Single shared embedding space for instruction and context
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.pos_enc = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)

        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying between embedding and output projection
        self.output_proj.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        instruction_ids: torch.Tensor,
        context_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate instruction and context with <sep> and encode together.
        Returns (memory, src_key_padding_mask).
        """
        B = instruction_ids.size(0)
        device = instruction_ids.device

        sep = torch.full((B, 1), self.config.sep_id, dtype=torch.long, device=device)
        sep_mask = torch.ones(B, 1, dtype=torch.long, device=device)

        src = torch.cat([instruction_ids, sep, context_ids], dim=1)
        src_mask = torch.cat([instruction_mask, sep_mask, context_mask], dim=1)

        x = self.embedding(src) * math.sqrt(self.config.d_model)
        x = self.pos_enc(x)

        # PyTorch convention: True = ignore (padding positions)
        src_key_padding_mask = src_mask == 0

        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode(
        self,
        target_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive decode step."""
        L_t = target_ids.size(1)
        device = target_ids.device

        tgt = self.embedding(target_ids) * math.sqrt(self.config.d_model)
        tgt = self.pos_enc(tgt)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(L_t, device=device)

        out = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(out)

    def forward(self, batch: dict) -> torch.Tensor:
        """Returns logits (B, L_t-1, vocab_size). Loss computed externally."""
        memory, mem_mask = self.encode(
            batch["instruction_ids"],
            batch["context_ids"],
            batch["instruction_mask"],
            batch["context_mask"],
        )
        # Teacher forcing: input is target[:-1], label is target[1:]
        logits = self.decode(batch["target_ids"][:, :-1], memory, mem_mask)
        return logits

    @torch.no_grad()
    def generate(
        self,
        instruction_ids: torch.Tensor,
        context_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
        context_mask: torch.Tensor,
        max_new_tokens: int = 128,
        bos_id: int = 2,
        eos_id: int = 3,
        pad_id: int = 0,
    ) -> torch.Tensor:
        """Greedy decoding for evaluation."""
        self.eval()
        device = instruction_ids.device
        B = instruction_ids.size(0)

        memory, mem_mask = self.encode(
            instruction_ids, context_ids, instruction_mask, context_mask
        )

        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.decode(generated, memory, mem_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            done |= next_token.squeeze(-1) == eos_id
            if done.all():
                break

        return generated

    def get_encoder_embeddings(
        self,
        instruction_ids: torch.Tensor,
        context_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Extract instruction and context embeddings for geometry analysis (Stage 1).

        In the baseline, both live in the same joint memory space -- they have
        already influenced each other through self-attention before extraction.
        This is the key difference from the dual encoder.
        """
        memory, _ = self.encode(
            instruction_ids, context_ids, instruction_mask, context_mask
        )
        L_i = instruction_ids.size(1)
        # Instruction occupies positions 0..L_i-1 in joint memory
        i_emb = memory[:, :L_i, :].mean(dim=1)
        # Context occupies positions L_i+1.. (+1 for <sep>)
        c_emb = memory[:, L_i + 1:, :].mean(dim=1)
        return {"instruction": i_emb, "context": c_emb}

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
