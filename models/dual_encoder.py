"""
dual_encoder.py -- Dual encoder with controlled cross-attention gate.

The architectural hypothesis:
  Instruction (what should be done) and context (what already exists)
  are encoded in separate spaces. Generation only happens after an
  explicit, auditable cross-attention operation:

      Q = vector_I  (instruction drives)
      K = vector_C  (context responds)
      V = vector_C

  The decoder never sees instruction and context in the same raw space.
  It receives only the output of the gate -- instruction filtered by context.

Parameters: ~17M (default d_model=256)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DualEncoderConfig:
    vocab_size: int = 8005
    d_model: int = 256
    n_heads: int = 8
    n_instruction_layers: int = 4   # instruction encoder depth
    n_context_layers: int = 4       # context encoder depth
    n_cross_attn_layers: int = 2    # gate depth (cross-attention layers between spaces)
    n_decoder_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    max_instruction_len: int = 128
    max_context_len: int = 128
    max_target_len: int = 128
    pad_id: int = 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
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


class InstructionContextGate(nn.Module):
    """
    The controlled bridge between instruction space (I) and context space (C).

    Implements: CrossAttention(Q=I, K=C, V=C) over multiple layers.
    The instruction drives what to look for; the context responds.
    The result stays in instruction space -- enriched but not fused with context.

    Each layer has a learnable scalar gate that controls how much context
    influences instruction. This gate is auditable during training:
      gate -> 0: instruction ignores context
      gate -> 1: instruction fully incorporates context
      gate ~0.3: observed convergence -- context influences ~30% of generation
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=n_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                "norm_q": nn.LayerNorm(d_model),
                "norm_kv": nn.LayerNorm(d_model),
                "norm_out": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                ),
            })
            for _ in range(n_layers)
        ])
        # Learnable scalar gate per layer -- initialized to 0 (sigmoid -> 0.5, neutral)
        self.gates = nn.Parameter(torch.zeros(n_layers))

    def forward(
        self,
        instruction_memory: torch.Tensor,       # (B, L_i, d_model) -- queries
        context_memory: torch.Tensor,           # (B, L_c, d_model) -- keys/values
        context_key_padding_mask: torch.Tensor, # (B, L_c) True=ignore
    ) -> tuple[torch.Tensor, list[float]]:
        """
        Returns:
          x           : (B, L_i, d_model) -- enriched instruction representation
          gate_values : [float] * n_layers -- for training auditability
        """
        x = instruction_memory
        gate_values = []

        for i, layer in enumerate(self.layers):
            gate = torch.sigmoid(self.gates[i])
            gate_values.append(gate.item())

            # Cross-attention: instruction queries context
            q = layer["norm_q"](x)
            kv = layer["norm_kv"](context_memory)

            attn_out, _ = layer["cross_attn"](
                query=q,
                key=kv,
                value=kv,
                key_padding_mask=context_key_padding_mask,
            )

            # Gated residual -- controls context influence
            x = x + gate * attn_out

            # FFN in instruction space
            x = x + layer["ffn"](layer["norm_out"](x))

        return x, gate_values


class DualEncoderModel(nn.Module):
    """
    Two independent encoders + cross-attention gate + decoder.

    Separation is by construction:
      - instruction_embedding: space I (only instruction tokens pass through here)
      - context_embedding:     space C (only context tokens pass through here)
      - InstructionContextGate: the only communication between I and C, auditable
      - decoder: receives only gate output, never raw I or C

    The key invariant: instruction_encoder and context_encoder have no shared
    weights. They develop independent representations from training start.
    This is what creates geometric orthogonality (cross-similarity -> 0).

    get_encoder_embeddings() returns both spaces BEFORE the gate for geometry
    analysis -- the pure separated representations.
    """

    def __init__(self, config: DualEncoderConfig):
        super().__init__()
        self.config = config

        # Independent embeddings -- space I and space C share no weights
        self.instruction_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_id
        )
        self.context_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_id
        )

        self.instruction_pos = PositionalEncoding(
            config.d_model, config.max_instruction_len, config.dropout
        )
        self.context_pos = PositionalEncoding(
            config.d_model, config.max_context_len, config.dropout
        )

        # Instruction encoder -- space I
        i_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.instruction_encoder = nn.TransformerEncoder(
            i_layer, num_layers=config.n_instruction_layers
        )

        # Context encoder -- space C
        c_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            c_layer, num_layers=config.n_context_layers
        )

        # The gate -- the only communication bridge between I and C
        self.gate = InstructionContextGate(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_cross_attn_layers,
            dropout=config.dropout,
        )

        # Decoder receives only gate output
        self.target_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_id
        )
        self.target_pos = PositionalEncoding(
            config.d_model, config.max_target_len, config.dropout
        )

        d_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(d_layer, num_layers=config.n_decoder_layers)

        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying with instruction embedding (decoder generates in instruction space)
        self.output_proj.weight = self.instruction_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Gate initialized to 0 -> sigmoid(0) = 0.5, neutral starting point
        nn.init.zeros_(self.gate.gates)

    def _encode_instruction(
        self,
        instruction_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode instruction in space I."""
        x = self.instruction_embedding(instruction_ids) * math.sqrt(self.config.d_model)
        x = self.instruction_pos(x)
        pad_mask = instruction_mask == 0
        memory = self.instruction_encoder(x, src_key_padding_mask=pad_mask)
        return memory, pad_mask

    def _encode_context(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode context in space C."""
        x = self.context_embedding(context_ids) * math.sqrt(self.config.d_model)
        x = self.context_pos(x)
        pad_mask = context_mask == 0
        memory = self.context_encoder(x, src_key_padding_mask=pad_mask)
        return memory, pad_mask

    def encode(
        self,
        instruction_ids: torch.Tensor,
        context_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        """
        Encode both spaces and apply the gate.
        Returns (gated_memory, instruction_pad_mask, gate_values).
        The decoder never receives raw instruction_memory or context_memory.
        """
        i_memory, i_pad = self._encode_instruction(instruction_ids, instruction_mask)
        c_memory, c_pad = self._encode_context(context_ids, context_mask)

        # Gate: instruction queries context
        gated, gate_values = self.gate(i_memory, c_memory, c_pad)

        return gated, i_pad, gate_values

    def decode(
        self,
        target_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        L_t = target_ids.size(1)
        device = target_ids.device

        tgt = self.target_embedding(target_ids) * math.sqrt(self.config.d_model)
        tgt = self.target_pos(tgt)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(L_t, device=device)

        out = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(out)

    def forward(self, batch: dict) -> tuple[torch.Tensor, list[float]]:
        """Returns (logits, gate_values). gate_values logged during training."""
        gated, i_pad, gate_values = self.encode(
            batch["instruction_ids"],
            batch["context_ids"],
            batch["instruction_mask"],
            batch["context_mask"],
        )
        logits = self.decode(batch["target_ids"][:, :-1], gated, i_pad)
        return logits, gate_values

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
    ) -> torch.Tensor:
        """Greedy decoding for evaluation."""
        self.eval()
        device = instruction_ids.device
        B = instruction_ids.size(0)

        gated, i_pad, _ = self.encode(
            instruction_ids, context_ids, instruction_mask, context_mask
        )

        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.decode(generated, gated, i_pad)
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
        Return embeddings BEFORE the gate -- the two pure separated spaces.
        Used for geometry analysis (Stage 1 applied to trained model).

        These vectors live in genuinely separate spaces (independent weights,
        no communication until the gate). This is what produces geometric
        orthogonality: cross-similarity -> 0 after training.
        """
        i_memory, _ = self._encode_instruction(instruction_ids, instruction_mask)
        c_memory, _ = self._encode_context(context_ids, context_mask)
        return {
            "instruction": i_memory.mean(dim=1),  # (B, d_model)
            "context": c_memory.mean(dim=1),       # (B, d_model)
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
