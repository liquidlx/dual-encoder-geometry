"""
dual_encoder_codet5.py -- Dual encoder built on CodeT5+ 220M pretrained weights.

The architectural hypothesis applied to a pretrained base:
  - Load CodeT5+ 220M pretrained model
  - Extract encoder and create TWO independent copies
  - Both copies start from identical pretrained weights
  - Fine-tuning allows them to diverge into separate spaces
  - The only communication between spaces is the InstructionContextGate

This design controls for pretraining: both baseline and dual encoder
start from the same CodeT5+ weights. Any geometric separation that
emerges after fine-tuning is attributable to the dual encoder architecture,
not to differences in pretraining.

Geometry analysis:
  - get_encoder_embeddings() returns embeddings BEFORE the gate
  - These are the two pure separated spaces
  - Expected: near-orthogonal after fine-tuning (same as Stage 2)
  - But now with pass@1 > 0 because of pretrained Python knowledge
"""

import torch
import torch.nn as nn
from copy import deepcopy
from dataclasses import dataclass
from transformers import T5ForConditionalGeneration

CHECKPOINT = "Salesforce/codet5p-220m-py"


@dataclass
class GateConfig:
    d_model: int = 512        # CodeT5+ 220M hidden size
    n_heads: int = 8
    n_layers: int = 2         # cross-attention gate layers
    dropout: float = 0.1


class InstructionContextGate(nn.Module):
    """
    Controlled cross-attention bridge between instruction space I and context space C.

    CrossAttention(Q=I, K=C, V=C) -- instruction drives, context responds.
    Scalar gate per layer controls context influence (auditable during training).

    Identical to Stage 2 implementation, adapted for CodeT5+ hidden size (512).
    """

    def __init__(self, config: GateConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=config.d_model,
                    num_heads=config.n_heads,
                    dropout=config.dropout,
                    batch_first=True,
                ),
                "norm_q":   nn.LayerNorm(config.d_model),
                "norm_kv":  nn.LayerNorm(config.d_model),
                "norm_attn": nn.LayerNorm(config.d_model),
                "norm_out": nn.LayerNorm(config.d_model),
                "ffn": nn.Sequential(
                    nn.Linear(config.d_model, config.d_model * 4),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.d_model * 4, config.d_model),
                    nn.Dropout(config.dropout),
                ),
            })
            for _ in range(config.n_layers)
        ])
        self.output_norm = nn.LayerNorm(config.d_model)
        # Scalar gate per layer in [-1, 1] via tanh.
        # Starts at 0.0 -> pretrained decoder initially sees pure instruction memory.
        self.gates = nn.Parameter(torch.zeros(config.n_layers))

    def forward(
        self,
        instruction_memory: torch.Tensor,       # (B, L_i, d_model)
        context_memory: torch.Tensor,           # (B, L_c, d_model)
        context_key_padding_mask: torch.Tensor, # (B, L_c) True=ignore
    ) -> tuple[torch.Tensor, list[float]]:
        x = instruction_memory
        base_rms = instruction_memory.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
        gate_values = []

        for i, layer in enumerate(self.layers):
            gate = torch.tanh(self.gates[i])
            gate_values.append(gate.item())

            q  = layer["norm_q"](x)
            kv = layer["norm_kv"](context_memory)

            attn_out, _ = layer["cross_attn"](
                query=q,
                key=kv,
                value=kv,
                key_padding_mask=context_key_padding_mask,
            )

            x = x + gate * layer["norm_attn"](attn_out)
            x = x + layer["ffn"](layer["norm_out"](x))

        # Keep decoder-facing memory on the same scale as pretrained encoder output.
        x = self.output_norm(x)
        out_rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
        x = x * (base_rms / out_rms)
        return x, gate_values


class DualEncoderCodeT5(nn.Module):
    """
    Dual encoder built on CodeT5+ 220M pretrained weights.

    Setup:
      1. Load CodeT5+ 220M (pretrained on Python code)
      2. deepcopy encoder twice -> instruction_encoder, context_encoder
      3. Both start from identical pretrained weights
      4. Fine-tuning allows independent weight evolution
      5. Gate bridges the two spaces for generation

    The decoder and lm_head are shared (same as CodeT5+ pretrained).
    This minimizes parameter count while maximizing the experimental
    isolation: only the encoder architecture differs from the baseline.

    Geometry analysis (get_encoder_embeddings):
      Returns embeddings BEFORE the gate -- the two pure separated spaces.
      Expected convergence after fine-tuning:
        baseline:     cross-similarity ~0.7 (shared space, high overlap)
        dual encoder: cross-similarity ~0.0 (near-orthogonal, separated)
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.sep_token_id = tokenizer.sep_token_id or tokenizer.eos_token_id

        # Load pretrained model once
        base = T5ForConditionalGeneration.from_pretrained(CHECKPOINT)

        # Two independent copies of the pretrained encoder
        self.instruction_encoder = deepcopy(base.encoder)  # space I
        self.context_encoder     = deepcopy(base.encoder)  # space C

        # Symmetry breaking: add small noise to context encoder weights.
        #
        # Root cause of frozen gate: at initialization both encoders are
        # identical, so instruction_memory == context_memory. Cross-attention
        # produces attn_out == instruction_memory, and the gate gradient
        # points only in the direction of scaling -- not selective attention.
        #
        # Breaking symmetry gives the gate an immediate signal to learn from:
        # the two encoder outputs are now slightly different, so the gate
        # gradient points in the direction that amplifies or suppresses the
        # difference -- which is exactly what we want.
        #
        # std=1e-3 is tiny relative to pretrained weight magnitudes.
        # Small enough to preserve pretrained representations.
        # Large enough to break the symmetry that causes frozen gates.
        SYMMETRY_BREAK_STD = 1e-3
        with torch.no_grad():
            for p in self.context_encoder.parameters():
                # Perturb matrix-shaped weights only; avoid bias/norm drift.
                if p.ndim >= 2:
                    p.add_(torch.randn_like(p) * SYMMETRY_BREAK_STD)

        # Verify symmetry was broken -- compare a sample of weights
        with torch.no_grad():
            # Sample first layer weights to verify noise was applied
            i_params = list(self.instruction_encoder.parameters())
            c_params = list(self.context_encoder.parameters())
            # Use first weight tensor for verification
            i_w = i_params[0].flatten()[:1000].float()
            c_w = c_params[0].flatten()[:1000].float()
            diff = (i_w - c_w).abs().mean().item()
            print(f"  Encoder symmetry broken: avg weight diff = {diff:.6f} (target: ~{SYMMETRY_BREAK_STD})")

        # Shared decoder and output head (pretrained)
        self.decoder  = base.decoder
        self.lm_head  = base.lm_head
        self.shared   = base.shared
        self.config   = base.config

        # Keep encoder token embeddings tied to the pretrained shared table.
        # This preserves lexical alignment with the pretrained decoder.
        self.instruction_encoder.set_input_embeddings(self.shared)
        self.context_encoder.set_input_embeddings(self.shared)

        # Gate bridges I and C -- only communication between the two spaces
        d_model = base.config.d_model  # T5/CodeT5+ always uses d_model
        self.gate = InstructionContextGate(GateConfig(
            d_model=d_model,
            n_heads=8,
            n_layers=2,
            dropout=0.1,
        ))

        # Delete the base model to free memory
        del base
        torch.cuda.empty_cache()

        # Gate initialization: tanh(0) = 0.0 (closed at start).
        # Decoder sees in-distribution memory at step 0, then context is added gradually.
        nn.init.zeros_(self.gate.gates)

    def _encode_instruction(
        self,
        instruction_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode instruction in space I using pretrained encoder copy."""
        out = self.instruction_encoder(
            input_ids=instruction_ids,
            attention_mask=instruction_mask,
        )
        pad_mask = instruction_mask == 0  # True = ignore (PyTorch convention)
        return out.last_hidden_state, pad_mask

    def _encode_context(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode context in space C using independent pretrained encoder copy."""
        out = self.context_encoder(
            input_ids=context_ids,
            attention_mask=context_mask,
        )
        pad_mask = context_mask == 0
        return out.last_hidden_state, pad_mask

    def encode(
        self,
        instruction_ids: torch.Tensor,
        context_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        """
        Encode both spaces and apply the gate.
        Returns (gated_memory, instruction_attention_mask, gate_values).
        The decoder never receives raw instruction_memory or context_memory.
        """
        i_memory, i_pad = self._encode_instruction(instruction_ids, instruction_mask)
        c_memory, c_pad = self._encode_context(context_ids, context_mask)

        # Gate: instruction queries context (Q=I, K=C, V=C)
        gated, gate_values = self.gate(i_memory, c_memory, c_pad)

        # Convert back to attention mask format for decoder (1=attend, 0=ignore)
        i_attn_mask = instruction_mask
        return gated, i_attn_mask, gate_values

    def forward(self, batch: dict) -> tuple[torch.Tensor, list[float]]:
        """Returns (logits, gate_values)."""
        gated, i_mask, gate_values = self.encode(
            batch["instruction_ids"],
            batch["context_ids"],
            batch["instruction_mask"],
            batch["context_mask"],
        )

        target_ids  = batch["target_ids"][:, :-1]
        target_mask = batch["target_mask"][:, :-1]

        decoder_out = self.decoder(
            input_ids=target_ids,
            attention_mask=target_mask,
            encoder_hidden_states=gated,
            encoder_attention_mask=i_mask,
        )
        logits = self.lm_head(decoder_out.last_hidden_state)
        return logits, gate_values

    @torch.no_grad()
    def generate(
        self,
        instruction_ids: torch.Tensor,
        context_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
        context_mask: torch.Tensor,
        max_new_tokens: int = 128,
        bos_id: int = None,
        eos_id: int = None,
        pad_id: int = None,
    ) -> torch.Tensor:
        """Greedy decoding."""
        self.eval()
        device = instruction_ids.device
        B = instruction_ids.size(0)

        bos_id = bos_id or self.tokenizer.bos_token_id or self.tokenizer.pad_token_id
        eos_id = eos_id or self.tokenizer.eos_token_id

        gated, i_mask, _ = self.encode(
            instruction_ids, context_ids, instruction_mask, context_mask
        )

        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            decoder_out = self.decoder(
                input_ids=generated,
                encoder_hidden_states=gated,
                encoder_attention_mask=i_mask,
            )
            logits = self.lm_head(decoder_out.last_hidden_state)
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
        Used for geometry analysis (Stage 1 methodology applied to Stage 3).

        These live in genuinely separate spaces (independent weights after
        fine-tuning). Expected to show near-orthogonal cross-similarity,
        same as Stage 2, but with better generation quality.
        """
        i_memory, _ = self._encode_instruction(instruction_ids, instruction_mask)
        c_memory, _ = self._encode_context(context_ids, context_mask)
        return {
            "instruction": i_memory.mean(dim=1),  # (B, d_model)
            "context":     c_memory.mean(dim=1),   # (B, d_model)
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
