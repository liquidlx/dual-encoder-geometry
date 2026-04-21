"""
baseline_codet5.py -- Baseline: CodeT5+ 220M with standard single encoder.

Wraps the pretrained CodeT5+ model with no architectural changes.
Instruction and context are concatenated with <sep> and encoded together
in the same pretrained encoder space.

This is the control condition for Stage 3.
The only difference from the dual encoder is the encoder architecture.
Everything else -- tokenizer, decoder, training loop, data -- is identical.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from transformers import T5ForConditionalGeneration, RobertaTokenizer

CHECKPOINT = "Salesforce/codet5p-220m-py"


class BaselineCodeT5(nn.Module):
    """
    CodeT5+ 220M with standard single encoder (control condition).

    Instruction and context are concatenated before encoding:
      [ instruction <sep> context ] -> CodeT5 encoder -> decoder -> code

    get_encoder_embeddings() extracts instruction and context regions
    from joint memory for geometry analysis -- they are already mixed
    through the shared attention layers before extraction.
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.sep_token_id = tokenizer.sep_token_id or tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id

        # Load full pretrained model
        base = T5ForConditionalGeneration.from_pretrained(CHECKPOINT)
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.lm_head = base.lm_head
        self.shared = base.shared  # shared embedding between encoder and decoder
        self.config = base.config

    def _embed(self, input_ids):
        return self.shared(input_ids)

    def encode(
        self,
        instruction_ids: torch.Tensor,   # (B, L_i)
        context_ids: torch.Tensor,       # (B, L_c)
        instruction_mask: torch.Tensor,  # (B, L_i)
        context_mask: torch.Tensor,      # (B, L_c)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate instruction and context, encode together.
        Returns (encoder_output, attention_mask).
        """
        B = instruction_ids.size(0)
        device = instruction_ids.device

        # <sep> token between instruction and context
        sep = torch.full((B, 1), self.sep_token_id, dtype=torch.long, device=device)
        sep_mask = torch.ones(B, 1, dtype=torch.long, device=device)

        src_ids = torch.cat([instruction_ids, sep, context_ids], dim=1)
        src_mask = torch.cat([instruction_mask, sep_mask, context_mask], dim=1)

        out = self.encoder(
            input_ids=src_ids,
            attention_mask=src_mask,
        )
        return out.last_hidden_state, src_mask

    def forward(self, batch: dict) -> torch.Tensor:
        """Returns logits (B, L_t-1, vocab_size)."""
        memory, src_mask = self.encode(
            batch["instruction_ids"],
            batch["context_ids"],
            batch["instruction_mask"],
            batch["context_mask"],
        )

        # Teacher forcing: target[:-1] as input, target[1:] as label
        target_ids = batch["target_ids"][:, :-1]
        target_mask = batch["target_mask"][:, :-1]

        decoder_out = self.decoder(
            input_ids=target_ids,
            attention_mask=target_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=src_mask,
        )
        logits = self.lm_head(decoder_out.last_hidden_state)
        return logits

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

        memory, src_mask = self.encode(
            instruction_ids, context_ids, instruction_mask, context_mask
        )

        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            decoder_out = self.decoder(
                input_ids=generated,
                encoder_hidden_states=memory,
                encoder_attention_mask=src_mask,
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
        Extract instruction and context embeddings for geometry analysis.

        In the baseline both live in the same joint memory -- they have already
        influenced each other through shared self-attention before extraction.
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