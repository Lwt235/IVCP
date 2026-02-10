# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Literal, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from ..extras import logging


logger = logging.get_logger(__name__)

ACTION_DECODER_WEIGHTS_NAME = "action_decoder.safetensors"


class TransformerDecoderLayer(nn.Module):
    r"""A single transformer decoder layer with self-attention and feed-forward network."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Forward pass of transformer decoder layer.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            attn_mask: Optional attention mask.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = residual + x

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)

        return x


class ActionDecoder(nn.Module):
    r"""A lightweight decoder head that maps the <ACT> token hidden state to action class logits.

    Supports multiple decoder types:
    - ``linear``: Single linear layer.
    - ``mlp``: Two-layer MLP with GELU activation.
    - ``transformer``: Linear projection + merge with visual tokens + 2 transformer layers + MLP.
    - ``transformer_no_vision``: Same as transformer but without visual tokens (for controlled comparison).
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        decoder_type: Literal["linear", "mlp", "transformer", "transformer_no_vision"] = "linear",
        mlp_hidden_size: Optional[int] = None,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.decoder_type = decoder_type
        self.hidden_size = hidden_size

        if decoder_type == "linear":
            self.head = nn.Linear(hidden_size, num_classes)
        elif decoder_type == "mlp":
            mid = mlp_hidden_size or hidden_size
            self.head = nn.Sequential(
                nn.Linear(hidden_size, mid),
                nn.GELU(),
                nn.Linear(mid, num_classes),
            )
        elif decoder_type in ("transformer", "transformer_no_vision"):
            # Linear projection layer G: H -> H(G)
            self.projection = nn.Linear(hidden_size, hidden_size)

            # Transformer layers
            self.transformer_layers = nn.ModuleList([
                TransformerDecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_transformer_layers)
            ])

            # Final layer norm
            self.final_norm = nn.LayerNorm(hidden_size)

            # Classification MLP head
            mid = mlp_hidden_size or hidden_size
            self.head = nn.Sequential(
                nn.Linear(hidden_size, mid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mid, num_classes),
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Compute action class logits from hidden states.

        Args:
            hidden_states: Tensor of shape ``(batch_size, hidden_size)`` for linear/mlp,
                or ``(batch_size, 1, hidden_size)`` for transformer types.
            visual_tokens: Optional tensor of shape ``(batch_size, num_visual_tokens, hidden_size)``.
                Required for ``transformer`` decoder type, ignored for others.

        Returns:
            Tensor of shape ``(batch_size, num_classes)``.
        """
        if self.decoder_type in ("linear", "mlp"):
            # Simple forward pass for linear/mlp
            if hidden_states.dim() == 3:
                hidden_states = hidden_states.squeeze(1)
            return self.head(hidden_states)

        elif self.decoder_type == "transformer":
            # Ensure hidden_states has shape (B, 1, D)
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1)

            # Project hidden states: H -> H(G)
            h_g = self.projection(hidden_states)  # (B, 1, D)

            if visual_tokens is None:
                raise ValueError("visual_tokens is required for 'transformer' decoder type")

            # Merge H(G) with visual tokens: [H(G), T_V]
            # H(G) acts as a cls token at the beginning
            x = torch.cat([h_g, visual_tokens], dim=1)  # (B, 1 + num_visual_tokens, D)

            # Pass through transformer layers
            for layer in self.transformer_layers:
                x = layer(x)

            # Apply final normalization
            x = self.final_norm(x)

            # Extract cls token (first token which is H(G)) for classification
            cls_token = x[:, 0, :]  # (B, D)

            # Classification MLP
            return self.head(cls_token)

        elif self.decoder_type == "transformer_no_vision":
            # Same as transformer but without visual tokens (for controlled comparison)
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1)

            # Project hidden states: H -> H(G)
            h_g = self.projection(hidden_states)  # (B, 1, D)

            # Only use H(G) without visual tokens
            x = h_g  # (B, 1, D)

            # Pass through transformer layers
            for layer in self.transformer_layers:
                x = layer(x)

            # Apply final normalization
            x = self.final_norm(x)

            # Extract cls token for classification
            cls_token = x[:, 0, :]  # (B, D)

            # Classification MLP
            return self.head(cls_token)

        else:
            raise ValueError(f"Unknown decoder_type: {self.decoder_type}")

    def save_pretrained(self, save_directory: str) -> None:
        r"""Save action decoder weights to *save_directory*."""
        os.makedirs(save_directory, exist_ok=True)
        save_file(self.state_dict(), os.path.join(save_directory, ACTION_DECODER_WEIGHTS_NAME))
        logger.info_rank0(f"Action decoder saved to {save_directory}")

    def load_pretrained(self, load_directory: str) -> None:
        r"""Load action decoder weights from *load_directory*."""
        path = os.path.join(load_directory, ACTION_DECODER_WEIGHTS_NAME)
        if os.path.isfile(path):
            state_dict = load_file(path)
            self.load_state_dict(state_dict)
            logger.info_rank0(f"Action decoder loaded from {path}")
        else:
            logger.warning_rank0(f"Action decoder weights not found at {path}, using random init.")
