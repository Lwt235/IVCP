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


class ActionDecoder(nn.Module):
    r"""A lightweight decoder head that maps the <ACTION> token hidden state to action class logits.

    Supports both a single linear layer and a two-layer MLP with GELU activation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        decoder_type: Literal["linear", "mlp"] = "linear",
        mlp_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.decoder_type = decoder_type
        if decoder_type == "linear":
            self.head = nn.Linear(hidden_size, num_classes)
        elif decoder_type == "mlp":
            mid = mlp_hidden_size or hidden_size
            self.head = nn.Sequential(
                nn.Linear(hidden_size, mid),
                nn.GELU(),
                nn.Linear(mid, num_classes),
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""Compute action class logits from hidden states.

        Args:
            hidden_states: Tensor of shape ``(batch_size, hidden_size)``.

        Returns:
            Tensor of shape ``(batch_size, num_classes)``.
        """
        return self.head(hidden_states)

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
