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

r"""Trainer for the ``action_cls`` training stage.

This trainer runs the full VLM forward pass with ``output_hidden_states=True``,
locates the ``<ACT>`` special token in each sequence, extracts the
corresponding last-layer hidden state, and feeds it through a lightweight
``ActionDecoder`` head.  The classification loss (cross-entropy over action
classes) is back-propagated through the LoRA parameters of the backbone so that
the reasoning-aware hidden representation is leveraged for action recognition.
"""

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer
from typing_extensions import override

from ...extras import logging
from ...model.action_decoder import ACTION_DECODER_WEIGHTS_NAME, ActionDecoder
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import ProcessorMixin
    from transformers.trainer_utils import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class ActionClassificationTrainer(Trainer):
    r"""HuggingFace Trainer subclass for video action classification.

    The trainer owns both the VLM backbone *and* the ``ActionDecoder`` head.
    During ``compute_loss`` it:

    1. Runs the backbone with ``output_hidden_states=True``.
    2. Extracts the hidden state at the ``<ACT>`` position.
    3. Passes it through the ``ActionDecoder`` to obtain logits.
    4. Returns the cross-entropy loss over the action labels.
    """

    def __init__(
        self,
        action_decoder: "ActionDecoder",
        action_token_id: int,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"] = None,
        model_args: Optional["ModelArguments"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(**kwargs)
        if processor is not None:
            self.model_accepts_loss_kwargs = False

        self.action_decoder = action_decoder
        self.action_token_id = action_token_id
        self.finetuning_args = finetuning_args
        self.ce_loss = nn.CrossEntropyLoss()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _get_action_hidden_states(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        r"""Extract the hidden state at the ``<ACT>`` token position for each sample.

        Args:
            input_ids: ``(batch_size, seq_len)``
            hidden_states: Last-layer hidden states ``(batch_size, seq_len, hidden_size)``

        Returns:
            ``(batch_size, hidden_size)``
        """
        batch_size = input_ids.size(0)
        action_mask = input_ids.eq(self.action_token_id)  # (B, L)
        device = hidden_states.device
        result = torch.zeros(batch_size, hidden_states.size(-1), device=device, dtype=hidden_states.dtype)
        for i in range(batch_size):
            positions = action_mask[i].nonzero(as_tuple=False)
            if positions.numel() > 0:
                # Use the last <ACT> occurrence if multiple exist.
                pos = positions[-1].item()
                result[i] = hidden_states[i, pos]
            else:
                logger.warning_rank0(f"Sample {i} has no <ACT> token; using zero vector.")
        return result

    def _extract_visual_tokens(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Any],
    ) -> Optional[torch.Tensor]:
        r"""Extract visual tokens from the vision encoder for transformer decoder.

        This method extracts the visual features from Qwen2.5-VL's visual encoder
        before they are merged with text embeddings.

        Args:
            model: The VLM model (potentially wrapped in accelerator).
            inputs: The input dict containing pixel_values, pixel_values_videos, etc.

        Returns:
            Visual tokens tensor of shape ``(batch_size, num_visual_tokens, hidden_size)``
            or None if no visual inputs are provided.
        """
        # Get the actual model (unwrap if needed)
        actual_model = model
        if hasattr(model, "module"):
            actual_model = model.module
        if hasattr(actual_model, "base_model"):
            actual_model = actual_model.base_model
        if hasattr(actual_model, "model"):
            # For PeftModel or wrapped models
            if hasattr(actual_model.model, "visual"):
                actual_model = actual_model.model

        # Check if this is a Qwen2-VL type model
        visual_module = getattr(actual_model, "visual", None)
        if visual_module is None:
            # Try to access through model attribute (transformers >= 4.52.0)
            if hasattr(actual_model, "model") and hasattr(actual_model.model, "visual"):
                visual_module = actual_model.model.visual

        if visual_module is None:
            logger.warning_rank0("Could not find visual module in model. Visual tokens not available.")
            return None

        # Extract visual inputs
        pixel_values = inputs.get("pixel_values", None)
        pixel_values_videos = inputs.get("pixel_values_videos", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        video_grid_thw = inputs.get("video_grid_thw", None)

        if pixel_values is None and pixel_values_videos is None:
            return None

        # Encode visual features through the vision encoder
        visual_features_list = []

        if pixel_values is not None and pixel_values.numel() > 0:
            # Encode image features
            image_features = visual_module(pixel_values, grid_thw=image_grid_thw)
            visual_features_list.append(image_features)

        if pixel_values_videos is not None and pixel_values_videos.numel() > 0:
            # Encode video features
            video_features = visual_module(pixel_values_videos, grid_thw=video_grid_thw)
            visual_features_list.append(video_features)

        if not visual_features_list:
            return None

        # Concatenate all visual features
        # Shape: (total_num_patches, hidden_size)
        all_visual_features = torch.cat(visual_features_list, dim=0)

        # Reshape to per-batch format
        # For simplicity, we assume all samples in the batch have the same visual tokens
        # This is a reasonable assumption for video action classification
        batch_size = inputs["input_ids"].size(0)
        num_visual_tokens = all_visual_features.size(0) // batch_size

        if all_visual_features.size(0) % batch_size != 0:
            # Handle case where visual tokens are not evenly distributed
            # Take average pooling or truncate
            num_visual_tokens = all_visual_features.size(0) // batch_size
            if num_visual_tokens == 0:
                # If fewer visual tokens than batch size, replicate
                all_visual_features = all_visual_features.unsqueeze(0).expand(batch_size, -1, -1)
                return all_visual_features

        # Reshape to (batch_size, num_visual_tokens, hidden_size)
        visual_tokens = all_visual_features.view(batch_size, num_visual_tokens, -1)

        return visual_tokens

    def _action_cls_forward(self, model, inputs):
        r"""Shared forward logic for both training and evaluation.

        Pops ``action_labels`` / ``labels`` from *inputs*, runs the backbone
        with ``output_hidden_states=True``, and returns ``(loss, logits, action_labels)``.
        """
        # Pop action_labels so that the model forward does not receive them.
        action_labels = inputs.pop("action_labels")  # (B,)
        # We do not need language-modelling labels for this stage.
        inputs.pop("labels", None)

        # Extract visual tokens if needed for transformer decoder
        visual_tokens = None
        if self.action_decoder.decoder_type == "transformer":
            visual_tokens = self._extract_visual_tokens(model, inputs)
            if visual_tokens is None:
                raise ValueError(
                    "transformer decoder type requires visual tokens, but none were found. "
                    "Make sure the input contains pixel_values or pixel_values_videos."
                )

        outputs = model(
            **inputs,
            output_hidden_states=True,
        )
        # Last-layer hidden states: (B, L, H)
        last_hidden = outputs.hidden_states[-1]
        action_hidden = self._get_action_hidden_states(inputs["input_ids"], last_hidden)

        # Move action decoder to the same device/dtype as the hidden states.
        self.action_decoder = self.action_decoder.to(device=action_hidden.device, dtype=action_hidden.dtype)

        # Pass visual tokens if using transformer decoder
        if self.action_decoder.decoder_type in ("transformer", "transformer_no_vision"):
            if visual_tokens is not None:
                visual_tokens = visual_tokens.to(device=action_hidden.device, dtype=action_hidden.dtype)
            logits = self.action_decoder(action_hidden, visual_tokens=visual_tokens)
        else:
            logits = self.action_decoder(action_hidden)  # (B, num_classes)

        action_labels = action_labels.to(device=logits.device)
        loss = self.ce_loss(logits, action_labels)

        return loss, logits, action_labels

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        loss, _, _ = self._action_cls_forward(model, inputs)
        return loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional["torch.Tensor"], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Run evaluation forward pass with proper ``action_labels`` handling.

        Returns ``(loss, logits, labels)`` where *labels* are the action class
        indices so that ``compute_metrics`` can calculate accuracy.
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, logits, action_labels = self._action_cls_forward(model, inputs)

        if prediction_loss_only:
            return loss.detach(), None, None

        return loss.detach(), logits.detach(), action_labels.detach()

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    def save_action_decoder(self, output_dir: Optional[str] = None) -> None:
        r"""Save the action decoder weights alongside the main checkpoint."""
        save_dir = output_dir or self.args.output_dir
        self.action_decoder.save_pretrained(save_dir)

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""Save prediction results to ``generated_predictions.jsonl``.

        Each line contains the predicted class, confidence score, and true label.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        logits = predict_results.predictions
        labels = predict_results.label_ids

        if isinstance(logits, tuple):
            logits = logits[0]

        logits = np.asarray(logits)
        labels = np.asarray(labels)

        preds = np.argmax(logits, axis=-1)
        # Compute softmax probabilities for confidence scores.
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        confidences = np.max(probs, axis=-1)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            for i, (pred, conf, label) in enumerate(zip(preds, confidences, labels)):
                if i > 0:
                    writer.write("\n")
                writer.write(json.dumps({
                    "label": int(label),
                    "predict": int(pred),
                    "confidence": round(float(conf), 4),
                }))
