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

r"""Dataset processor for the ``action_cls`` training stage.

Each example is expected to carry the standard LlamaFactory fields produced by
``align_dataset`` (``_prompt``, ``_response``, ``_system``, …) **plus** a
numeric ``action_label`` that will be used as the classification target.

The processor builds the token-level inputs in the same way as the supervised
processor, but:

* Prompt tokens are masked (``IGNORE_INDEX``) in ``labels`` while response
  tokens retain their IDs so that an optional token-level language modelling
  loss can help the model learn to use the ``<ACT>`` token in context.
* The integer ``action_label`` is stored in the output batch and later used by
  the ``ActionClassificationTrainer`` to compute the cross-entropy loss on the
  ``<ACT>`` hidden state.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)

ACTION_TOKEN = "<ACT>"


@dataclass
class ActionClassificationDatasetProcessor(DatasetProcessor):
    r"""Build inputs for the video action classification stage.

    The final ``input_ids`` follows the same layout as supervised training (the
    prompt encourages reasoning).  Prompt tokens are masked with ``IGNORE_INDEX``
    while response tokens retain their IDs so that an optional token-level
    language modelling loss can be computed alongside the classification loss
    from the ``<ACT>`` hidden state.
    """

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(
            prompt + response, images, videos, audios, self.processor
        )
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)

        for source_ids, target_ids in encoded_pairs:
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            input_ids += source_ids + target_ids
            # Mask prompt tokens; keep response tokens for optional token-level LM loss.
            labels += [IGNORE_INDEX] * source_len + target_ids

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [IGNORE_INDEX]

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs: dict[str, list[Any]] = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            # Retrieve the integer action label from the dataset.
            action_label = examples.get("action_label", [None])[i]
            if action_label is None:
                logger.warning_rank0(f"Dropped example {i}: missing 'action_label'.")
                logger.warning_rank0(f"Example data: {examples}")
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])
            model_inputs["action_labels"].append(int(action_label))

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print(
            "inputs:\n{}".format(
                self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)
            )
        )
        print("action_label: {}".format(example.get("action_labels", "N/A")))
