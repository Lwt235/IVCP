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

r"""Metrics for the ``action_cls`` evaluation stage.

Provides top-1 and top-5 classification accuracy computed from the logits
returned by :class:`ActionClassificationTrainer.prediction_step`.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np


if TYPE_CHECKING:
    from transformers import EvalPrediction


@dataclass
class ComputeActionAccuracy:
    r"""Compute top-1 and top-5 action classification accuracy.

    Designed to be passed as ``compute_metrics`` to the HuggingFace
    :class:`~transformers.Trainer`.  Expects ``EvalPrediction`` with:

    - ``predictions``: logits of shape ``(N, num_classes)``
    - ``label_ids``:   integer labels of shape ``(N,)``
    """

    top1: list[float] = field(default_factory=list)
    top5: list[float] = field(default_factory=list)

    def __call__(
        self, eval_preds: "EvalPrediction", compute_result: bool = True
    ) -> Optional[dict[str, float]]:
        logits, labels = eval_preds.predictions, eval_preds.label_ids
        if isinstance(logits, tuple):
            logits = logits[0]

        logits = np.asarray(logits)
        labels = np.asarray(labels)

        # Top-1 accuracy
        preds = np.argmax(logits, axis=-1)
        self.top1.extend((preds == labels).tolist())

        # Top-5 accuracy (clamp k to number of classes)
        num_classes = logits.shape[-1]
        k = min(5, num_classes)
        top_k_indices = np.argsort(logits, axis=-1)[:, -k:]
        top_k_correct = np.any(top_k_indices == labels[:, None], axis=-1)
        self.top5.extend(top_k_correct.tolist())

        if compute_result:
            return self._dump()
        return None

    def _dump(self) -> dict[str, float]:
        result = {}
        if self.top1:
            result["action_accuracy_top1"] = float(np.mean(self.top1))
        if self.top5:
            result["action_accuracy_top5"] = float(np.mean(self.top5))
        self.top1.clear()
        self.top5.clear()
        return result
