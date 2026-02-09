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

r"""Diverse conversation templates for the ``action_cls`` training stage.

Each template is a ``(user_prompt, assistant_response)`` pair.  The user prompt
should contain ``<video>`` and the assistant response **must** contain the
``<ACT>`` special token so that the trainer can locate its hidden state for
classification.

Multiple templates are provided to avoid model overfitting on a single
question–answer pattern.  During data preparation the caller should randomly
sample from these templates.
"""

from typing import NamedTuple

from .action_cls import ACTION_TOKEN


class ConversationTemplate(NamedTuple):
    """A single user–assistant conversation template."""
    user: str
    assistant: str


# ---------------------------------------------------------------------------
# Conversation templates – the ``<ACT>`` token is always in the assistant part.
# ---------------------------------------------------------------------------

ACTION_CLS_TEMPLATES: list[ConversationTemplate] = [
    ConversationTemplate(
        user="<video>What action is being performed in this video?",
        assistant=f"The action being performed is {ACTION_TOKEN}.",
    ),
    ConversationTemplate(
        user="<video>Describe the activity shown in this video.",
        assistant=f"The activity shown in this video is {ACTION_TOKEN}.",
    ),
    ConversationTemplate(
        user="<video>What is the person doing in this video?",
        assistant=f"The person is doing {ACTION_TOKEN}.",
    ),
    ConversationTemplate(
        user="<video>Can you identify the action in this video?",
        assistant=f"Yes, the action is {ACTION_TOKEN}.",
    ),
    ConversationTemplate(
        user="<video>What activity is taking place in the video?",
        assistant=f"The activity taking place is {ACTION_TOKEN}.",
    ),
    ConversationTemplate(
        user="<video>Please recognize the action in this video clip.",
        assistant=f"The recognized action is {ACTION_TOKEN}.",
    ),
    ConversationTemplate(
        user="<video>What kind of action can you observe in this video?",
        assistant=f"I can observe {ACTION_TOKEN} in this video.",
    ),
    ConversationTemplate(
        user="<video>Tell me what action is happening in this video.",
        assistant=f"The action happening in this video is {ACTION_TOKEN}.",
    ),
    ConversationTemplate(
        user="<video>Identify the action performed in this video.",
        assistant=f"The action performed is {ACTION_TOKEN}.",
    ),
    ConversationTemplate(
        user="<video>What is happening in this video?",
        assistant=f"What is happening is {ACTION_TOKEN}.",
    ),
    ConversationTemplate(
        user="<video>Watch the video and tell me what action is being done.",
        assistant=f"The action being done is {ACTION_TOKEN}.",
    ),
    ConversationTemplate(
        user="<video>Classify the action shown in this video.",
        assistant=f"This video shows {ACTION_TOKEN}.",
    ),
    ConversationTemplate(
        user="<video>What movement or action do you see in this video?",
        assistant=f"I see {ACTION_TOKEN} in this video.",
    ),
    ConversationTemplate(
        user="<video>Briefly describe the action captured in this video.",
        assistant=f"The action captured is {ACTION_TOKEN}.",
    ),
    ConversationTemplate(
        user="<video>What is the main action in this video clip?",
        assistant=f"The main action is {ACTION_TOKEN}.",
    ),
]


def get_random_template(rng=None) -> ConversationTemplate:
    """Return a randomly chosen conversation template.

    Args:
        rng: An optional ``random.Random`` instance for reproducibility.
             If *None*, the module-level ``random`` functions are used.

    Returns:
        A :class:`ConversationTemplate` instance.
    """
    import random

    if rng is None:
        return random.choice(ACTION_CLS_TEMPLATES)
    return rng.choice(ACTION_CLS_TEMPLATES)
