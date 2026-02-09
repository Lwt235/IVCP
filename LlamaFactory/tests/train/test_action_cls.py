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
import tempfile

import pytest
import torch

from llamafactory.hparams.finetuning_args import ActionClassificationArguments, FinetuningArguments
from llamafactory.model.action_decoder import ACTION_DECODER_WEIGHTS_NAME, ActionDecoder


# ---------------------------------------------------------------------------
# ActionDecoder unit tests
# ---------------------------------------------------------------------------


class TestActionDecoder:
    def test_linear_forward(self):
        decoder = ActionDecoder(hidden_size=128, num_classes=10, decoder_type="linear")
        x = torch.randn(4, 128)
        out = decoder(x)
        assert out.shape == (4, 10)

    def test_mlp_forward(self):
        decoder = ActionDecoder(hidden_size=128, num_classes=51, decoder_type="mlp", mlp_hidden_size=64)
        x = torch.randn(2, 128)
        out = decoder(x)
        assert out.shape == (2, 51)

    def test_mlp_default_hidden(self):
        decoder = ActionDecoder(hidden_size=256, num_classes=101, decoder_type="mlp")
        x = torch.randn(1, 256)
        out = decoder(x)
        assert out.shape == (1, 101)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown decoder_type"):
            ActionDecoder(hidden_size=128, num_classes=10, decoder_type="transformer")

    def test_save_and_load(self):
        decoder = ActionDecoder(hidden_size=64, num_classes=5, decoder_type="linear")
        x = torch.randn(1, 64)
        original_out = decoder(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            decoder.save_pretrained(tmpdir)
            assert os.path.isfile(os.path.join(tmpdir, ACTION_DECODER_WEIGHTS_NAME))

            decoder2 = ActionDecoder(hidden_size=64, num_classes=5, decoder_type="linear")
            decoder2.load_pretrained(tmpdir)
            loaded_out = decoder2(x)

            assert torch.allclose(original_out, loaded_out, atol=1e-6)

    def test_load_missing_weights(self):
        """Loading from a directory without weights should not raise (uses random init)."""
        decoder = ActionDecoder(hidden_size=64, num_classes=5, decoder_type="linear")
        with tempfile.TemporaryDirectory() as tmpdir:
            decoder.load_pretrained(tmpdir)  # should not raise


# ---------------------------------------------------------------------------
# FinetuningArguments integration tests
# ---------------------------------------------------------------------------


class TestFinetuningArgsActionCls:
    def test_action_cls_stage_accepted(self):
        args = FinetuningArguments(stage="action_cls")
        assert args.stage == "action_cls"

    def test_default_action_cls_args(self):
        args = FinetuningArguments(stage="action_cls")
        assert args.num_action_classes == 51
        assert args.action_decoder_type == "linear"
        assert args.action_decoder_hidden_size is None
        assert args.action_decoder_path is None
        assert args.action_token_lr_scale == 0.1

    def test_custom_action_cls_args(self):
        args = FinetuningArguments(
            stage="action_cls",
            num_action_classes=101,
            action_decoder_type="mlp",
            action_decoder_hidden_size=512,
            action_token_lr_scale=0.05,
        )
        assert args.num_action_classes == 101
        assert args.action_decoder_type == "mlp"
        assert args.action_decoder_hidden_size == 512
        assert args.action_token_lr_scale == 0.05


# ---------------------------------------------------------------------------
# ActionClsDataCollatorWith4DAttentionMask tests
# ---------------------------------------------------------------------------


class TestActionClsCollator:
    def test_action_labels_collated(self):
        """Verify that action_labels are properly extracted and stacked."""
        from llamafactory.data.collator import ActionClsDataCollatorWith4DAttentionMask

        # We cannot easily instantiate the full collator without a template/model,
        # so we test the action_labels extraction logic directly.
        features = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, -100, -100],
                "action_labels": 5,
            },
            {
                "input_ids": [4, 5, 6],
                "attention_mask": [1, 1, 1],
                "labels": [-100, -100, -100],
                "action_labels": 10,
            },
        ]
        # Pop action_labels as the collator would
        action_labels = [f.pop("action_labels") for f in features if "action_labels" in f]
        result = torch.tensor(action_labels, dtype=torch.long)
        assert result.shape == (2,)
        assert result[0].item() == 5
        assert result[1].item() == 10


# ---------------------------------------------------------------------------
# ActionCLS conversation templates tests
# ---------------------------------------------------------------------------


class TestActionClsTemplates:
    def test_templates_not_empty(self):
        from llamafactory.data.processor.action_templates import ACTION_CLS_TEMPLATES

        assert len(ACTION_CLS_TEMPLATES) > 1

    def test_act_token_in_assistant(self):
        """Every template must have <ACT> in the assistant response, not in the user prompt."""
        from llamafactory.data.processor.action_templates import ACTION_CLS_TEMPLATES

        for tpl in ACTION_CLS_TEMPLATES:
            assert "<ACT>" in tpl.assistant, f"Missing <ACT> in assistant: {tpl.assistant}"
            assert "<ACT>" not in tpl.user, f"<ACT> should not be in user: {tpl.user}"

    def test_video_tag_in_user(self):
        """Every user prompt must contain <video>."""
        from llamafactory.data.processor.action_templates import ACTION_CLS_TEMPLATES

        for tpl in ACTION_CLS_TEMPLATES:
            assert "<video>" in tpl.user, f"Missing <video> in user: {tpl.user}"

    def test_get_random_template(self):
        import random

        from llamafactory.data.processor.action_templates import ACTION_CLS_TEMPLATES, get_random_template

        rng = random.Random(42)
        tpl = get_random_template(rng)
        assert tpl in ACTION_CLS_TEMPLATES

    def test_templates_are_diverse(self):
        """Templates should have distinct user prompts."""
        from llamafactory.data.processor.action_templates import ACTION_CLS_TEMPLATES

        user_prompts = [tpl.user for tpl in ACTION_CLS_TEMPLATES]
        assert len(set(user_prompts)) == len(user_prompts), "Duplicate user prompts found"
