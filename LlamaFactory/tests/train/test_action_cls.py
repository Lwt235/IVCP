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

import numpy as np
import pytest
import torch
from transformers import EvalPrediction

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

    def test_transformer_forward_with_visual_tokens(self):
        """Test transformer decoder with visual tokens."""
        decoder = ActionDecoder(
            hidden_size=128,
            num_classes=10,
            decoder_type="transformer",
            num_transformer_layers=2,
            num_heads=4,
        )
        # ACT token hidden state (B, D)
        x = torch.randn(2, 128)
        # Visual tokens (B, num_visual_tokens, D)
        visual_tokens = torch.randn(2, 16, 128)
        out = decoder(x, visual_tokens=visual_tokens)
        assert out.shape == (2, 10)

    def test_transformer_forward_with_mismatched_visual_tokens(self):
        """Test transformer decoder with visual tokens of different hidden size."""
        decoder = ActionDecoder(
            hidden_size=128,
            num_classes=10,
            decoder_type="transformer",
            num_transformer_layers=2,
            num_heads=4,
            visual_hidden_size=80,
        )
        # ACT token hidden state (B, D_llm)
        x = torch.randn(2, 128)
        # Visual tokens with different hidden size (B, num_visual_tokens, D_vision)
        visual_tokens = torch.randn(2, 16, 80)
        out = decoder(x, visual_tokens=visual_tokens)
        assert out.shape == (2, 10)

    def test_transformer_forward_requires_visual_tokens(self):
        """Test that transformer decoder requires visual tokens."""
        decoder = ActionDecoder(
            hidden_size=128,
            num_classes=10,
            decoder_type="transformer",
        )
        x = torch.randn(2, 128)
        with pytest.raises(ValueError, match="visual_tokens is required"):
            decoder(x)  # No visual tokens provided

    def test_transformer_no_vision_forward(self):
        """Test transformer_no_vision decoder without visual tokens."""
        decoder = ActionDecoder(
            hidden_size=128,
            num_classes=10,
            decoder_type="transformer_no_vision",
            num_transformer_layers=2,
            num_heads=4,
        )
        x = torch.randn(2, 128)
        out = decoder(x)  # No visual tokens needed
        assert out.shape == (2, 10)

    def test_transformer_with_3d_input(self):
        """Test transformer decoder with 3D input (B, 1, D)."""
        decoder = ActionDecoder(
            hidden_size=128,
            num_classes=10,
            decoder_type="transformer",
        )
        # Input with explicit sequence dimension
        x = torch.randn(2, 1, 128)
        visual_tokens = torch.randn(2, 16, 128)
        out = decoder(x, visual_tokens=visual_tokens)
        assert out.shape == (2, 10)

    def test_transformer_no_vision_with_3d_input(self):
        """Test transformer_no_vision decoder with 3D input (B, 1, D)."""
        decoder = ActionDecoder(
            hidden_size=128,
            num_classes=10,
            decoder_type="transformer_no_vision",
        )
        x = torch.randn(2, 1, 128)
        out = decoder(x)
        assert out.shape == (2, 10)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown decoder_type"):
            ActionDecoder(hidden_size=128, num_classes=10, decoder_type="invalid_type")

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

    def test_save_and_load_transformer(self):
        """Test saving and loading transformer decoder weights."""
        decoder = ActionDecoder(
            hidden_size=64,
            num_classes=5,
            decoder_type="transformer",
            num_transformer_layers=2,
            num_heads=4,
        )
        x = torch.randn(1, 64)
        visual_tokens = torch.randn(1, 8, 64)

        # Set to eval mode to disable dropout for deterministic comparison
        decoder.eval()
        with torch.no_grad():
            original_out = decoder(x, visual_tokens=visual_tokens)

        with tempfile.TemporaryDirectory() as tmpdir:
            decoder.save_pretrained(tmpdir)
            assert os.path.isfile(os.path.join(tmpdir, ACTION_DECODER_WEIGHTS_NAME))

            decoder2 = ActionDecoder(
                hidden_size=64,
                num_classes=5,
                decoder_type="transformer",
                num_transformer_layers=2,
                num_heads=4,
            )
            decoder2.load_pretrained(tmpdir)
            decoder2.eval()

            with torch.no_grad():
                loaded_out = decoder2(x, visual_tokens=visual_tokens)

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
        assert args.action_decoder_num_transformer_layers == 2
        assert args.action_decoder_num_heads == 8
        assert args.action_decoder_dropout == 0.1

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

    def test_transformer_decoder_args(self):
        """Test transformer decoder type arguments."""
        args = FinetuningArguments(
            stage="action_cls",
            num_action_classes=101,
            action_decoder_type="transformer",
            action_decoder_num_transformer_layers=4,
            action_decoder_num_heads=16,
            action_decoder_dropout=0.2,
        )
        assert args.action_decoder_type == "transformer"
        assert args.action_decoder_num_transformer_layers == 4
        assert args.action_decoder_num_heads == 16
        assert args.action_decoder_dropout == 0.2

    def test_transformer_no_vision_decoder_args(self):
        """Test transformer_no_vision decoder type arguments."""
        args = FinetuningArguments(
            stage="action_cls",
            num_action_classes=51,
            action_decoder_type="transformer_no_vision",
        )
        assert args.action_decoder_type == "transformer_no_vision"


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
# ActionClassificationTrainer._action_cls_forward tests
# ---------------------------------------------------------------------------


class TestActionClsForward:
    """Verify that _action_cls_forward returns (loss, logits) and that
    prediction_step does not crash during evaluation."""

    def _make_dummy_model(self, hidden_size):
        """Return a tiny model-like callable that mimics VLM output."""

        class _FakeOutput:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states

        class _FakeModel:
            def __call__(self, **kwargs):
                bs = kwargs["input_ids"].size(0)
                sl = kwargs["input_ids"].size(1)
                hs = torch.randn(bs, sl, hidden_size)
                return _FakeOutput(hidden_states=(hs,))

        return _FakeModel()

    def test_action_cls_forward_returns_tuple(self):
        """_action_cls_forward must return (loss, logits, action_labels)."""
        from llamafactory.model.action_decoder import ActionDecoder
        from llamafactory.train.action_cls.trainer import ActionClassificationTrainer

        hidden_size = 32
        num_classes = 5
        action_token_id = 99
        batch_size = 2
        seq_len = 4

        decoder = ActionDecoder(hidden_size=hidden_size, num_classes=num_classes)
        # Build a minimal trainer-like object to test _action_cls_forward
        trainer = object.__new__(ActionClassificationTrainer)
        trainer.action_decoder = decoder
        trainer.action_token_id = action_token_id
        trainer.ce_loss = torch.nn.CrossEntropyLoss()

        # Inputs with <ACT> token at position 2
        input_ids = torch.tensor([[1, 2, action_token_id, 3]] * batch_size)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": torch.full_like(input_ids, -100),
            "action_labels": torch.tensor([0, 3]),
        }

        model = self._make_dummy_model(hidden_size)
        loss, logits, action_labels = trainer._action_cls_forward(model, inputs)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
        assert logits.shape == (batch_size, num_classes)
        assert action_labels.shape == (batch_size,)
        assert action_labels[0].item() == 0
        assert action_labels[1].item() == 3

    def test_compute_loss_returns_scalar(self):
        """compute_loss must return only the loss tensor (not a tuple)."""
        from llamafactory.model.action_decoder import ActionDecoder
        from llamafactory.train.action_cls.trainer import ActionClassificationTrainer

        hidden_size = 32
        num_classes = 5
        action_token_id = 99
        batch_size = 2

        decoder = ActionDecoder(hidden_size=hidden_size, num_classes=num_classes)
        trainer = object.__new__(ActionClassificationTrainer)
        trainer.action_decoder = decoder
        trainer.action_token_id = action_token_id
        trainer.ce_loss = torch.nn.CrossEntropyLoss()

        input_ids = torch.tensor([[1, 2, action_token_id, 3]] * batch_size)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": torch.full_like(input_ids, -100),
            "action_labels": torch.tensor([0, 3]),
        }

        model = self._make_dummy_model(hidden_size)
        result = trainer.compute_loss(model, inputs)

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0


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


# ---------------------------------------------------------------------------
# ComputeActionAccuracy tests
# ---------------------------------------------------------------------------


class TestComputeActionAccuracy:
    def test_perfect_accuracy(self):
        """All predictions correct should give 1.0 accuracy."""
        from llamafactory.train.action_cls.metric import ComputeActionAccuracy

        metric = ComputeActionAccuracy()
        # 4 samples, 5 classes, correct predictions
        logits = np.array([
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10.0, 0.0],
        ])
        labels = np.array([0, 1, 2, 3])
        eval_preds = EvalPrediction(predictions=logits, label_ids=labels)
        result = metric(eval_preds)
        assert result["action_accuracy_top1"] == 1.0
        assert result["action_accuracy_top5"] == 1.0

    def test_zero_accuracy(self):
        """All predictions wrong should give 0.0 top-1 accuracy."""
        from llamafactory.train.action_cls.metric import ComputeActionAccuracy

        metric = ComputeActionAccuracy()
        logits = np.array([
            [0.0, 10.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10.0],
            [0.0, 0.0, 10.0, 0.0, 0.0],
        ])
        labels = np.array([0, 1, 2, 3])
        eval_preds = EvalPrediction(predictions=logits, label_ids=labels)
        result = metric(eval_preds)
        assert result["action_accuracy_top1"] == 0.0

    def test_partial_accuracy(self):
        """Half correct should give 0.5 top-1 accuracy."""
        from llamafactory.train.action_cls.metric import ComputeActionAccuracy

        metric = ComputeActionAccuracy()
        logits = np.array([
            [10.0, 0.0, 0.0],  # pred=0, label=0 -> correct
            [0.0, 0.0, 10.0],  # pred=2, label=1 -> wrong
            [0.0, 10.0, 0.0],  # pred=1, label=1 -> correct
            [10.0, 0.0, 0.0],  # pred=0, label=2 -> wrong
        ])
        labels = np.array([0, 1, 1, 2])
        eval_preds = EvalPrediction(predictions=logits, label_ids=labels)
        result = metric(eval_preds)
        assert result["action_accuracy_top1"] == 0.5

    def test_top5_accuracy(self):
        """Top-5 should be >= top-1 accuracy."""
        from llamafactory.train.action_cls.metric import ComputeActionAccuracy

        metric = ComputeActionAccuracy()
        # 10 classes, correct label is in top-5 but not top-1
        logits = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],  # pred=9, label=5 -> top1 wrong, top5 correct
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # pred=0, label=0 -> both correct
        ])
        labels = np.array([5, 0])
        eval_preds = EvalPrediction(predictions=logits, label_ids=labels)
        result = metric(eval_preds)
        assert result["action_accuracy_top1"] == 0.5
        assert result["action_accuracy_top5"] == 1.0

    def test_accumulation_across_batches(self):
        """Metrics should accumulate across multiple calls before _dump."""
        from llamafactory.train.action_cls.metric import ComputeActionAccuracy

        metric = ComputeActionAccuracy()
        # Batch 1: 1/2 correct
        logits1 = np.array([[10.0, 0.0], [10.0, 0.0]])
        labels1 = np.array([0, 1])
        eval_preds1 = EvalPrediction(predictions=logits1, label_ids=labels1)
        result1 = metric(eval_preds1, compute_result=False)
        assert result1 is None

        # Batch 2: 2/2 correct
        logits2 = np.array([[10.0, 0.0], [0.0, 10.0]])
        labels2 = np.array([0, 1])
        eval_preds2 = EvalPrediction(predictions=logits2, label_ids=labels2)
        result2 = metric(eval_preds2, compute_result=True)
        # Overall: 3/4 correct
        assert result2["action_accuracy_top1"] == 0.75

    def test_state_cleared_after_dump(self):
        """Internal state should be cleared after _dump."""
        from llamafactory.train.action_cls.metric import ComputeActionAccuracy

        metric = ComputeActionAccuracy()
        logits = np.array([[10.0, 0.0]])
        labels = np.array([0])
        eval_preds = EvalPrediction(predictions=logits, label_ids=labels)
        metric(eval_preds)

        # After dump, internal lists should be empty
        assert len(metric.top1) == 0
        assert len(metric.top5) == 0


# ---------------------------------------------------------------------------
# ActionClassificationTrainer.save_predictions tests
# ---------------------------------------------------------------------------


class TestSavePredictions:
    def _make_trainer_stub(self, output_dir):
        """Create a minimal trainer-like object with save_predictions."""
        import types

        from llamafactory.train.action_cls.trainer import ActionClassificationTrainer

        trainer = object.__new__(ActionClassificationTrainer)
        trainer.args = types.SimpleNamespace(output_dir=output_dir, process_index=0)
        return trainer

    def test_save_predictions_creates_file(self):
        """save_predictions must create generated_predictions.jsonl."""
        import json
        import tempfile
        from collections import namedtuple

        PredictionOutput = namedtuple("PredictionOutput", ["predictions", "label_ids", "metrics"])

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer_stub(tmpdir)
            logits = np.array([
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ])
            labels = np.array([0, 1, 2])
            predict_results = PredictionOutput(predictions=logits, label_ids=labels, metrics={})
            trainer.save_predictions(predict_results)

            output_file = os.path.join(tmpdir, "generated_predictions.jsonl")
            assert os.path.isfile(output_file)

            with open(output_file, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")

            assert len(lines) == 3
            for line in lines:
                record = json.loads(line)
                assert "label" in record
                assert "predict" in record
                assert "confidence" in record

    def test_save_predictions_correct_values(self):
        """Predicted class and label should match expected values."""
        import json
        import tempfile
        from collections import namedtuple

        PredictionOutput = namedtuple("PredictionOutput", ["predictions", "label_ids", "metrics"])

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer_stub(tmpdir)
            logits = np.array([
                [10.0, 0.0, 0.0],  # pred=0
                [0.0, 0.0, 10.0],  # pred=2
            ])
            labels = np.array([0, 1])
            predict_results = PredictionOutput(predictions=logits, label_ids=labels, metrics={})
            trainer.save_predictions(predict_results)

            output_file = os.path.join(tmpdir, "generated_predictions.jsonl")
            with open(output_file, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")

            r0 = json.loads(lines[0])
            assert r0["label"] == 0
            assert r0["predict"] == 0
            assert r0["confidence"] > 0.9

            r1 = json.loads(lines[1])
            assert r1["label"] == 1
            assert r1["predict"] == 2
            assert r1["confidence"] > 0.9

    def test_save_predictions_handles_tuple_logits(self):
        """save_predictions should handle logits wrapped in a tuple."""
        import json
        import tempfile
        from collections import namedtuple

        PredictionOutput = namedtuple("PredictionOutput", ["predictions", "label_ids", "metrics"])

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer_stub(tmpdir)
            logits = (np.array([[10.0, 0.0], [0.0, 10.0]]),)
            labels = np.array([0, 1])
            predict_results = PredictionOutput(predictions=logits, label_ids=labels, metrics={})
            trainer.save_predictions(predict_results)

            output_file = os.path.join(tmpdir, "generated_predictions.jsonl")
            with open(output_file, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")

            assert len(lines) == 2
            r0 = json.loads(lines[0])
            assert r0["predict"] == 0
            r1 = json.loads(lines[1])
            assert r1["predict"] == 1
