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

r"""Workflow for the ``action_cls`` training stage.

This module wires together the tokenizer, template, dataset, model, action
decoder, and the ``ActionClassificationTrainer``.
"""

from typing import TYPE_CHECKING, Optional

from ...data import ActionClsDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ...model.action_decoder import ActionDecoder
from ..trainer_utils import create_modelcard_and_push
from .trainer import ActionClassificationTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)

ACTION_TOKEN = "{{action}}"


def run_action_cls(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    # ---- tokenizer & template ------------------------------------------------
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    # Register the {{action}} special token if not already present.
    if ACTION_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [ACTION_TOKEN]})
        logger.info_rank0(f"Added special token {ACTION_TOKEN} to tokenizer (id={tokenizer.convert_tokens_to_ids(ACTION_TOKEN)}).")

    action_token_id: int = tokenizer.convert_tokens_to_ids(ACTION_TOKEN)

    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # ---- dataset -------------------------------------------------------------
    dataset_module = get_dataset(
        template, model_args, data_args, training_args, stage="action_cls", **tokenizer_module
    )

    # ---- model ---------------------------------------------------------------
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)

    # Resize embeddings to accommodate {{action}} token.
    model.resize_token_embeddings(len(tokenizer))

    # ---- action decoder ------------------------------------------------------
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Cannot determine model hidden_size from config.")

    action_decoder = ActionDecoder(
        hidden_size=hidden_size,
        num_classes=finetuning_args.num_action_classes,
        decoder_type=finetuning_args.action_decoder_type,
        mlp_hidden_size=finetuning_args.action_decoder_hidden_size,
    )
    if finetuning_args.action_decoder_path is not None:
        action_decoder.load_pretrained(finetuning_args.action_decoder_path)

    # ---- data collator -------------------------------------------------------
    data_collator = ActionClsDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not getattr(training_args, "predict_with_generate", False) else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # ---- trainer -------------------------------------------------------------
    trainer = ActionClassificationTrainer(
        model=model,
        args=training_args,
        action_decoder=action_decoder,
        action_token_id=action_token_id,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    # ---- train / eval --------------------------------------------------------
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.save_action_decoder()

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss"])

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ---- model card ----------------------------------------------------------
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
