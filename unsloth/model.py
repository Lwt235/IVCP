import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
from unsloth import FastVisionModel
from unsloth_zoo import vision_utils
from myDataset import VideoQADataset
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

if __name__ == "__main__":

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = "/home/liwt/IVCP/model/unsloth/Qwen2.5-VL-3B-Instruct",
        max_seq_length = 4096,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers      = False,
        finetune_language_layers    = True,
        finetune_attention_modules  = True,
        finetune_mlp_modules        = True,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0.05,
        bias = "none",
        random_state = 42,
        use_rslora = False,
        loftq_config = None,
        use_gradient_checkpointing = False,
    )

    dataset = VideoQADataset("/home/liwt/IVCP/data/VideoEspresso", "train_stage2.json")

    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = dataset,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1,
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 42,
            output_dir = "/home/liwt/IVCP/unsloth/outputs/vqa_stage1",
            report_to = "wandb",

            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),

            dataset_text_field = "",
            dataset_kwargs = {"skip_special_tokens": True},
            dataset_num_proc = 4,
            max_length = 2048,

            # auto_find_batch_size = True,
            # gradient_checkpointing = False,

            packing = False
        ),
    )

    # ============================================
    # 5. 开始训练
    # ============================================
    trainer_stats = trainer.train()

    # ============================================
    # 6. 保存模型
    # ============================================
    # 保存LoRA适配器
    model.save_pretrained("qwen2_5vl_video_lora")
    tokenizer.save_pretrained("qwen2_5vl_video_lora")

    # 保存为合并后的16bit模型
    model.save_pretrained_merged("qwen2_5vl_video_merged_16bit", tokenizer, save_method="merged_16bit")


    # messages = dataset[0]['messages']

    # image_input, video_input, video_kwargs = vision_utils.process_vision_info(messages, return_video_kwargs=True)
    # input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenizer=False)
    # inputs = tokenizer(text=input_text, images=image_input, videos=video_input, return_tensors="pt", **video_kwargs).to("cuda")

    # from transformers import TextStreamer
    # text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    # _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256, use_cache=True, temperature=1.5, min_p=0.1)

    # from collections import Counter
    # import numpy as np

    # lengths = []
    # for i in range(min(200, len(dataset))):  # 先看前200个样本
    #     sample = dataset[i]['messages']
    #     prompt = tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=True)
    #     inputs = tokenizer(text=prompt, return_tensors="pt")  # 先不加视频，粗估文本部分
    #     lengths.append(inputs.input_ids.shape[1])

    # print("文本部分 token 长度统计（不含视频）：")
    # print("平均:", np.mean(lengths))
    # print("中位数:", np.median(lengths))
    # print("90分位:", np.percentile(lengths, 90))
    # print("最大:", max(lengths))
    # print("超过 1000 token 的样本比例:", sum(l > 1000 for l in lengths) / len(lengths))