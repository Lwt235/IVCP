# IVCP 项目使用说明

## 概述

本文档说明了本次修改的内容以及如何使用集成到项目中的 LLaMA Factory 框架进行视频理解和动作分类任务。

## 修改内容

本次提交（Revert PR）向项目中添加了以下内容：

### 1. LLaMA Factory 框架集成

添加了完整的 LLaMA Factory 框架，这是一个功能强大的大语言模型微调工具，支持：

- **多种模型**：支持 LLaMA、Qwen3-VL、Mistral、GLM 等 100+ 种大语言模型和多模态模型
- **多种训练方法**：
  - 预训练（Pretraining）
  - 监督微调（SFT - Supervised Fine-Tuning）
  - 奖励模型训练（Reward Modeling）
  - PPO、DPO、KTO、ORPO 等强化学习方法
- **灵活的微调方式**：
  - 全参数微调（Full Fine-tuning）
  - LoRA 微调
  - QLoRA 微调（支持 4/8 比特量化）
- **多模态能力**：支持图像、视频、音频的理解和处理

### 2. VideoEspresso 数据集

添加了用于视频理解训练的 VideoEspresso 数据集，包含三个训练阶段：

- **VideoEspresso_stage1.json**：第一阶段训练数据（约 31,587 条样本）
- **VideoEspresso_stage2.json**：第二阶段训练数据（约 40,132 条样本）
- **VideoEspresso_stage3.json**：第三阶段训练数据（约 3,762 条样本）

数据集特点：
- 包含视频问答对话数据
- 支持视频时间序列理解
- 包含复杂的推理和分析任务

### 3. 训练配置示例

项目包含了多种训练配置文件，位于 `LlamaFactory/examples/` 目录下：

- `train_full/`：全参数微调配置
- `train_lora/`：LoRA 微调配置
- `train_qlora/`：量化 LoRA 微调配置
- `deepspeed/`：DeepSpeed 分布式训练配置
- `accelerate/`：Accelerate 加速训练配置

## 安装和环境配置

### 1. 安装依赖

```bash
cd LlamaFactory
pip install -e ".[torch,metrics]"
```

如需支持多模态训练（视频/图像），安装额外依赖：

```bash
pip install -e ".[torch,metrics,deepspeed,bitsandbytes,vllm,liger-kernel]"
```

### 2. 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8（推荐）
- GPU 显存：建议 24GB+ 用于全参数微调，8GB+ 用于 LoRA 微调

## 使用方法

### 方式一：命令行训练

#### 1. 视频理解模型微调（使用 VideoEspresso 数据集）

使用 Qwen3-VL 模型进行视频理解任务的 LoRA 微调：

```bash
cd LlamaFactory

# 使用 LoRA 微调
llamafactory-cli train examples/train_lora/qwen3vl_lora_sft.yaml
```

#### 2. 自定义训练配置

创建自己的配置文件（例如 `my_video_training.yaml`）：

```yaml
### model
model_name_or_path: Qwen/Qwen3-VL-4B-Instruct
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

### dataset
dataset: mllm_video_demo  # 或使用自定义数据集
template: qwen3_vl
cutoff_len: 2048
max_samples: 1000

### output
output_dir: saves/my_video_model
logging_steps: 10
save_steps: 500

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
```

然后运行：

```bash
llamafactory-cli train my_video_training.yaml
```

### 方式二：使用 Web UI（LLaMA Board）

LLaMA Factory 提供了图形化界面，无需编写代码即可进行模型微调：

```bash
cd LlamaFactory
llamafactory-cli webui
```

然后在浏览器中打开显示的地址（通常是 `http://localhost:7860`），通过界面配置和启动训练。

### 方式三：使用 Python API

```python
from llamafactory.train.train import run_train

# 配置训练参数
args = {
    "model_name_or_path": "Qwen/Qwen3-VL-4B-Instruct",
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "dataset": "mllm_video_demo",
    "template": "qwen3_vl",
    "output_dir": "saves/my_model",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "num_train_epochs": 3,
    "bf16": True,
}

# 启动训练
run_train(args)
```

## 数据集准备

### 使用内置的 VideoEspresso 数据集

数据集已经包含在项目中，可以直接使用。在配置文件中设置：

```yaml
dataset: VideoEspresso_stage1  # 或 VideoEspresso_stage2, VideoEspresso_stage3
```

### 准备自定义视频数据集

1. 在 `LlamaFactory/data/` 目录下创建你的数据 JSON 文件

2. 数据格式示例（`my_video_data.json`）：

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<video>视频中的人在做什么动作？"
      },
      {
        "from": "gpt",
        "value": "视频中的人正在跑步。"
      }
    ],
    "videos": [
      "path/to/your/video.mp4"
    ]
  }
]
```

3. 在 `LlamaFactory/data/dataset_info.json` 中注册数据集：

```json
{
  "my_video_data": {
    "file_name": "my_video_data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "videos": "videos"
    }
  }
}
```

4. 在训练配置中使用：

```yaml
dataset: my_video_data
```

## 模型推理和部署

### 1. 命令行推理

训练完成后，使用模型进行推理：

```bash
llamafactory-cli chat \
    --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
    --adapter_name_or_path saves/my_video_model \
    --template qwen3_vl
```

### 2. 部署为 API 服务

使用 vLLM 部署高性能推理服务：

```bash
llamafactory-cli api \
    --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
    --adapter_name_or_path saves/my_video_model \
    --template qwen3_vl
```

这将启动一个 OpenAI 兼容的 API 服务器。

## 高级功能

### 1. 分布式训练

使用 DeepSpeed 进行多GPU训练：

```yaml
deepspeed: examples/deepspeed/ds_z3_config.json
```

### 2. 实验追踪

支持多种实验追踪工具：

```yaml
report_to: wandb  # 或 tensorboard, mlflow, swanlab
```

### 3. 模型量化

训练量化模型以减少显存占用：

```bash
llamafactory-cli train examples/train_qlora/qwen3_lora_sft_bnb.yaml
```

## 目录结构

```
LlamaFactory/
├── data/                          # 数据集目录
│   ├── VideoEspresso_stage1.json  # VideoEspresso 第一阶段数据
│   ├── VideoEspresso_stage2.json  # VideoEspresso 第二阶段数据
│   ├── VideoEspresso_stage3.json  # VideoEspresso 第三阶段数据
│   ├── dataset_info.json          # 数据集配置文件
│   └── ...
├── examples/                      # 训练配置示例
│   ├── train_full/                # 全参数微调示例
│   ├── train_lora/                # LoRA 微调示例
│   ├── train_qlora/               # 量化 LoRA 示例
│   ├── deepspeed/                 # DeepSpeed 配置
│   └── ...
├── src/                           # 源代码
└── scripts/                       # 辅助脚本
```

## 常见问题

### Q1: 显存不足怎么办？

**方案 1**：使用 LoRA 微调代替全参数微调

```yaml
finetuning_type: lora
lora_rank: 8
```

**方案 2**：减小批次大小

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

**方案 3**：使用量化训练（QLoRA）

```yaml
quantization_bit: 4
```

**方案 4**：使用 DeepSpeed ZeRO-3 优化

```yaml
deepspeed: examples/deepspeed/ds_z3_offload_config.json
```

### Q2: 如何加速训练？

1. 使用 FlashAttention-2：自动启用（如果环境支持）
2. 使用更大的批次大小
3. 使用混合精度训练（bf16 或 fp16）
4. 多GPU分布式训练

### Q3: 训练好的模型在哪里？

模型保存在配置文件中指定的 `output_dir` 目录下：

```
saves/my_model/
├── checkpoint-500/
├── checkpoint-1000/
└── adapter_model/  # LoRA 权重
```

## 相关资源

- **LLaMA Factory 官方文档**：https://llamafactory.readthedocs.io/zh-cn/latest/
- **LLaMA Factory GitHub**：https://github.com/hiyouga/LLaMA-Factory
- **Qwen3-VL 模型**：https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- **技术博客**：https://blog.llamafactory.net/

## 技术支持

如有问题，可以：

1. 查看 LLaMA Factory 官方文档
2. 在 GitHub 上提交 Issue
3. 加入 LLaMA Factory 社区群组

## 许可证

本项目中的 LLaMA Factory 框架遵循 Apache 2.0 许可证。详见 `LlamaFactory/LICENSE` 文件。
