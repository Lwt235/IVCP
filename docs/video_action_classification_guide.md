# Qwen2.5-VL-3B 视频动作分类联合训练指导文档

本文档详细说明如何使用 `copilot/migrate-video-action-classification` 分支中提交的修改，基于 UCF101 和 SomethingSomething 数据集对 Qwen2.5-VL-3B 模型以及新增的 ActionDecoder 进行视频动作分类任务的联合训练。

## 目录

- [系统概述](#系统概述)
- [核心特性](#核心特性)
- [环境准备](#环境准备)
- [数据集准备](#数据集准备)
- [配置文件设置](#配置文件设置)
- [训练流程](#训练流程)
- [模型评估](#模型评估)
- [故障排查](#故障排查)
- [高级配置](#高级配置)

---

## 系统概述

### 架构说明

本系统通过引入新的 `action_cls` 训练阶段，实现了视频动作分类任务的端到端训练。主要组件包括：

1. **Qwen2.5-VL-3B 主干网络**：使用 LoRA 进行参数高效微调
2. **特殊 Token `<ACT>`**：在输入序列中标记动作分类位置
3. **ActionDecoder 解码器头**：轻量级分类器，将 `<ACT>` token 的隐藏状态映射到动作类别
4. **联合训练机制**：通过反向传播同时优化视觉-语言主干和分类头

### 工作流程

```
视频输入 
   ↓
Qwen2.5-VL-3B (LoRA) - 视觉特征提取 + 多模态理解
   ↓
<ACT> token 隐藏状态
   ↓
ActionDecoder
   ↓
动作类别预测 → 交叉熵损失 → 反向传播更新（LoRA + Decoder）
```

---

## 核心特性

### 1. ActionDecoder 分类头

**位置**：`LlamaFactory/src/llamafactory/model/action_decoder.py`

支持四种架构：
- **linear**：单层线性映射 `hidden_size → num_classes`
- **mlp**：两层 MLP 带 GELU 激活 `hidden_size → mlp_hidden_size → num_classes`
- **transformer**：将 ACT token 隐藏状态通过线性投影层后与视觉编码器的视觉 token 合并，经过 transformer layers 和 MLP 输出类别
- **transformer_no_vision**：与 transformer 相同的架构但不使用视觉 token，用于控制变量对比实验

关键方法：
```python
# 初始化 - 基础版本
decoder = ActionDecoder(
    hidden_size=1536,      # Qwen2.5-VL-3B 的隐藏维度
    num_classes=101,       # UCF101 有 101 个类别
    decoder_type="linear",
    mlp_hidden_size=None   # mlp 模式下可选
)

# 初始化 - Transformer 版本（使用视觉 token）
decoder = ActionDecoder(
    hidden_size=1536,
    num_classes=101,
    decoder_type="transformer",  # 或 "transformer_no_vision"
    mlp_hidden_size=768,         # MLP 隐藏层大小
    num_transformer_layers=2,    # Transformer 层数
    num_heads=8,                 # 注意力头数
    dropout=0.1,                 # Dropout 比率
)

# 保存/加载
decoder.save_pretrained("/path/to/checkpoint")
decoder.load_pretrained("/path/to/checkpoint")
```

#### Transformer 解码器架构说明

当使用 `transformer` 或 `transformer_no_vision` 类型时：

1. **线性投影层 G**：将 ACT token 的隐藏状态 H 投影为 H(G)，形状为 `[B, 1, D]`
2. **视觉 token 合并**（仅 `transformer` 类型）：将 H(G) 与 Qwen2.5-VL 视觉编码器编码的视觉 token T_V 合并，H(G) 作为类似 CLS token 的角色
3. **Transformer Layers**：通过可配置数量的 Transformer 层进行特征融合
4. **分类 MLP**：从 H(G) 对应位置提取特征，通过 MLP 输出最终类别

### 2. 数据处理流程

**位置**：`LlamaFactory/src/llamafactory/data/processor/action_cls.py`

数据格式要求：
```json
{
  "messages": [
    {
      "content": "<video>What action is being performed in this video?",
      "role": "user"
    },
    {
      "content": "The action being performed is <ACT>.",
      "role": "assistant"
    }
  ],
  "videos": ["path/to/video.mp4"],
  "action_label": 5
}
```

**关键要点**：
- 必须在助手回复中包含 `<ACT>` token
- `action_label` 字段必须是整数（类别索引，从0开始）
- 所有 token 的标签都设为 `IGNORE_INDEX`，分类损失仅来自 ActionDecoder

### 3. 训练器实现

**位置**：`LlamaFactory/src/llamafactory/train/action_cls/trainer.py`

核心方法：
```python
def compute_loss(self, model, inputs):
    # 1. 提取 action_labels
    action_labels = inputs.pop("action_labels")
    
    # 2. 前向传播，获取隐藏状态
    outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]
    
    # 3. 定位 <ACT> token 位置并提取隐藏状态
    action_hidden = self._get_action_hidden_states(
        inputs["input_ids"], last_hidden
    )
    
    # 4. 通过 ActionDecoder 计算 logits
    logits = self.action_decoder(action_hidden)
    
    # 5. 计算交叉熵损失
    loss = self.ce_loss(logits, action_labels)
    return loss
```

---

## 环境准备

### 1. 系统要求

- **操作系统**：Linux (推荐 Ubuntu 20.04+)
- **GPU**：NVIDIA GPU with CUDA 11.8+ (推荐 A100/V100，至少 24GB 显存)
- **Python**：3.9+
- **存储空间**：至少 500GB（用于数据集和模型）

### 2. 安装依赖

```bash
# 进入 LlamaFactory 目录
cd /path/to/IVCP/LlamaFactory

# 安装 LLaMA Factory
pip install -e ".[torch,metrics]"

# 安装额外依赖
pip install pillow opencv-python decord accelerate deepspeed

# 验证安装
llamafactory-cli version
```

### 3. 下载预训练模型

```bash
# 方法1：从 HuggingFace 下载
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
    --local-dir /path/to/models/Qwen2.5-VL-3B-Instruct

# 方法2：从 ModelScope 下载（国内用户）
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct \
    --local_dir /path/to/models/Qwen2.5-VL-3B-Instruct
```

---

## 数据集准备

### 1. UCF101 数据集

#### 下载数据集
```bash
# 创建数据目录
mkdir -p /data/ucf101
cd /data/ucf101

# 下载视频数据
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
unrar x UCF101.rar

# 下载标注文件
wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unzip UCF101TrainTestSplits-RecognitionTask.zip
```

#### 数据格式转换

创建脚本 `scripts/prepare_ucf101.py`：

```python
import os
import json
from pathlib import Path

def prepare_ucf101_action_cls(
    video_root="/data/ucf101/UCF-101",
    split_file="/data/ucf101/ucfTrainTestlist/trainlist01.txt",
    output_json="LlamaFactory/data/ucf101_train.json"
):
    """
    将 UCF101 数据集转换为 action_cls 训练格式
    """
    # UCF101 类别列表（101个类别）
    with open("/data/ucf101/ucfTrainTestlist/classInd.txt") as f:
        class_to_idx = {}
        for line in f:
            idx, class_name = line.strip().split()
            class_to_idx[class_name] = int(idx) - 1  # 转为0-based索引
    
    samples = []
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            video_path = parts[0]  # 格式: ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
            
            # 提取类别名称
            class_name = video_path.split('/')[0]
            action_label = class_to_idx[class_name]
            
            # 构建完整路径
            full_path = os.path.join(video_root, video_path)
            
            # 构造样本 – 使用多样化模板，<ACT> 放在 assistant 回复中
            from llamafactory.data.processor.action_templates import get_random_template
            tpl = get_random_template()
            sample = {
                "messages": [
                    {
                        "content": tpl.user,
                        "role": "user"
                    },
                    {
                        "content": tpl.assistant,
                        "role": "assistant"
                    }
                ],
                "videos": [full_path],
                "action_label": action_label
            }
            samples.append(sample)
    
    # 保存为 JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(samples)} training samples")
    print(f"Saved to {output_json}")

if __name__ == "__main__":
    # 生成训练集
    prepare_ucf101_action_cls(
        split_file="/data/ucf101/ucfTrainTestlist/trainlist01.txt",
        output_json="LlamaFactory/data/ucf101_train.json"
    )
    
    # 生成测试集
    prepare_ucf101_action_cls(
        split_file="/data/ucf101/ucfTrainTestlist/testlist01.txt",
        output_json="LlamaFactory/data/ucf101_test.json"
    )
```

运行脚本：
```bash
cd /path/to/IVCP
python scripts/prepare_ucf101.py
```

#### 注册数据集

编辑 `LlamaFactory/data/dataset_info.json`，添加：

```json
{
  "ucf101_train": {
    "file_name": "ucf101_train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "videos": "videos"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },
  "ucf101_test": {
    "file_name": "ucf101_test.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "videos": "videos"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

### 2. Something-Something V2 数据集

#### 下载数据集
```bash
mkdir -p /data/something-something-v2
cd /data/something-something-v2

# 从官网下载（需要注册账号）
# https://developer.qualcomm.com/software/ai-datasets/something-something

# 下载后解压
tar -xzf 20bn-something-something-v2.tar.gz
```

#### 数据格式转换

创建脚本 `scripts/prepare_sthsthv2.py`：

```python
import os
import json
from pathlib import Path

def prepare_sthsthv2_action_cls(
    video_root="/data/something-something-v2/videos",
    annotation_file="/data/something-something-v2/train.json",
    label_file="/data/something-something-v2/labels.json",
    output_json="LlamaFactory/data/sthsthv2_train.json"
):
    """
    将 Something-Something V2 数据集转换为 action_cls 格式
    """
    # 加载标签映射
    with open(label_file) as f:
        label_data = json.load(f)
        # 创建 label -> index 映射
        label_to_idx = {label: idx for idx, label in enumerate(sorted(label_data.keys()))}
    
    # 加载标注文件
    with open(annotation_file) as f:
        annotations = json.load(f)
    
    samples = []
    for anno in annotations:
        video_id = anno['id']
        template = anno['template']
        template_filled = anno['template'].replace('[', '').replace(']', '')
        
        action_label = label_to_idx[template]
        
        # 视频路径
        video_path = os.path.join(video_root, f"{video_id}.webm")
        
        # 构造样本 – 使用多样化模板，<ACT> 放在 assistant 回复中
        from llamafactory.data.processor.action_templates import get_random_template
        tpl = get_random_template()
        sample = {
            "messages": [
                {
                    "content": tpl.user,
                    "role": "user"
                },
                {
                    "content": tpl.assistant,
                    "role": "assistant"
                }
            ],
            "videos": [video_path],
            "action_label": action_label
        }
        samples.append(sample)
    
    # 保存为 JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(samples)} training samples")
    print(f"Number of classes: {len(label_to_idx)}")
    print(f"Saved to {output_json}")

if __name__ == "__main__":
    # 生成训练集
    prepare_sthsthv2_action_cls(
        annotation_file="/data/something-something-v2/train.json",
        output_json="LlamaFactory/data/sthsthv2_train.json"
    )
    
    # 生成验证集
    prepare_sthsthv2_action_cls(
        annotation_file="/data/something-something-v2/validation.json",
        output_json="LlamaFactory/data/sthsthv2_val.json"
    )
```

运行脚本：
```bash
cd /path/to/IVCP
python scripts/prepare_sthsthv2.py
```

#### 注册数据集

在 `LlamaFactory/data/dataset_info.json` 中添加：

```json
{
  "sthsthv2_train": {
    "file_name": "sthsthv2_train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "videos": "videos"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },
  "sthsthv2_val": {
    "file_name": "sthsthv2_val.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "videos": "videos"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

---

## 配置文件设置

### 1. UCF101 训练配置

创建 `LlamaFactory/examples/train_lora_action_cls/qwen2_5vl_ucf101.yaml`：

```yaml
### 模型配置
model_name_or_path: /path/to/models/Qwen2.5-VL-3B-Instruct
image_max_pixels: 262144
video_max_pixels: 16384  # UCF101 视频较短，可以适当提高
trust_remote_code: true

### 训练方法
stage: action_cls  # 关键：使用 action_cls 训练阶段
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all
lora_dropout: 0.05
use_dora: true  # 推荐：使用 DoRA 提升性能

### 动作分类参数
num_action_classes: 101  # UCF101 有 101 个类别
action_decoder_type: linear  # 可选: linear, mlp, transformer, transformer_no_vision
action_decoder_hidden_size: null  # mlp/transformer 模式下可设置，如 512
action_decoder_path: null  # 如果有预训练的 decoder 可指定路径
action_token_lr_scale: 0.1  # <ACT> token embedding 的学习率缩放
# transformer/transformer_no_vision 模式下的额外参数：
# action_decoder_num_transformer_layers: 2  # Transformer 层数
# action_decoder_num_heads: 8  # 注意力头数
# action_decoder_dropout: 0.1  # Dropout 比率

### 数据集配置
dataset: ucf101_train  # 使用准备好的 UCF101 训练集
template: qwen2_vl
cutoff_len: 2048
max_samples: null  # 使用全部数据
preprocessing_num_workers: 16
dataloader_num_workers: 4

### 输出配置
output_dir: outputs/qwen2_5vl_action_cls_ucf101
logging_steps: 10
save_steps: 500
save_total_limit: 3
plot_loss: true
overwrite_output_dir: false
save_only_model: false
report_to: wandb  # 可选: none, wandb, tensorboard

### 训练超参数
per_device_train_batch_size: 4
gradient_accumulation_steps: 4  # 有效 batch size = 4 * 4 = 16
learning_rate: 5.0e-5  # 建议对动作分类使用较小的学习率
num_train_epochs: 10
lr_scheduler_type: cosine
warmup_ratio: 0.1
weight_decay: 0.01
bf16: true
ddp_timeout: 180000000

### 优化器配置
flash_attn: fa2  # 使用 FlashAttention-2 加速
enable_liger_kernel: true  # 使用 Liger Kernel 优化

### 评估配置（可选）
# val_size: 0.1  # 从训练集中划分 10% 作为验证集
# per_device_eval_batch_size: 4
# eval_strategy: steps
# eval_steps: 500
# load_best_model_at_end: true
# metric_for_best_model: eval_loss
```

### 2. Something-Something V2 训练配置

创建 `LlamaFactory/examples/train_lora_action_cls/qwen2_5vl_sthsthv2.yaml`：

```yaml
### 模型配置
model_name_or_path: /path/to/models/Qwen2.5-VL-3B-Instruct
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### 训练方法
stage: action_cls
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all
lora_dropout: 0.05
use_dora: true

### 动作分类参数
num_action_classes: 174  # Something-Something V2 有 174 个类别
action_decoder_type: mlp  # 类别较多，使用 MLP 可能效果更好
action_decoder_hidden_size: 768  # MLP 中间层维度
action_decoder_path: null
action_token_lr_scale: 0.1

### 数据集配置
dataset: sthsthv2_train
template: qwen2_vl
cutoff_len: 2048
max_samples: null
preprocessing_num_workers: 16
dataloader_num_workers: 4

### 输出配置
output_dir: outputs/qwen2_5vl_action_cls_sthsthv2
logging_steps: 10
save_steps: 1000
save_total_limit: 3
plot_loss: true
overwrite_output_dir: false
save_only_model: false
report_to: wandb

### 训练超参数
per_device_train_batch_size: 2  # Something-Something 数据集较大，可能需要减小 batch size
gradient_accumulation_steps: 8  # 有效 batch size = 2 * 8 = 16
learning_rate: 5.0e-5
num_train_epochs: 5
lr_scheduler_type: cosine
warmup_ratio: 0.1
weight_decay: 0.01
bf16: true
ddp_timeout: 180000000

### 优化器配置
flash_attn: fa2
enable_liger_kernel: true
```

### 3. Transformer 解码器配置示例（使用视觉 token）

如需使用融合视觉 token 的 Transformer 解码器：

```yaml
### 模型配置
model_name_or_path: /path/to/models/Qwen2.5-VL-3B-Instruct
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### 训练方法
stage: action_cls
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all
use_dora: true

### 动作分类参数 - Transformer 版本
num_action_classes: 101
action_decoder_type: transformer  # 使用融合视觉token的transformer解码器
action_decoder_hidden_size: 768  # MLP隐藏层大小
action_decoder_num_transformer_layers: 2  # Transformer 层数
action_decoder_num_heads: 8  # 注意力头数
action_decoder_dropout: 0.1  # Dropout 比率

### 其他配置与基础版本相同...
```

对于控制变量实验（不使用视觉 token 但保持相同架构），可使用：

```yaml
action_decoder_type: transformer_no_vision  # 相同架构但不融合视觉token
```

---

## 训练流程

### 1. 单 GPU 训练

```bash
cd /path/to/IVCP/LlamaFactory

# UCF101 训练
export CUDA_VISIBLE_DEVICES=0
llamafactory-cli train examples/train_lora_action_cls/qwen2_5vl_ucf101.yaml

# Something-Something V2 训练
llamafactory-cli train examples/train_lora_action_cls/qwen2_5vl_sthsthv2.yaml
```

### 2. 多 GPU 训练 (DDP)

```bash
cd /path/to/IVCP/LlamaFactory

# 使用 torchrun 启动多 GPU 训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 --master_port 29500 \
    -m llamafactory.cli train \
    examples/train_lora_action_cls/qwen2_5vl_ucf101.yaml
```

### 3. 断点续训

```bash
# 在配置文件中指定 checkpoint 路径
llamafactory-cli train examples/train_lora_action_cls/qwen2_5vl_ucf101.yaml \
    --resume_from_checkpoint outputs/qwen2_5vl_action_cls_ucf101/checkpoint-1000
```

或在 YAML 文件中添加：
```yaml
resume_from_checkpoint: outputs/qwen2_5vl_action_cls_ucf101/checkpoint-1000
```

---

## 模型评估

训练完成后，模型检查点包含：
- LoRA 权重：`adapter_model.safetensors`
- ActionDecoder 权重：`action_decoder.safetensors`

可以使用标准的评估流程进行模型评估。

---

## 故障排查

### 常见问题

#### 1. 显存不足 (CUDA Out of Memory)

**症状**：训练时出现 `RuntimeError: CUDA out of memory`

**解决方案**：
```yaml
# 减小 batch size
per_device_train_batch_size: 1
gradient_accumulation_steps: 16

# 减小视频分辨率
video_max_pixels: 8192

# 使用梯度检查点
gradient_checkpointing: true
```

#### 2. <ACT> token 未找到

**症状**：警告 `Sample X has no <ACT> token`

**原因**：输入数据中缺少 `<ACT>` token

**解决方案**：
- 检查数据格式，确保助手回复中包含 `<ACT>`
- 示例：`"content": "The action is <ACT>."`

#### 3. 损失不下降

**症状**：训练损失始终很高或不收敛

**解决方案**：
- 检查学习率：尝试 `1e-4` 到 `1e-5`
- 检查标签：确保 `action_label` 范围在 `[0, num_classes-1]`
- 增加训练轮数
- 尝试不同的 decoder 类型（linear vs mlp）

---

## 高级配置

### 1. 学习率调度

```yaml
# 余弦退火
lr_scheduler_type: cosine
warmup_ratio: 0.1

# 或使用自定义调度器
lr_scheduler_type: polynomial
lr_scheduler_kwargs:
  power: 1.0
```

### 2. LoRA 高级配置

```yaml
# 使用 LoRA+
loraplus_lr_ratio: 16.0

# 使用 DoRA
use_dora: true

# 使用 rsLoRA
use_rslora: true

# 增加 LoRA 秩
lora_rank: 32
lora_alpha: 64
```

---

## 参考资源

### 代码文件

- **ActionDecoder**: `LlamaFactory/src/llamafactory/model/action_decoder.py`
- **数据处理器**: `LlamaFactory/src/llamafactory/data/processor/action_cls.py`
- **训练器**: `LlamaFactory/src/llamafactory/train/action_cls/trainer.py`
- **工作流**: `LlamaFactory/src/llamafactory/train/action_cls/workflow.py`
- **测试用例**: `LlamaFactory/tests/train/test_action_cls.py`

### 外部资源

- **LLaMA Factory**: https://github.com/hiyouga/LLaMA-Factory
- **Qwen2.5-VL**: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- **UCF101**: https://www.crcv.ucf.edu/data/UCF101.php
- **Something-Something V2**: https://developer.qualcomm.com/software/ai-datasets/something-something

---

## 总结

本指导文档详细介绍了如何使用 `copilot/migrate-video-action-classification` 分支中的修改来训练视频动作分类模型。主要步骤包括：

1. ✅ 环境准备和依赖安装
2. ✅ UCF101 和 Something-Something V2 数据集下载和格式转换
3. ✅ 配置文件设置（单数据集、联合训练）
4. ✅ 训练流程（单 GPU、多 GPU）
5. ✅ 模型评估和故障排查

通过遵循本文档，您应该能够成功地对 Qwen2.5-VL-3B 模型进行视频动作分类任务的联合训练。

如有问题，请参考代码库中的测试用例 (`LlamaFactory/tests/train/test_action_cls.py`) 或提交 Issue。
