# 视频动作分类训练文档

本目录包含了使用 `copilot/migrate-video-action-classification` 分支的修改进行视频动作分类训练的完整文档和示例。

## 📖 主要文档

- **[视频动作分类联合训练指导文档](video_action_classification_guide.md)** - 完整的训练指南

## 🎯 核心功能

### 新增的 `action_cls` 训练阶段

本次修改引入了专门用于视频动作分类的训练阶段，主要特性包括：

1. **ActionDecoder 分类头** (`LlamaFactory/src/llamafactory/model/action_decoder.py`)
   - 支持 `linear` 和 `mlp` 两种架构
   - 轻量级设计，易于训练和部署
   - 支持保存和加载预训练权重

2. **特殊 Token `<ACT>`**
   - 在输入序列中标记动作分类位置
   - 系统自动提取该 token 的隐藏状态用于分类

3. **联合训练机制**
   - 同时优化视觉-语言主干（通过 LoRA）和分类头
   - 端到端的训练流程
   - 支持多数据集混合训练

## 📁 文件结构

```
docs/
├── README.md                                    # 本文件
└── video_action_classification_guide.md        # 详细指导文档

LlamaFactory/
├── src/llamafactory/
│   ├── model/action_decoder.py                  # ActionDecoder 实现
│   ├── data/processor/action_cls.py             # 数据处理器
│   └── train/action_cls/                        # 训练模块
│       ├── trainer.py                           # 自定义 Trainer
│       └── workflow.py                          # 训练工作流
├── tests/train/test_action_cls.py               # 单元测试
└── examples/train_lora_action_cls/              # 配置示例
    ├── qwen2_5vl_ucf101.yaml                    # UCF101 配置
    └── qwen2_5vl_sthsthv2.yaml                  # Something-Something 配置
```

## 🚀 快速开始

### 1. 准备数据

参考指导文档中的 [数据集准备](video_action_classification_guide.md#数据集准备) 章节，准备 UCF101 或 Something-Something V2 数据集。

### 2. 配置训练

使用提供的示例配置文件或根据需要自定义：

```bash
# 编辑配置文件
vi LlamaFactory/examples/train_lora_action_cls/qwen2_5vl_ucf101.yaml

# 关键参数：
# - model_name_or_path: Qwen2.5-VL-3B-Instruct 模型路径
# - num_action_classes: 动作类别数（UCF101=101, SthSthV2=174）
# - dataset: 数据集名称
```

### 3. 启动训练

```bash
cd LlamaFactory

# 单 GPU 训练
export CUDA_VISIBLE_DEVICES=0
llamafactory-cli train examples/train_lora_action_cls/qwen2_5vl_ucf101.yaml

# 多 GPU 训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 --master_port 29500 \
    -m llamafactory.cli train \
    examples/train_lora_action_cls/qwen2_5vl_ucf101.yaml
```

## 📊 支持的数据集

| 数据集 | 类别数 | 训练样本 | 配置文件 |
|--------|--------|----------|----------|
| UCF101 | 101 | ~9,537 | `qwen2_5vl_ucf101.yaml` |
| Something-Something V2 | 174 | ~168k | `qwen2_5vl_sthsthv2.yaml` |

## 💡 数据格式

训练数据需要遵循以下格式：

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
  "videos": ["/path/to/video.mp4"],
  "action_label": 5
}
```

**关键要点**：
- 助手回复中必须包含 `<ACT>` token
- `action_label` 必须是整数（从 0 开始）
- `videos` 包含视频文件的路径

## 🔧 主要参数说明

### 训练阶段参数

```yaml
stage: action_cls  # 必须设置为 action_cls
```

### 动作分类参数

```yaml
num_action_classes: 101              # 动作类别总数
action_decoder_type: linear          # decoder 类型: linear 或 mlp
action_decoder_hidden_size: null     # mlp 模式下的隐藏层维度
action_decoder_path: null            # 预训练 decoder 路径（可选）
action_token_lr_scale: 0.1           # <ACT> token 学习率缩放因子
```

### LoRA 参数

```yaml
finetuning_type: lora
lora_rank: 16                        # LoRA 秩
lora_alpha: 32                       # LoRA alpha
lora_target: all                     # 应用 LoRA 的目标模块
use_dora: true                       # 使用 DoRA（推荐）
```

## 📝 示例脚本

### 数据准备脚本

在 `scripts/` 目录下创建数据转换脚本：

- `prepare_ucf101.py` - UCF101 数据集转换
- `prepare_sthsthv2.py` - Something-Something V2 数据集转换

详细代码参见[指导文档](video_action_classification_guide.md#数据集准备)。

### 训练脚本

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=video-action-classification

cd /path/to/IVCP/LlamaFactory

torchrun --nproc_per_node 4 --master_port 29500 \
    -m llamafactory.cli train \
    examples/train_lora_action_cls/qwen2_5vl_ucf101.yaml
```

## 🐛 故障排查

常见问题及解决方案：

1. **显存不足**：减小 `per_device_train_batch_size` 和 `video_max_pixels`
2. **<ACT> token 未找到**：检查数据格式，确保助手回复中包含 `<ACT>` token
3. **损失不下降**：调整学习率，检查标签范围
4. **数据加载慢**：增加 `preprocessing_num_workers` 和 `dataloader_num_workers`

更多详情参见[故障排查章节](video_action_classification_guide.md#故障排查)。

## 📚 参考资源

- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen2.5-VL 模型](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [UCF101 数据集](https://www.crcv.ucf.edu/data/UCF101.php)
- [Something-Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something)

## ⚙️ 系统要求

- **GPU**: NVIDIA GPU with CUDA 11.8+ (推荐 A100/V100, 至少 24GB 显存)
- **Python**: 3.9+
- **存储**: 至少 500GB

## 📄 许可证

本项目遵循 Apache License 2.0。

---

**需要帮助？** 请查看[完整指导文档](video_action_classification_guide.md)或提交 Issue。
