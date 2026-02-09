# 视频动作分类训练指导

本仓库包含了基于 Qwen2.5-VL-3B 模型进行视频动作分类训练的完整实现和文档。

## 📚 文档

详细的训练指导文档位于 `docs/` 目录：

- **[视频动作分类联合训练指导文档](docs/video_action_classification_guide.md)** - 完整的训练指南（中文，734行）
- **[文档索引和快速开始](docs/README.md)** - 快速入门指南

## 🚀 快速开始

### 1. 查看文档

```bash
# 查看完整指导文档
cat docs/video_action_classification_guide.md

# 或在浏览器中打开
# https://github.com/Lwt235/IVCP/blob/main/docs/video_action_classification_guide.md
```

### 2. 使用示例配置

配置文件位于 `LlamaFactory/examples/train_lora_action_cls/`：

- `qwen2_5vl_ucf101.yaml` - UCF101 数据集训练配置
- `qwen2_5vl_sthsthv2.yaml` - Something-Something V2 数据集训练配置

### 3. 开始训练

```bash
cd LlamaFactory

# 使用 UCF101 配置训练
llamafactory-cli train examples/train_lora_action_cls/qwen2_5vl_ucf101.yaml
```

## 🎯 核心特性

本实现基于 `copilot/migrate-video-action-classification` 分支的修改，主要特性包括：

1. **ActionDecoder 分类头** - 轻量级分类器，支持 linear 和 mlp 两种架构
2. **特殊 Token `<ACT>`** - 标记动作分类位置
3. **联合训练** - 同时优化 Qwen2.5-VL-3B（LoRA）和 ActionDecoder
4. **action_cls 训练阶段** - 新增的专门训练模式

## 📊 支持的数据集

- **UCF101** - 101 个动作类别，约 13,320 个视频
- **Something-Something V2** - 174 个动作类别，约 220k 个视频

## 🔗 相关链接

- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [UCF101 数据集](https://www.crcv.ucf.edu/data/UCF101.php)
- [Something-Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something)

## 📝 许可证

本项目遵循 Apache License 2.0。

---

**需要帮助？** 请查看[完整文档](docs/video_action_classification_guide.md)或提交 Issue。
