import os
import json
from pathlib import Path

def prepare_ucf101_action_cls(
    video_root="data/UCF101",
    split_file="data/UCF101/trainlist01.txt",
    output_json="LlamaFactory/data/ucf101_train.json",
):
    """
    将 UCF101 数据集转换为 action_cls 训练格式
    """
    # UCF101 类别列表（101个类别）
    with open("data/UCF101/classInd.txt") as f:
        class_to_idx = {}
        for line in f:
            idx, class_name = line.strip().split()
            class_to_idx[class_name] = int(idx) - 1  # 转为0-based索引

    # 生成训练json
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
    with open(output_json, 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(samples)} training samples")
    print(f"Saved to {output_json}")

if __name__ == "__main__":
    # 生成训练集
    prepare_ucf101_action_cls(
        split_file="data/UCF101/trainlist01.txt",
        output_json="LlamaFactory/data/ucf101_train.json"
    )

    # 生成测试集
    prepare_ucf101_action_cls(
        split_file="data/UCF101/testlist01.txt",
        output_json="LlamaFactory/data/ucf101_test.json"
    )