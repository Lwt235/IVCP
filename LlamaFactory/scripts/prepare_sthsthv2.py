import os
import json
from pathlib import Path

def prepare_sthsthv2_action_cls(
    video_root="data/something-something-v2/videos",
    annotation_file="data/something-something-v2/train.json",
    label_file="data/something-something-v2/labels.json",
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

        action_label = label_to_idx[template_filled]

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
    with open(output_json, 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(samples)} training samples")
    print(f"Number of classes: {len(label_to_idx)}")
    print(f"Saved to {output_json}")

if __name__ == "__main__":
    # 生成训练集
    prepare_sthsthv2_action_cls(
        annotation_file="data/something-something-v2/train.json",
        output_json="LlamaFactory/data/sthsthv2_train.json"
    )

    # 生成验证集
    prepare_sthsthv2_action_cls(
        annotation_file="data/something-something-v2/validation.json",
        output_json="LlamaFactory/data/sthsthv2_val.json"
    )