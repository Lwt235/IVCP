from torch.utils.data import Dataset
from PIL import Image
import json
import os

class VideoQADataset(Dataset):
    def __init__(self, root_dir, data_path):
        with open(os.path.join(root_dir, data_path), 'r') as f:
            data = json.load(f)
        self.root_dir = root_dir
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        instruct = f"{item['question']}"
        response = f"<think>{item['cot']}</think><answer>{item['answer']}</answer>"

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"{os.path.join(self.root_dir, item['video_path'])}", "min_pixels": 4 * 28 * 28, "max_pixels": 4 * 28 * 28, "fps": 1},
                    {"type": "text", "text": instruct}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response}
                ]
            },
        ]

        return { "messages": conversation }

# def collate_fn(batch):
#     texts = [b["messages"] for b in batch]
#     videos = [b["video"] for b in batch]

#     model_inputs = tokenizer.apply_chat_template(
#         texts,
#         tokenize=True,
#         add_generation_prompt=False,
#         return_tensors="pt",
#         padding=True,
#     )

#     model_inputs["videos"] = videos
#     labels = model_inputs["input_ids"].clone()
#     model_inputs["labels"] = labels

#     return model_inputs
