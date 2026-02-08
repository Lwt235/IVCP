import os
import json
from tqdm import tqdm

def merge_cot_data(input_dir):
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.json') and 'VideoEspresso' in filename:
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
                for idx, sample in enumerate(data):
                    if sample.get('cot') is not None:
                        data[idx]['answer'] = f"<think>{sample['cot']}</think><answer>{sample['answer']}</answer>"
                        del data[idx]['cot']

            with open(file_path, 'w', encoding='utf-8') as infile:
                json.dump(data, infile, ensure_ascii=False, indent=2)

def change_format(input_dir):
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.json') and 'VideoEspresso' in filename and 'copy' not in filename:
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
                for idx, sample in enumerate(data):
                    data[idx] = {
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"<video>{sample['question']}"
                            },
                            {
                                "from": "gpt",
                                "value": sample['answer']
                            }
                        ],
                        "videos": [
                            sample['video_path']
                        ]
                    }

            with open(file_path, 'w', encoding='utf-8') as infile:
                json.dump(data, infile, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_directory = "LlamaFactory/data"  # Replace with your actual directory
    # merge_cot_data(input_directory)
    change_format(input_directory)