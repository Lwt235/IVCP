# 运行以下 python 脚本，需要修改 <user_name> 和 <data_path>
from datasets import load_dataset

dataset_path = './data/gsm8k/'
dataset = load_dataset('parquet', data_files={
    'train': f'{dataset_path}train.parquet',
    'test': f'{dataset_path}test.parquet'
})

print(dataset['train'][0])  # 查看第一条数据
print(dataset['train'].features)  # 查看特征结构