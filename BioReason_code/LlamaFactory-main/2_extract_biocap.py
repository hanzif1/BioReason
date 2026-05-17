import pandas as pd
import json
import os

# 1. 读取 Parquet 文件
df = pd.read_parquet('uuid_caption_description.parquet')

# 查看列名，确认 ID 和 Caption 所在的列 (通常是 'uuid' 和 'caption')
print("数据列名:", df.columns.tolist())

sft_data_a = []
# 假设你的图片已经解压到了某个目录
IMAGE_BASE_DIR = "/data0/data/A_bioclip/TreeOfLife-10M/images/"

for _, row in df.head(10000).iterrows(): # 先取 1 万条做测试
    # 根据 README 确认如何从 UUID 转换成图片路径，通常是 uuid + ".jpg"
    img_path = os.path.join(IMAGE_BASE_DIR, f"{row['uuid']}.jpg")
    
    if os.path.exists(img_path):
        sft_data_a.append({
            "instruction": "Describe the visual features of this organism.", # 固定 Prompt
            "input": "",
            "output": row['caption'], # 提取出的细粒度描述
            "images": [os.path.abspath(img_path)]
        })

# 保存为 Source A 训练集
with open('data/bioreason_source_a.json', 'w', encoding='utf-8') as f:
    json.dump(sft_data_a, f, ensure_ascii=False, indent=2)

print(f"成功生成 Source A 数据集，共 {len(sft_data_a)} 条。")