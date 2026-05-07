import json
import re
import os

# 路径配置
sft_data_path = 'data/B_bioreason_reasoning.json'
grpo_data_path = 'data/bioreason_grpo_train.json'
dataset_info_path = 'data/dataset_info.json'

with open(sft_data_path, 'r', encoding='utf-8') as f:
    sft_data = json.load(f)

grpo_data = []
for item in sft_data:
    # 提取正确答案的字母
    match = re.search(r'<answer>\s*([A-E])\s*</answer>', item['output'], re.IGNORECASE)
    if match:
        correct_letter = match.group(1).upper()
        grpo_data.append({
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": correct_letter, # 关键：输出只保留正确字母
            "images": item.get("images", [])
        })

with open(grpo_data_path, 'w', encoding='utf-8') as f:
    json.dump(grpo_data, f, ensure_ascii=False, indent=2)

# 自动注册到 dataset_info.json
with open(dataset_info_path, 'r', encoding='utf-8') as f:
    dataset_info = json.load(f)

dataset_info["bioreason_grpo_train"] = {
    "file_name": "bioreason_grpo_train.json",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "images": "images"
    }
}

with open(dataset_info_path, 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, ensure_ascii=False, indent=2)

print(f"GRPO 数据集准备完成并注册成功！总样本数: {len(grpo_data)}")