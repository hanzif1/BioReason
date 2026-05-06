import json
import os
import string
import glob
import random

# 路径配置
input_dirs = ["A_out_json/data1", "A_out_json/data2"]
output_dir = "data"
dataset_info_path = os.path.join(output_dir, "dataset_info.json")

# 读取现有的 dataset_info.json
with open(dataset_info_path, 'r', encoding='utf-8') as f:
    dataset_info = json.load(f)

for d in input_dirs:
    for filepath in glob.glob(os.path.join(d, "*.json")):
        filename = os.path.basename(filepath)
        dataset_name = f"test_{filename.replace('.json', '')}"
        out_filepath = os.path.join(output_dir, f"{dataset_name}.json")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        formatted_data = []
        for item in raw_data:
            options = item.get("top_5_predictions", [])
            gt = item.get("ground_truth", "")
            
            # 如果真实答案不在前五名里，强行替换掉最后一个，保证有解
            if gt not in options:
                options[-1] = gt
            
            random.shuffle(options)
            
            labeled_options = []
            correct_letter = ""
            for i, opt in enumerate(options):
                letter = string.ascii_uppercase[i]
                labeled_options.append(f"{letter}. {opt}")
                if opt == gt:
                    correct_letter = letter
                    
            options_str = "\n".join(labeled_options)
            instruction = (
                "<image>\nIdentify the species in the image from the following candidates. "
                "Provide your analysis inside <think> tags and the final answer after it.\n\n"
                f"Candidates:\n{options_str}"
            )
            
            formatted_data.append({
                "instruction": instruction,
                "input": "",
                "output": correct_letter, # 将正确答案存在 output 中，预测时不影响输入，但方便后续算准确率
                "images": [item["image_path"]],
                "ground_truth": gt
            })
            
        # 保存格式化后的数据
        with open(out_filepath, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        # 自动注册到 dataset_info.json
        dataset_info[dataset_name] = {
            "file_name": f"{dataset_name}.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "images": "images"
            }
        }
        print(f"已处理并注册数据集: {dataset_name} (样本数: {len(formatted_data)})")

# 写回 dataset_info.json
with open(dataset_info_path, 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, ensure_ascii=False, indent=2)

print("\n所有测试集已准备就绪！")