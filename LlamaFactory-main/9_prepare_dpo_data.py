import json
import re
import random

# 你的原始 SFT 数据集路径（包含完整 <think> 和 <answer> 的那个）
input_file = "data/B_bioreason_reasoning.json" 
output_file = "data/bioreason_dpo_train.json"

with open(input_file, 'r', encoding='utf-8') as f:
    sft_data = json.load(f)

dpo_data = []
for item in sft_data:
    original_output = item.get("output", "")
    
    # 用正则提取出 SFT 数据里的推理过程和最终答案
    match = re.search(r'<think>(.*?)</think>\s*<answer>\s*([A-E])\s*</answer>', original_output, re.DOTALL | re.IGNORECASE)
    
    if match:
        reasoning = match.group(1).strip()
        answer = match.group(2).upper()
        
        # 【Chosen: 完美格式】模型应该学习的榜样
        chosen = f"<think>\n{reasoning}\n</think>\n<answer>{answer}</answer>"
        
        # 【Rejected: 劣质格式】模拟模型偷懒的几种情况，让它学会厌恶这些格式
        flaw_type = random.choice([1, 2, 3])
        if flaw_type == 1:
            # 错误 1：把答案写在了 think 里面（正是你之前遇到的情况）
            rejected = f"<think>\n{reasoning}\nFinal answer: {answer}\n</think>"
        elif flaw_type == 2:
            # 错误 2：完全没有 think 标签，直接白话输出
            rejected = f"Based on the visual features, {reasoning}\nTherefore, the answer is {answer}."
        else:
            # 错误 3：极度敷衍，连推理都没有
            rejected = f" {answer}"
            
        dpo_data.append({
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "chosen": chosen,
            "rejected": rejected,
            "images": item.get("images", [])
        })

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dpo_data, f, ensure_ascii=False, indent=2)

print(f"DPO 数据集准备完成！共生成 {len(dpo_data)} 条拉踩数据。")