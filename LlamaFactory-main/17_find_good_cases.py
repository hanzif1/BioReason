import json
import os
import re

# 🌟 强烈推荐去 nabird (北美鸟类特写) 找，图片非常高清，写论文放上去贼漂亮！
# 当然，你也可以换成 test_CameraTrap_island-balanced 或者其他你想展示的数据集
DATASET_NAME = "test_nabird_metadata" 

OURS5_PATH = f"saves/bioreason/predict_results_grpo/{DATASET_NAME}/generated_predictions.jsonl"
ORIGINAL_DATA_PATH = f"data/{DATASET_NAME}.json"

def extract_answer(text):
    match = re.search(r"<answer>\s*([A-E])\s*</answer>", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    fallback = re.search(r'\b([A-E])\b', text)
    return fallback.group(1).upper() if fallback else "UNKNOWN"

def extract_think(text):
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else "无思维链"

def load_predictions(filepath):
    data = []
    if not os.path.exists(filepath): 
        print(f"❌ 找不到预测文件: {filepath}")
        return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f: data.append(json.loads(line))
    return data

def main():
    print(f"🌟 正在 {DATASET_NAME} 中为您提取 Ours5 的高光正确案例...\n")
    
    ours5_data = load_predictions(OURS5_PATH)
    
    if not os.path.exists(ORIGINAL_DATA_PATH):
        print(f"❌ 找不到原始数据文件: {ORIGINAL_DATA_PATH}")
        return
        
    with open(ORIGINAL_DATA_PATH, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    found_cases = 0

    for i in range(len(ours5_data)):
        gt_label = ours5_data[i].get("label", "").strip()
        if not gt_label: continue
        gt_label = gt_label[0].upper() 
        
        o5_raw = ours5_data[i].get("predict", "")
        o5_ans = extract_answer(o5_raw)
        
        # 👑 纯粹的展示逻辑：只要 Ours5 答对了，就抓出来！
        if o5_ans == gt_label:
            found_cases += 1
            
            img_path = original_data[i].get("images", ["无图片路径"])[0]
            
            print("="*80)
            print(f"✨ 发现神级推理 Case #{i+1} | 正确答案: 【{gt_label}】")
            print(f"🖼️ 图片本地地址: {img_path} ")
            print("="*80)
            
            print(f"\n✅ [Ours5 - 全家桶] 最终答案: {o5_ans}")
            print(f"🤔 Ours5 的细粒度视觉推理过程:\n{extract_think(o5_raw)}")
            print("\n")
            
            # 找到 5 个就先停下来，方便你挑图
            if found_cases >= 5: break

if __name__ == "__main__":
    main()