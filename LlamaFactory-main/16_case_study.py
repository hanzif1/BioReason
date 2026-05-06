import json
import os
import re

DATASET_NAME = "test_CameraTrap_desert-lion-balanced"

OURS3_PATH = f"saves/bioreason/predict_results_sft_only_b/{DATASET_NAME}/generated_predictions.jsonl"
OURS5_PATH = f"saves/bioreason/predict_results_grpo/{DATASET_NAME}/generated_predictions.jsonl"
# 👑 新增：原始数据集路径，用来拿图片地址
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
    if not os.path.exists(filepath): return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f: data.append(json.loads(line))
    return data

def main():
    print(f"正在交叉比对 {DATASET_NAME} 的结果...\n")
    
    ours3_data = load_predictions(OURS3_PATH)
    ours5_data = load_predictions(OURS5_PATH)
    
    # 👑 新增：加载原始数据
    with open(ORIGINAL_DATA_PATH, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    found_cases = 0

    for i in range(len(ours3_data)):
        gt_label = ours3_data[i].get("label", "").strip()
        if not gt_label: continue
        gt_label = gt_label[0].upper() 
        
        o3_raw = ours3_data[i].get("predict", "")
        o5_raw = ours5_data[i].get("predict", "")
        
        o3_ans = extract_answer(o3_raw)
        o5_ans = extract_answer(o5_raw)
        
        if o3_ans == gt_label and o5_ans != gt_label:
            found_cases += 1
            
            # 👑 新增：提取图片绝对路径
            img_path = original_data[i].get("images", ["无图片路径"])[0]
            
            print("="*80)
            print(f"🔥 发现绝佳 Case #{i+1} | 正确答案: 【{gt_label}】")
            print(f"🖼️ 图片本地地址: {img_path}")
            print("="*80)
            
            print(f"\n🤡 [Ours3 - 瞎猜队] 最终答案: {o3_ans} (正确)")
            print(f"🤔 Ours3 的思考过程:\n{extract_think(o3_raw)}")
            
            print(f"\n--------------------------------------------------")
            
            print(f"\n🛡️ [Ours5 - 诚实队] 最终答案: {o5_ans} (错误)")
            print(f"🤔 Ours5 的思考过程:\n{extract_think(o5_raw)}")
            print("\n")
            
            if found_cases >= 5: break

if __name__ == "__main__":
    main()