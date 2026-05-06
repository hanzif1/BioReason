import os
import json
import glob
import re
# BASE_DIR = "saves/bioreason/predict_results_dpo"

# 结果所在的根目录
BASE_DIR = "saves/bioreason/predict_results_grpo"
BASE_DIR = "saves/bioreason/predict_results_ckpt6741"
BASE_DIR = "saves/bioreason/predict_results_baseline"
BASE_DIR = "saves/bioreason/predict_results_sft_only_b"

def extract_answer(predict_text):
    """
    双重正则匹配，提取模型预测的答案字母
    """
    # 规则 1: 完美遵循格式 <answer>A</answer>
    match_tag = re.search(r'<answer>\s*([A-E])\s*</answer>', predict_text, re.IGNORECASE)
    if match_tag:
        return match_tag.group(1).upper()
        
    # 规则 2: 格式漂移，偷懒写在了文本里 Final answer: A 或 Final answer: Option A
    match_text = re.search(r'\s*(?:Option\s*)?([A-E])', predict_text, re.IGNORECASE)
    if match_text:
        return match_text.group(1).upper()
        
    return None

def main():
    print(f"{'='*80}")
    print(f"{'BioReason Stage 1 - Zero-Shot Classification Evaluation':^80}")
    print(f"{'='*80}")
    print(f"{'Dataset Name':<45} | {'Total':<6} | {'Correct':<8} | {'Accuracy':<10} | {'Format Miss'}")
    print(f"{'-'*45}-|-{'-'*6}-|-{'-'*8}-|-{'-'*10}-|-{'-'*12}")

    # 查找所有的预测结果文件
    search_pattern = os.path.join(BASE_DIR, "test_*", "generated_predictions.jsonl")
    jsonl_files = glob.glob(search_pattern)
    
    if not jsonl_files:
        print("未找到任何 generated_predictions.jsonl 文件，请检查路径。")
        return

    # 按字母顺序排列结果
    jsonl_files.sort()

    for file_path in jsonl_files:
        dataset_name = os.path.basename(os.path.dirname(file_path))
        dataset_name = dataset_name.replace("test_", "") # 去掉 test_ 前缀让表格更好看
        
        total = 0
        correct = 0
        format_miss = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                data = json.loads(line)
                label = data.get("label", "").strip().upper()
                predict_text = data.get("predict", "")
                
                # 提取预测字母
                pred_letter = extract_answer(predict_text)
                
                total += 1
                if pred_letter == label:
                    correct += 1
                
                if pred_letter is None:
                    format_miss += 1
                    
        # 计算准确率
        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        # 打印单行成绩
        print(f"{dataset_name:<45} | {total:<6} | {correct:<8} | {accuracy:>5.2f}%    | {format_miss}")

    print(f"{'='*80}")
    print("评估完成！Format Miss 表示模型既没有输出 <answer> 标签，也没说 Final answer 的残缺样本数。")

if __name__ == "__main__":
    main()