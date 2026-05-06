import os
import json
import re

# 定义我们要检查的选项组和数据集列表
opts_list = [2, 3, 4]
datasets = [
    "test_CameraTrap_desert-lion-balanced",
  "test_CameraTrap_island-balanced",
  "test_CameraTrap_orinoquia-balanced",
  "test_CameraTrap_ENA24-balanced",
  "test_CameraTrap_ohio-small-animals-balanced"
#  "test_FNG_Mini_labels",
#  "test_INS_Mini_labels",
#  "test_nabird_metadata",
#  "test_PLT_NET_Mini_labels",
#  "test_INS_2_Mini_labels",
#  "test_MED_LF_Mini_labels",
#  "test_PLK_Mini_labels",
#  "test_PLT_VIL_Mini_labels"
]

# 结果保存的根目录
base_save_dir = "saves/bioreason"

def extract_answer(text):
    """
    从文本中提取出最后一个大写字母作为答案。
    这可以完美兼容模型输出格式： <think>...</think> B
    """
    matches = re.findall(r'\b[A-Z]\b', text)
    if matches:
        return matches[-1] # 取最后一个孤立的大写字母
    return ""

print("="*60)
print(" 选项数量消融实验准确率 (Accuracy) 评估汇总 ")
print("="*60)

for opt in opts_list:
    print(f"\n【 {opt} 个候选选项模型 】")
    total_correct = 0
    total_samples = 0
    
    for dataset in datasets:
        # 拼接生成的 jsonl 文件路径
        file_path = os.path.join(base_save_dir, f"predict_results_grpo_{opt}opts_rand", dataset, "generated_predictions.jsonl")
        
        if not os.path.exists(file_path):
            print(f"  [等待生成] {dataset}")
            continue
            
        correct = 0
        count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                label_raw = str(data.get("label", "")).strip()
                predict_raw = str(data.get("predict", "")).strip()
                
                # 提取纯字母进行对比
                predict_ans = extract_answer(predict_raw)
                label_ans = extract_answer(label_raw) 
                
                if predict_ans == label_ans and label_ans != "":
                    correct += 1
                    total_correct += 1
                count += 1
                total_samples += 1
                
        acc = (correct / count) * 100 if count > 0 else 0
        print(f"  - {dataset:45}: {acc:.2f}% ({correct}/{count})")
        
    if total_samples > 0:
        avg_acc = (total_correct / total_samples) * 100
        print(f"  >>> 该组综合平均准确率: {avg_acc:.2f}%")
    else:
        print("  >>> 暂无有效数据")
