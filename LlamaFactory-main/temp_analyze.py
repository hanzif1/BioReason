import json
import re

file_path = "saves/bioreason/predict_results_grpo_2opts/test_CameraTrap_ohio-small-animals-balanced/generated_predictions.jsonl"

total = 0
correct = 0

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        label = data.get("label", "").strip()
        predict = data.get("predict", "")
        
        # 提取 </think> 之后的最终答案内容
        if "</think>" in predict:
            final_answer_part = predict.split("</think>")[-1]
        else:
            final_answer_part = predict
            
        # 使用正则提取第一个大写字母作为预测选项
        match = re.search(r'[A-Z]', final_answer_part)
        predicted_option = match.group(0) if match else ""
        
        if predicted_option == label:
            correct += 1
        total += 1

accuracy = (correct / total) * 100 if total > 0 else 0

print("="*50)
print(f"数据集: ohio-small-animals-balanced (2选项消融)")
print(f"总样本数: {total}")
print(f"正确预测: {correct}")
print(f"真实准确率 (Accuracy): {accuracy:.2f}%")
print("="*50)
