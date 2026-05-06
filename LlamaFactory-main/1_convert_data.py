import json
import re
import string

# 加载原始数据
with open('A_out_json/Analysis_Results_5000/Global_Analysis_5000_Full.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

def validate_item(item):
    """
    实现 Step 3: Quality Control 过滤规则 
    """
    response = item.get('formatted_response', '')
    
    # 提取 <think> 内容和 <answer> 标签内容
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    
    think_content = think_match.group(1).strip() if think_match else ""
    answer_content = answer_match.group(1).strip() if answer_match else ""

    # 规则 1: 答案正确 (Doubao 最终答案必须 = ground truth) 
    # 这里比对的是 <answer> 里的字母是否与 correct_option_letter 一致
    if answer_content != item.get('correct_option_letter'):
        return False, "Wrong Answer"

    # 规则 2: Think 非空 (必须包含 <think> 推理过程) 
    if not think_content:
        return False, "Empty Think"

    # 规则 3: 有比对 (推理中必须提及 >= 2 个候选) 
    # 通过正则匹配 "Option A", "Option B" 等关键字的出现次数
    mentioned_options = set(re.findall(r'Option [A-Z]', think_content))
    if len(mentioned_options) < 2:
        return False, "Lack of Comparison"

    # 规则 4: 长度合理 (think 部分 100-400 tokens) 
    # 简单估算：中文字符数 + 英文单词数
    token_count = len(re.findall(r'\w+', think_content)) + len(re.findall(r'[\u4e00-\u9fff]', think_content))
    if not (100 <= token_count <= 400):
        return False, f"Invalid Length ({token_count})"

    return True, "Valid"

formatted_data = []
stats = {"Total": 0, "Valid": 0, "Filtered": 0}

for item in raw_data:
    stats["Total"] += 1
    
    # 执行 QC 过滤
    is_ok, reason = validate_item(item)
    if not is_ok:
        stats["Filtered"] += 1
        continue
    
    # 构造带有字母标签的选项列表 (优化项)
    labeled_options = [f"{string.ascii_uppercase[i]}. {opt}" for i, opt in enumerate(item['options'])]
    options_str = "\n".join(labeled_options)
    
    # 构造指令 [cite: 209-211]
    instruction = (
        "Identify the species in the image from the following candidates. "
        "Provide your analysis inside <think> tags and the final answer after it.\n\n"
        f"Candidates:\n{options_str}"
    )
    
    formatted_data.append({
        "instruction": instruction,
        "input": "",
        "output": item['formatted_response'],
        "images": [item['image_path']]
    })
    stats["Valid"] += 1

# 保存数据
with open('data/B_bioreason_reasoning.json', 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=2)

# 输出过滤统计
print(f"--- 数据转换完成 ---")
print(f"总数据量: {stats['Total']}")
print(f"有效数据: {stats['Valid']}")
print(f"过滤数据: {stats['Filtered']} (过滤率: {stats['Filtered']/stats['Total']:.1%})")