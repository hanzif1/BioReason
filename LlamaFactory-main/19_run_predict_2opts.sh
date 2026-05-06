#!/bin/bash
cat << 'PYEOF' > temp_ablation.py
import json, re, random, sys, os
dataset_name = sys.argv[1]
num_options = int(sys.argv[2])
info_path = 'data/dataset_info.json'
with open(info_path, 'r', encoding='utf-8') as f: info = json.load(f)
if dataset_name not in info: sys.exit(0)
file_path = os.path.join('data', info[dataset_name]['file_name'])
with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)

new_data = []
for item in data:
    prompt_key = 'instruction' if 'instruction' in item else 'input'
    prompt, output = item.get(prompt_key, ""), item.get('output', "")
    ans_matches = re.findall(r'\b[A-Z]\b', output)
    if not ans_matches:
        new_data.append(item); continue
    correct_letter = ans_matches[-1]
    
    lines = prompt.split('\n')
    opts_found, opt_start_idx, opt_end_idx = {}, -1, -1
    for i, line in enumerate(lines):
        m = re.match(r'^([A-Z])\.\s*(.+)$', line.strip())
        if m:
            if opt_start_idx == -1: opt_start_idx = i
            opt_end_idx = i
            opts_found[m.group(1)] = m.group(2)
            
    if correct_letter not in opts_found or len(opts_found) <= num_options:
        new_data.append(item); continue
        
    correct_text = opts_found[correct_letter]
    
    # 【核心修改】完全随机抽取 num_options 个选项，不保证包含 correct_text
    all_texts = list(opts_found.values())
    selected_all = random.sample(all_texts, num_options)
    random.shuffle(selected_all)
    
    new_opts_block = []
    new_correct_letter = 'Z' # 如果正确答案被随机去掉了，答案设为 Z（注定错）
    for i, text in enumerate(selected_all):
        letter = chr(ord('A') + i)
        if text == correct_text: new_correct_letter = letter
        new_opts_block.append(f"{letter}. {text}")
        
    new_item = item.copy()
    new_item[prompt_key] = '\n'.join(lines[:opt_start_idx] + new_opts_block + lines[opt_end_idx+1:])
    new_item['output'] = new_correct_letter.join(output.rsplit(correct_letter, 1))
    new_data.append(new_item)

new_file_name = info[dataset_name]['file_name'].replace('.json', f'_{num_options}opts.json')
with open(os.path.join('data', new_file_name), 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

info[f"{dataset_name}_{num_options}opts"] = info[dataset_name].copy()
info[f"{dataset_name}_{num_options}opts"]['file_name'] = new_file_name
with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
PYEOF

DATASETS=("test_CameraTrap_desert-lion-balanced" "test_CameraTrap_island-balanced" "test_CameraTrap_orinoquia-balanced" "test_CameraTrap_ENA24-balanced" "test_CameraTrap_ohio-small-animals-balanced")

for DATASET in "${DATASETS[@]}"; do
  echo "====================================================="
  echo "处理数据并预测 2 选项数据集: $DATASET"
  echo "====================================================="
  python temp_ablation.py $DATASET 2
  ABLATION_DATASET="${DATASET}_2opts"
  
  cat <<YAML_EOF > examples/bioreason/temp_predict_grpo_2opts.yaml
model_name_or_path: models/Qwen2.5-VL-7B-BioReason-SFT
adapter_name_or_path: saves/bioreason/grpo_trl/checkpoint-final
stage: sft
do_predict: true
finetuning_type: lora
template: qwen2_vl
eval_dataset: $ABLATION_DATASET
output_dir: saves/bioreason/predict_results_grpo_2opts_rand/$DATASET
per_device_eval_batch_size: 32
predict_with_generate: true
max_new_tokens: 512
fp16: true
YAML_EOF
  
  FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=5 llamafactory-cli train examples/bioreason/temp_predict_grpo_2opts.yaml
done
