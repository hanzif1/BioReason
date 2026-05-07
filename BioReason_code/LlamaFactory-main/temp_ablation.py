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
