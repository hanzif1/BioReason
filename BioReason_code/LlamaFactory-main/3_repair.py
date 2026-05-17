import json

def patch_bioreason_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        # 如果开头没有 <image>，则添加
        if not item['instruction'].startswith("<image>"):
            item['instruction'] = "<image>\n" + item['instruction']
            
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已修复: {file_path}")

# 执行修复
patch_bioreason_data('data/A_biocap_captioning.json')
patch_bioreason_data('data/B_bioreason_reasoning.json')