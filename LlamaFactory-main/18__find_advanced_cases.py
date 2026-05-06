import json
import os

# 目标数据集
DATASET_PATH = "data/test_nabird_metadata.json"

# 我们重点狙击这四种极其容易混淆的鸟
target_species = [
    "Least Flycatcher",      # 最小蝇霸鹟
    "Acadian Flycatcher",    # 阿卡迪亚蝇霸鹟 
    "Song Sparrow",          # 歌雀
    "Savannah Sparrow"       # 稀树草鹀 
]

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ 找不到文件: {DATASET_PATH}，请检查路径。")
        return

    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("🎯 正在启用暴力破解模式，搜索高难度图片...\n")

    # 存储结果
    results = {species: [] for species in target_species}

    for item in data:
        # 👑 暴力破解：把整个字典变成字符串，忽略大小写直接搜！
        item_str = str(item).lower()
        
        # 兼容不同的图片键名，通常是 'images' 或 'image'
        img_path = item.get("images", item.get("image", ["无路径"]))
        if isinstance(img_path, list):
            img_path = img_path[0]

        # 匹配目标物种
        for species in target_species:
            if species.lower() in item_str:
                results[species].append(img_path)

    # 打印结果
    for species, paths in results.items():
        print(f"{'='*80}")
        print(f"🦅 【{species}】候选图片 (请下载查看):")
        print(f"{'='*80}")
        
        # 去重（因为有的题目选项里可能出现好几次）
        unique_paths = list(set(paths))
        
        if not unique_paths:
            print("未找到该物种的图片（可能你的测试集没抽到这种鸟）。")
        else:
            for i, path in enumerate(unique_paths[:5]):  # 给你吐出前5张
                print(f"图片 {i+1}: {path}")
        print("\n")

if __name__ == "__main__":
    main()