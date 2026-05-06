#!/bin/bash

# 你的数据集列表
DATASETS=(
  "test_CameraTrap_desert-lion-balanced"
  "test_CameraTrap_island-balanced"
  "test_CameraTrap_orinoquia-balanced"
  "test_CameraTrap_ENA24-balanced"
  "test_CameraTrap_ohio-small-animals-balanced"
  "test_FNG_Mini_labels"
  "test_INS_Mini_labels"
  "test_nabird_metadata"
  "test_PLT_NET_Mini_labels"
  "test_INS_2_Mini_labels"
  "test_MED_LF_Mini_labels"
  "test_PLK_Mini_labels"
  "test_PLT_VIL_Mini_labels"
)

# dataset_info.json 的路径（请根据你的实际路径调整）
DATASET_INFO="data/dataset_info.json"

echo "-----------------------------------------------------"
printf "%-50s | %-10s\n" "数据集名称" "样本数量"
echo "-----------------------------------------------------"

TOTAL_COUNT=0

for DATASET in "${DATASETS[@]}"
do
  # 从 dataset_info.json 中提取文件名
  FILE_NAME=$(python3 -c "import json; print(json.load(open('$DATASET_INFO'))['$DATASET'].get('file_name', ''))" 2>/dev/null)
  
  if [ -z "$FILE_NAME" ]; then
    printf "%-50s | %-10s\n" "$DATASET" "未找到配置"
    continue
  fi

  # 检查文件是否存在并计算行数（假设是标准的 JSON 数组格式）
  # 这里的逻辑是计算 JSON 文件中 '{' 出现的次数，或者直接用 Python 读取长度
  FILE_PATH="data/$FILE_NAME"
  if [ -f "$FILE_PATH" ]; then
    COUNT=$(python3 -c "import json; print(len(json.load(open('$FILE_PATH'))))" 2>/dev/null)
    printf "%-50s | %-10s\n" "$DATASET" "$COUNT"
    TOTAL_COUNT=$((TOTAL_COUNT + COUNT))
  else
    printf "%-50s | %-10s\n" "$DATASET" "文件不存在"
  fi
done

echo "-----------------------------------------------------"
echo "总计待处理数据量: $TOTAL_COUNT"