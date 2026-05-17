#!/bin/bash

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

for DATASET in "${DATASETS[@]}"
do
  echo "====================================================="
  echo "开始用 DPO 终极模型预测数据集: $DATASET"
  echo "====================================================="
  
  cat <<EOF > examples/bioreason/temp_predict_dpo.yaml
### 基础设置
# 这里用的是合并后的 SFT 字典作为底座！
model_name_or_path: models/Qwen2.5-VL-7B-BioReason-SFT
# 这里挂载你刚刚跑出来的 DPO 增量权重！(⚠️ 请把下面这行的 checkpoint-XXX 替换为你刚才查到的真实文件夹名)
adapter_name_or_path: saves/bioreason/dpo/checkpoint-final
stage: sft
do_predict: true
finetuning_type: lora
template: qwen2_vl

### 动态替换的参数 (存入新的 dpo 结果文件夹)
eval_dataset: $DATASET
output_dir: saves/bioreason/predict_results_dpo/$DATASET

### 预测参数
per_device_eval_batch_size: 1
predict_with_generate: true
max_new_tokens: 512
fp16: true
EOF
  
  # 使用 4 张卡并行推理
  FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=5 llamafactory-cli train examples/bioreason/temp_predict_dpo.yaml
    
done

echo "所有数据集的 DPO 预测已全部完成！"