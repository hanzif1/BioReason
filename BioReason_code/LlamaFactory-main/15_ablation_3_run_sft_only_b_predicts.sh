#!/bin/bash

DATASETS=(
  "test_CameraTrap_desert-lion-balanced"
  "test_CameraTrap_island-balanced"
  "test_CameraTrap_orinoquia-balanced"
  "test_CameraTrap_ENA24-balanced"
  "test_CameraTrap_ohio-small-animals-balanced"
  "test_FNG_Mini_labels"
  "test_INS_Mini_labels"
  # "test_nabird_metadata"
  "test_PLT_NET_Mini_labels"
  "test_INS_2_Mini_labels"
  "test_MED_LF_Mini_labels"
  "test_PLK_Mini_labels"
  "test_PLT_VIL_Mini_labels"
)

for DATASET in "${DATASETS[@]}"
do
  echo "====================================================="
  echo "开始评估消融实验模型: $DATASET"
  echo "====================================================="
  
  cat <<EOF > examples/bioreason/temp_predict_sft_only_b.yaml
### 基础设置
# 必须使用未微调的官方底座
model_name_or_path: /data1/pretrained/bioclip/models/Qwen2.5-VL-7B
# 挂载我们刚刚只用 Source B 训出来的权重
adapter_name_or_path: saves/bioreason/sft_only_b/
stage: sft
do_predict: true
finetuning_type: lora
template: qwen2_vl

### 动态替换的参数
eval_dataset: $DATASET
output_dir: saves/bioreason/predict_results_sft_only_b/$DATASET

### 预测参数，直接上 32 拉满你的显卡
per_device_eval_batch_size: 32
predict_with_generate: true
max_new_tokens: 512
fp16: true
EOF
  
  FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=5 llamafactory-cli train examples/bioreason/temp_predict_sft_only_b.yaml
    
done

echo "仅使用Source B的消融实验预测已全部完成！"