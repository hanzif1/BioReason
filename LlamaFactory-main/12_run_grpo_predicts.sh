#!/bin/bash

DATASETS=(
#   "test_CameraTrap_desert-lion-balanced"
#   "test_CameraTrap_island-balanced"
#   "test_CameraTrap_orinoquia-balanced"
  # "test_CameraTrap_ENA24-balanced"
  # "test_CameraTrap_ohio-small-animals-balanced"
  # "test_FNG_Mini_labels"
  # "test_INS_Mini_labels"
  # "test_PLT_NET_Mini_labels"
  # "test_INS_2_Mini_labels"
  # "test_MED_LF_Mini_labels"
  # "test_PLK_Mini_labels"
  # "test_PLT_VIL_Mini_labels"
  "test_nabird_metadata"
)

for DATASET in "${DATASETS[@]}"
do
  echo "====================================================="
  echo "开始用 GRPO 模型预测数据集: $DATASET"
  echo "====================================================="
  
  cat <<EOF > examples/bioreason/temp_predict_grpo.yaml
### 基础设置
# 底座依然是你合并好的 SFT 模型
model_name_or_path: models/Qwen2.5-VL-7B-BioReason-SFT
# 挂载我们刚刚用 TRL 跑出来的 GRPO 终极权重！
adapter_name_or_path: saves/bioreason/grpo_trl/checkpoint-final
stage: sft
do_predict: true
finetuning_type: lora
template: qwen2_vl

### 动态替换的参数 (存入专属的 grpo 结果文件夹)
eval_dataset: $DATASET
output_dir: saves/bioreason/predict_results_grpo/$DATASET

### 预测参数
per_device_eval_batch_size: 32
predict_with_generate: true
max_new_tokens: 512
fp16: true
EOF
  
  # 测试推理不需要算梯度，所以 4 张卡可以轻松一起跑，速度飞快
  FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=5 llamafactory-cli train examples/bioreason/temp_predict_grpo.yaml
    
done

echo "所有数据集的 GRPO 预测已全部完成！"