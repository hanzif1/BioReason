#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# 提取刚刚注册的所有 test_ 开头的数据集名称
DATASETS=(
#   "test_CameraTrap_desert-lion-balanced"
#   "test_CameraTrap_island-balanced"
#   "test_CameraTrap_orinoquia-balanced"
#   "test_CameraTrap_ENA24-balanced"
#   "test_CameraTrap_ohio-small-animals-balanced"
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
  echo "开始预测数据集: $DATASET"
  echo "====================================================="
  
  # 每次循环动态生成一个专属的 yaml 文件
  cat <<EOF > examples/bioreason/temp_predict.yaml
### 基础设置
model_name_or_path: /data1/pretrained/bioclip/models/Qwen2.5-VL-7B
adapter_name_or_path: saves/bioreason/sft/checkpoint-6741
stage: sft
do_predict: true
finetuning_type: lora
template: qwen2_vl

### 动态替换的参数
eval_dataset: $DATASET
output_dir: saves/bioreason/predict_results_ckpt6741/$DATASET

### 预测参数
per_device_eval_batch_size: 32
predict_with_generate: true
max_new_tokens: 512
fp16: true
EOF
  
  # 直接读取动态生成的 yaml 文件，不加任何额外命令行参数
  FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=3 llamafactory-cli train examples/bioreason/temp_predict.yaml
    
done

echo "所有数据集预测完成！"