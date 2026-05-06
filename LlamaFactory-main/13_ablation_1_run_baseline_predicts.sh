#!/bin/bash

# 你的 13 个测试集
DATASETS=(
  "test_CameraTrap_desert-lion-balanced"
  "test_CameraTrap_island-balanced"
  "test_CameraTrap_orinoquia-balanced"
  "test_CameraTrap_ENA24-balanced"
  "test_CameraTrap_ohio-small-animals-balanced"
#   "test_FNG_Mini_labels"
#   "test_INS_Mini_labels"
#   "test_nabird_metadata"
#   "test_PLT_NET_Mini_labels"
#   "test_INS_2_Mini_labels"
#   "test_MED_LF_Mini_labels"
#   "test_PLK_Mini_labels"
#   "test_PLT_VIL_Mini_labels"
)

for DATASET in "${DATASETS[@]}"
do
  echo "====================================================="
  echo "开始用纯 Base 模型预测数据集: $DATASET"
  echo "====================================================="
  
  cat <<EOF > examples/bioreason/temp_predict_baseline.yaml
### 基础设置
# 【核心注意】这里绝对不能用 models/Qwen2.5-VL-7B-BioReason-SFT！
# 请替换为你服务器上最初始、没经过任何微调的 Qwen 官方权重路径 (比如下面这个，请根据实际情况修改)
model_name_or_path: /data1/pretrained/bioclip/models/Qwen2.5-VL-7B

# 不写 adapter_name_or_path，这就代表不加载任何微调参数！
stage: sft
do_predict: true
finetuning_type: lora
template: qwen2_vl

### 动态替换的参数 (存入 baseline 结果文件夹)
eval_dataset: $DATASET
output_dir: saves/bioreason/predict_results_baseline/$DATASET

### 预测参数 (刚才测试过 32 毫无压力，你可以写 16 或 32)
per_device_eval_batch_size: 32
predict_with_generate: true
max_new_tokens: 512
fp16: true
EOF
  
  # 使用 4 张卡火力全开跑推理
  FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=4 llamafactory-cli train examples/bioreason/temp_predict_baseline.yaml
    
done

echo "所有数据集的 Baseline 预测已全部完成！"