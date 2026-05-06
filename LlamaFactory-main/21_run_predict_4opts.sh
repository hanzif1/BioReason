#!/bin/bash
DATASETS=("test_CameraTrap_desert-lion-balanced" "test_CameraTrap_island-balanced" "test_CameraTrap_orinoquia-balanced" "test_CameraTrap_ENA24-balanced" "test_CameraTrap_ohio-small-animals-balanced")

for DATASET in "${DATASETS[@]}"; do
  echo "====================================================="
  echo "处理数据并预测 4 选项数据集: $DATASET"
  echo "====================================================="
  python temp_ablation.py $DATASET 4
  ABLATION_DATASET="${DATASET}_4opts"
  
  cat <<YAML_EOF > examples/bioreason/temp_predict_grpo_4opts.yaml
model_name_or_path: models/Qwen2.5-VL-7B-BioReason-SFT
adapter_name_or_path: saves/bioreason/grpo_trl/checkpoint-final
stage: sft
do_predict: true
finetuning_type: lora
template: qwen2_vl
eval_dataset: $ABLATION_DATASET
output_dir: saves/bioreason/predict_results_grpo_4opts_rand/$DATASET
per_device_eval_batch_size: 32
predict_with_generate: true
max_new_tokens: 512
fp16: true
YAML_EOF
  
  FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=5 llamafactory-cli train examples/bioreason/temp_predict_grpo_4opts.yaml
done
