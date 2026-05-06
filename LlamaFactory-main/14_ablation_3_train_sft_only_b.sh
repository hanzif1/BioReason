#!/bin/bash

echo "====================================================="
echo "开始消融实验：仅使用 Source B (Reasoning) 进行 SFT"
echo "====================================================="

FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=5 llamafactory-cli train examples/bioreason/sft_only_b.yaml

echo "消融实验 SFT 训练完成！"