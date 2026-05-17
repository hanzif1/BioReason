# BioReason: Retrieve-Then-Reason MLLMs for Interpretable Fine-Grained Species Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model: Qwen2.5-VL](https://img.shields.io/badge/Base_Model-Qwen2.5--VL--7B-red)](https://github.com/QwenLM/Qwen2.5-VL)

This is the official repository for the paper **"BioReason: Retrieve-Then-Reason MLLMs for Interpretable Fine-Grained Species Classification"**.

## 📖 Introduction

Fine-grained species classification (FGSC) requires distinguishing among hundreds of thousands of visually similar taxa. While contrastive vision-language models like BioCLIP offer strong global matching performance, they operate as black boxes. Conversely, Multimodal Large Language Models (MLLMs) offer interpretable reasoning but lack fine-grained domain knowledge and suffer from hallucination. 

**BioReason** is a novel retrieve-then-reason framework that bridges this gap. By leveraging BioCLIP to retrieve a top-K candidate set from an expansive 454K label space, we convert open-domain classification into a multiple-choice reasoning task. We further apply a decoupled supervised fine-tuning strategy and outcome-driven reinforcement learning to elicit emergent reasoning capabilities. BioReason achieves state-of-the-art performance across ten benchmark datasets while providing explicit, biologically meaningful chain-of-thought (CoT) explanations.

## 🛠️ Installation

1. Clone this repository and navigate to the working directory:
   ```bash
   git clone [https://github.com/hanzif1/BioReason.git](https://github.com/hanzif1/BioReason.git)
   cd BioReason
2. Activate the Conda environment:
    ```bash
    source ./miniforge3/bin/activate
    conda activate llama
3. Install the required dependencies (We use LLaMA-Factory as our primary training framework):
    ```bash
    pip install -r requirements.txt
## 🗂️ Data Preparation

Place your JSON data files in the `LlamaFactory-main/data/` directory. The primary datasets used are:

* `A_biocap_captioning.json`
* `B_bioreason_reasoning.json`
If you plan to run Direct Preference Optimization (DPO), prepare the pairwise preference data using our script:
    ```bash
    python LlamaFactory-main/9_prepare_dpo_data.py
Note: Make sure to register the DPO dataset in `data/dataset_info.json` and set `"ranking": true`.
## 🚀 Training
##### Stage 1: Supervised Fine-Tuning (SFT)
Run the SFT process using LLaMA-Factory with the provided configuration file.
```bash
CUDA_VISIBLE_DEVICES=5 llamafactory-cli train examples/bioreason/sft.yaml
```
The SFT checkpoints will be saved in `saves/bioreason/sft/` (e.g., `checkpoint-6741`).
##### Stage 2: Alignment (GRPO / DPO)
To further elicit reasoning capabilities, you can apply either GRPO or DPO.
##### Option A: GRPO (Recommended)
    CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=4 11_run_trl_grpo.py
    # Alternatively, for single GPU:
    # CUDA_VISIBLE_DEVICES=5 python 11_run_trl_grpo.py
##### Option B: DPO
If using DPO, ensure the learning rate is kept very low (e.g., 5.0e-6) to prevent catastrophic forgetting of SFT knowledge.
```Bash
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=5 llamafactory-cli train examples/bioreason/dpo.yaml
```
DPO checkpoints will be saved to `saves/bioreason/dpo/checkpoint-final.`
## 📊 Prediction and Evaluation
To automatically run predictions across all benchmark datasets:
```Bash
# Standard prediction (uses examples/bioreason/temp_predict.yaml)
./6_run_all_predicts.sh

# For DPO model prediction
./10_run_dpo_predicts.sh
```
Once the predictions are saved to the `saves/bioreason/predict_results_...` directories, evaluate the classification accuracy by running:
```Bash
python 7_evaluate_accuracy.py
```
## 🏆 Main Results
Our BioReason framework demonstrates significant improvements over baseline models. Below is a subset of our evaluation results across different taxonomy groups (measured in Accuracy %):
| Model | nabird | plankton (PLK) | insects (INS) |
| :--- | :---: | :---: | :---: |
| Baseline | 58.8 | 6.1 | 34.9 |
| **BioReason (Ours)** | **53.67** | **25.93** | **38.97** |

(Please refer to the paper for the full comprehensive results across all 10 datasets, including detailed ablation studies.)
## 📝 CitationIf you find this code or our paper useful for your research, please consider citing:

```bibtex
@article{duan2026bioreason,
  title={BioReason: Retrieve-Then-Reason MLLMs for Interpretable Fine-Grained Species Classification},
  author={Duan, Yicheng and Wan, Quan and Zhu, Xingyu and Zhang, Yifan and Wang, Ganlin and Dang, Jisheng and Wang, Jiawei and Wang, Bimei and Tian, Qi and Chua, Tat-Seng},
  journal={ACM Transactions on Information Systems},
  year={2026}
}
```

## 📧 ContactFor any questions regarding the code or paper, please contact:
* `Jisheng Dang: dangjisheng@lzu.edu.cn`
* `Jiawei Wang: jiaweiwang@nus.edu.sg`
