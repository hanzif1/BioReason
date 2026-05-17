#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=[account]
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=bioclip-eval
#SBATCH --time=8:00:00
#SBATCH --mem=400GB

TEST_SET_DIR="/data0/data/A_bioclip"

export CUDA_VISIBLE_DEVICES=6,7
export HF_ENDPOINT="https://hf-mirror.com"

LOG_FILEPATH="../storage/logs"
MODEL_TYPE="hf-hub:imageomics/bioclip-2"
PRETRAINED=False
# BASE_DIR="/data2/datasets/your_actual_data_folder" 


echo "=========================================================================================="
echo "STARTING GROUP: data1 (CameraTrap)"
echo "=========================================================================================="

TASK_TYPE="all"
TEXT_TYPE="taxon_com"

DATA_ROOTS=(
    "${TEST_SET_DIR}/CameraTrap/data/test/"
    "${TEST_SET_DIR}/CameraTrap/data/test/"
    "${TEST_SET_DIR}/CameraTrap/data/test/"
    "${TEST_SET_DIR}/CameraTrap/data/test/"
    "${TEST_SET_DIR}/CameraTrap/data/test/"
)
LABEL_FILES=(
    "${TEST_SET_DIR}/CameraTrap/desert-lion-balanced.csv"
    "${TEST_SET_DIR}/CameraTrap/ENA24-balanced.csv"
    "${TEST_SET_DIR}/CameraTrap/island-balanced.csv"
    "${TEST_SET_DIR}/CameraTrap/orinoquia-balanced.csv"
    "${TEST_SET_DIR}/CameraTrap/ohio-small-animals-balanced.csv"
)

for i in "${!DATA_ROOTS[@]}"; do
    DATA_ROOT=${DATA_ROOTS[$i]}
    LABEL_FILE=${LABEL_FILES[$i]}

    python -m src.evaluation.zero_shot_iid \
            --model $MODEL_TYPE \
            --batch-size 256 \
            --data_root $DATA_ROOT \
            --pretrained $PRETRAINED \
            --label_filename $LABEL_FILE \
            --log $LOG_FILEPATH \
            --text_type $TEXT_TYPE \
            --output_group "data1"\

#     python -m src.evaluation.few_shot \
#             --model $MODEL_TYPE \
#             --batch-size 256 \
#             --data_root $DATA_ROOT \
#             --pretrained $PRETRAINED \
#             --label_filename $LABEL_FILE \
#             --log $LOG_FILEPATH \
#             --task_type $TASK_TYPE \
#             --nfold 5 \
#             --kshot_list 1 5 \

done

echo "=========================================================================================="
echo "STARTING GROUP: data2 (OpenML)"
echo "=========================================================================================="

TEXT_TYPE="asis"
OPENML_BASE="/home/maviuserwq/.cache/openml/org/openml/www/datasets"

DATA_ROOTS=(
    "${OPENML_BASE}/44282/PLK_Mini/images"
    "${OPENML_BASE}/44306/INS_Mini/images"
    "${OPENML_BASE}/44292/INS_2_Mini/images"
    "${OPENML_BASE}/44293/PLT_NET_Mini/images"
    "${OPENML_BASE}/44302/FNG_Mini/images"
    "${OPENML_BASE}/44286/PLT_VIL_Mini/images"
    "${OPENML_BASE}/44299/MED_LF_Mini/images"
    "${TEST_SET_DIR}/nabird/images/"
)
LABEL_FILES=(
    "${OPENML_BASE}/44282/PLK_Mini/labels.csv"
    "${OPENML_BASE}/44306/INS_Mini/labels.csv"
    "${OPENML_BASE}/44292/INS_2_Mini/labels.csv"
    "${OPENML_BASE}/44293/PLT_NET_Mini/labels.csv"
    "${OPENML_BASE}/44302/FNG_Mini/labels.csv"
    "${OPENML_BASE}/44286/PLT_VIL_Mini/labels.csv"
    "${OPENML_BASE}/44299/MED_LF_Mini/labels.csv"
    "${TEST_SET_DIR}/nabird/metadata.csv"
)

for i in "${!DATA_ROOTS[@]}"; do
    DATA_ROOT=${DATA_ROOTS[$i]}
    LABEL_FILE=${LABEL_FILES[$i]}

    python -m src.evaluation.zero_shot_iid \
            --model $MODEL_TYPE \
            --batch-size 256 \
            --data_root $DATA_ROOT \
            --pretrained $PRETRAINED \
            --label_filename $LABEL_FILE \
            --log $LOG_FILEPATH \
            --text_type $TEXT_TYPE \
            --output_group "data2"\

#     python -m src.evaluation.few_shot \
#             --model $MODEL_TYPE \
#             --batch-size 256 \
#             --data_root $DATA_ROOT \
#             --pretrained $PRETRAINED \
#             --label_filename $LABEL_FILE \
#             --log $LOG_FILEPATH \
#             --task_type $TASK_TYPE \
#             --nfold 5 \
#             --kshot_list 1 5 \

done

# echo "=========================================================================================="
# echo "STARTING GROUP: data3 (Rare Species)"
# echo "=========================================================================================="
# TEXT_TYPE="taxon_com"
# DATA_ROOT="${TEST_SET_DIR}/rare-species/"
# LABEL_FILE="${TEST_SET_DIR}/rare-species/metadata.csv"

# python -m src.evaluation.zero_shot_iid \
#         --model $MODEL_TYPE \
#         --batch-size 256 \
#         --data_root $DATA_ROOT \
#         --pretrained $PRETRAINED \
#         --label_filename $LABEL_FILE \
#         --log $LOG_FILEPATH \
#         --text_type $TEXT_TYPE \
#         --output_group "data3"\

# python -m src.evaluation.few_shot \
#         --model $MODEL_TYPE \
#         --batch-size 256 \
#         --data_root $DATA_ROOT \
#         --pretrained $PRETRAINED \
#         --label_filename $LABEL_FILE \
#         --log $LOG_FILEPATH \
#         --task_type $TASK_TYPE \
#         --nfold 5 \
#         --kshot_list 1 5 \
