MODELS=(
    "Llama3Instruct8B"
    "Llama3Instruct70B"
    "Deepseek-Llama70B-distill"
)

SEEDS=(1 397 783 1024 1565 2025 2699 3298 4848 5866)

PATHOLOGIES=(
    "pulmonary embolism"
    "pneumonia"
    "pericarditis"
    "pancreatitis"
    "appendicitis"
    "cholecystitis"
    "diverticulitis"
)

GPU_ID=2

start_time=$(date +%s)

for model_name in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for pathology in "${PATHOLOGIES[@]}"; do
            CUDA_VISIBLE_DEVICES=$GPU_ID python infer.py \
                model="$model_name" \
                pathology="$pathology" \
                seed="$seed" \
                guideline=1
        done
    done
done
