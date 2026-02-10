for pathology in appendicitis cholecystitis diverticulitis pancreatitis pneumonia "pulmonary embolism" pericarditis
do
    CUDA_VISIBLE_DEVICES=4 python infer.py \
        model="Llama3Instruct8B" \
        pathology="$pathology"
done

for pathology in appendicitis cholecystitis diverticulitis pancreatitis pneumonia "pulmonary embolism" pericarditis
do
    CUDA_VISIBLE_DEVICES=4 python infer.py \
        model="Llama3Instruct70B" \
        pathology="$pathology"
done

for pathology in appendicitis cholecystitis diverticulitis pancreatitis pneumonia "pulmonary embolism" pericarditis
do
    CUDA_VISIBLE_DEVICES=4 python infer.py \
        model="Deepseek-Llama70B-distill" \
        pathology="$pathology"
done

for pathology in appendicitis cholecystitis diverticulitis pancreatitis pneumonia "pulmonary embolism" pericarditis
do
    CUDA_VISIBLE_DEVICES=2 python infer.py \
        model="Llama3Instruct8B" \
        criteria=1 \
        pathology="$pathology"
done

for pathology in appendicitis cholecystitis diverticulitis pancreatitis pneumonia "pulmonary embolism" pericarditis
do
    CUDA_VISIBLE_DEVICES=6 python infer.py \
        model="Llama3Instruct70B" \
        criteria=1 \
        pathology="$pathology"
done

for pathology in appendicitis cholecystitis diverticulitis pancreatitis pneumonia "pulmonary embolism" pericarditis
do
    CUDA_VISIBLE_DEVICES=6 python infer.py \
        model="Deepseek-Llama70B-distill" \
        criteria=1 \
        pathology="$pathology"
done
