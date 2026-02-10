for pathology in appendicitis cholecystitis diverticulitis pancreatitis pneumonia "pulmonary embolism" pericarditis
do
    CUDA_VISIBLE_DEVICES=3 python infer.py \
        model="Llama3Instruct8B" \
        guideline=1 \
        pathology="$pathology"
done

for pathology in appendicitis cholecystitis diverticulitis pancreatitis pneumonia "pulmonary embolism" pericarditis
do
    CUDA_VISIBLE_DEVICES=3 python infer.py \
        model="Llama3Instruct70B" \
        guideline=1 \
        pathology="$pathology"
done

for pathology in appendicitis cholecystitis diverticulitis pancreatitis pneumonia "pulmonary embolism" pericarditis
do
    CUDA_VISIBLE_DEVICES=3 python infer.py \
        model="Deepseek-Llama70B-distill" \
        guideline=1 \
        pathology="$pathology"
done
