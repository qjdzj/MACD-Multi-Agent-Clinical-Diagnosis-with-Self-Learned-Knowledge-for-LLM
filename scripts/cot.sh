for pathology in appendicitis cholecystitis diverticulitis pancreatitis pneumonia "pulmonary embolism" pericarditis
do
    CUDA_VISIBLE_DEVICES=5 python infer.py \
        prompt_template=COT \
        guideline=1 \
        model="Llama3Instruct8B" \
        pathology="$pathology"
done

for pathology in appendicitis cholecystitis diverticulitis pancreatitis pneumonia "pulmonary embolism" pericarditis
do
    CUDA_VISIBLE_DEVICES=5 python infer.py \
        prompt_template=COT \
        guideline=1 \
        model="Llama3Instruct70B" \
        pathology="$pathology"
done

for pathology in appendicitis cholecystitis diverticulitis pancreatitis pneumonia "pulmonary embolism" pericarditis
do
    CUDA_VISIBLE_DEVICES=5 python infer.py \
        prompt_template=COT \
        guideline=1 \
        model="Deepseek-Llama70B-distill" \
        pathology="$pathology"
done
