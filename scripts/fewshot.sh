for pathology in appendicitis cholecystitis diverticulitis pancreatitis pneumonia "pulmonary embolism" pericarditis
do
    CUDA_VISIBLE_DEVICES=2 python infer.py \
        model="Llama3Instruct8B" \
        fewshot=1 \
        guideline=1 \
        pathology="$pathology"
done

for pathology in pancreatitis \ 
cholecystitis \
diverticulitis \
pneumonia \
"pulmonary embolism"\
pericarditis \
do
    CUDA_VISIBLE_DEVICES=2 python infer.py \
        model="Llama3Instruct70B" \
        fewshot=1 \
        guideline=1 \
        pathology="$pathology"
done

for pathology in appendicitis cholecystitis diverticulitis pancreatitis pneumonia "pulmonary embolism" pericarditis
do
    CUDA_VISIBLE_DEVICES=2 python infer.py \
        model="Deepseek-Llama70B-distill" \
        fewshot=1 \
        guideline=1 \
        pathology="$pathology"
done
