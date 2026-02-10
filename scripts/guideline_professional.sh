for pathology in pneumonia "pulmonary embolism" pericarditis
# appendicitis cholecystitis diverticulitis pancreatitis 
do
    CUDA_VISIBLE_DEVICES=4 python infer.py \
        model="Llama3Instruct8B" \
        guideline=1 \
        pathology="$pathology"
done

for pathology in "pulmonary embolism" pneumonia pericarditispancreatitis appendicitis cholecystitis diverticulitis 

do
    CUDA_VISIBLE_DEVICES=5 python infer.py \
        model="Llama3Instruct70B" \
        guideline=1 \
        pathology="$pathology"
done

for pathology in pneumonia "pulmonary embolism" pericarditis
# appendicitis cholecystitis diverticulitis pancreatitis 

do
    CUDA_VISIBLE_DEVICES=5 python infer.py \
        model="Deepseek-Llama70B-distill" \
        guideline=1 \
        pathology="$pathology"
done
