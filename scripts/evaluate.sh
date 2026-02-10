for run_name in "Llama-3.1-8B-Instruct_guideline_seed1" \
"Llama-3.1-8B-Instruct_guideline_seed397" \
"Llama-3.1-8B-Instruct_guideline_seed783" \
"Llama-3.1-8B-Instruct_guideline_seed1024" \
"Llama-3.1-8B-Instruct_guideline_seed1565" \
"Llama-3.1-8B-Instruct_guideline_seed2025" \
"Llama-3.1-8B-Instruct_guideline_seed2699" \
"Llama-3.1-8B-Instruct_guideline_seed3298" \
"Llama-3.1-8B-Instruct_guideline_seed4848" \
"Llama-3.1-8B-Instruct_guideline_seed5866" \

do
    python evaluate.py \
        +run_name="${run_name}"
done