import os 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pickle
import logging
from datetime import datetime

import numpy as np
from os.path import join
from transformers import AutoTokenizer,BitsAndBytesConfig, AutoModelForCausalLM, pipeline
import torch

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter



def load_correct_diag_result(file_path, patho,model,i):
    with open(join(file_path, f"{patho}_{model}_PLI_N_INFO_CORRECT_30_{i}.pkl"), 'rb') as f:
        correct_diag_result = str(pickle.load(f))
        text = Document(page_content = correct_diag_result)
    return text

def text_splitter(text, max_length=4096):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=max_length,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(text)
    return split_docs

def prepare_diagnostic_reports(diagcrit_data):
    diagnostic_reports = ""
    for pid, report in diagcrit_data.items():
        diagnostic_reports += f"Patient ID: {pid}\nReport:\n{report}\n\n"
    return diagnostic_reports

def create_prompt(correct_diag_result):
    prompt = f"""
[Role]
You are a medical artificial intelligence assistant. Your task is to review the given report and summarize the diagnostic evidence for the identified disease. 
[/Role]

[INSTRUCTIONS]
1. Analyze the report and identify the main disease.
2. Summarize the diagnostic evidence into two structured categories:
   - General Criteria: The 5 most relevant common clinical manifestations and diagnostic findings typically associated with the disease.
   - Rare Criteria: The 5 most relevant specific or unique clinical manifestations and diagnostic findings observed in a subset of patients with the disease.
3. Only one response is required. Do not repeat or provide multiple outputs.
4. Ensure the summarized content is specific, concise, and directly extracted from the input report. Avoid adding explanations, references, or unnecessary details.
5. Strictly adhere to the specified output format.
[/INSTRUCTIONS]

[REPORT]
{correct_diag_result}
[/REPORT]

[OUTPUT FORMAT]
Disease: (Identified disease)
General Criteria:
1. (Most relevant common clinical manifestation or diagnostic finding)
2. (Most relevant common clinical manifestation or diagnostic finding)
3. (Most relevant common clinical manifestation or diagnostic finding)
4. (Most relevant common clinical manifestation or diagnostic finding)
5. (Most relevant common clinical manifestation or diagnostic finding)
Rare Criteria:
1. (Most relevant specific or unique clinical manifestation or diagnostic find
2. (Most relevant specific or unique clinical manifestation or diagnostic finding)
3. (Most relevant specific or unique clinical manifestation or diagnostic finding)
4. (Most relevant specific or unique clinical manifestation or diagnostic finding)
5. (Most relevant specific or unique clinical manifestation or diagnostic finding)
[/OUTPUT FORMAT]

[REQUIEMENTS]
- Each category must contain exactly 5 summarized criteria. If fewer than 5 rare criteria are available, state "Not available" for the remaining items.
- Focus solely on diagnostic evidence relevant to clinical practice, directly extracted from the report.
- Ensure the response is generated only once, with no repetitions or additional outputs.
Your response must strictly adhere to the above format. Any repeated or additional outputs will be considered a deviation.**
[/REQUIEMENTS]

[OUTPUT]
"""
    return prompt

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(
                model_id,
            )

    eot = "<|eot_id|>"
    eot_id = tokenizer.convert_tokens_to_ids(eot)
    tokenizer.pad_token = eot
    tokenizer.pad_token_id = eot_id

    print("loaded tokenizer")
    bb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bb_cfg,
    )

    text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=8192,
    temperature=0.01,
    top_p=0.95,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
    truncation=False  
)
    return text_gen_pipeline

def generate_summary(text_gen_pipeline, prompt):
    input_text = str(f"{prompt}")
    outputs = text_gen_pipeline(input_text, max_length=8192, num_return_sequences=1)
    summary = outputs[0]['generated_text']
    return summary

def main():
    for model in [
        "Llama-3.1-70B-Instruct",
        "Llama-3.1-70B-Instruct",
          "DeepSeek-R1-Distill-Llama-70B",
        ]:
        
        correct_diag_result_file = join(f"") # path to the correct diagnoses results
        model_id = join(f"/data2/kunzhang/MIMIC-CDM_Models/{model}") 
        text_gen_pipeline = load_model(model_id)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = f""

        with open(output_file, "w") as f:
            for patho in [
                "appendicitis", 
                "cholecystitis", 
                "diverticulitis", 
                "pancreatitis",
                "pericarditis",
                "pneumonia",
                "pulmonary embolism",
        ]:
                n = 3
                for i in range(n):
                    correct_diag_result = load_correct_diag_result(correct_diag_result_file, patho,model,i)
                    
                    prompt = create_prompt(correct_diag_result)
                    
                    summary = generate_summary(text_gen_pipeline, prompt)
                    
                    print(summary)
                    f.write(f"Pathology:{patho}, Iteration:{i}\n\n")
                    f.write(summary+"\n\n\n")

if __name__ == "__main__":
    main()

