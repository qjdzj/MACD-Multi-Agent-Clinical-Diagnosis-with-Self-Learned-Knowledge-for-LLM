import os
import re
import pickle
import glob
import csv
import torch
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from datetime import datetime
import string
from negspacy.negation import Negex
from thefuzz import fuzz
from utils.nlp import keyword_positive, remove_punctuation


nlp = spacy.load("en_core_sci_lg")
nlp.add_pipe(
    "negex",
    config={
        "chunk_prefix": ["no"],
    },
    last=True,
)

PATHOLOGY_MAPPING_RULES = {
    "appendicitis": {
        "alternatives": [{"location": "appendi", "modifiers": ["gangren", "infect", "inflam", "abscess", "rupture", "necros", "perf"]}],
        "gracious_alternatives": [
            {"location": "Appendercitis", "modifiers": []}
        ]
    },
    "cholecystitis": {
        "alternatives": [
            {"location": "gallbladder", "modifiers": ["gangren", "infect", "inflam", "abscess", "necros", "perf"]},
            {"location": "cholangitis", "modifiers": ["cholangitis"]}
        ],
        "gracious_alternatives": [
            {"location": "acute gallbladder", "modifiers": ["disease", "attack"]},
            {"location": "acute biliary", "modifiers": ["colic"]}
        ]
    },
    "diverticulitis": {
        "alternatives": [{"location": "diverticul", "modifiers": ["inflam", "infect", "abscess", "perf", "rupture"]}],
        "gracious_alternatives": [
            {"location": "acute colonic", "modifiers": ["perfor"]},
            {"location": "sigmoid", "modifiers": ["perfor", "colitis"]}
        ]
    },
    "pancreatitis": {
        "alternatives": [{"location": "pancrea", "modifiers": ["gangren", "infect", "inflam", "abscess", "necros"]}],
        "gracious_alternatives": []
    },
    "pericarditis": {
        "alternatives": [
            {"location": "pericard", "modifiers": ["inflammation", "inflammatory disease"]},
            {"location": "pericardial", "modifiers": ["inflammation", "inflammatory change", "pericardial inflammation"]},
            {"location": "heart sac", "modifiers": ["inflammation"]},
            {"location": "cardiac membrane", "modifiers": ["inflammation"]}
        ],
        "gracious_alternatives": [
            {"location": "pericard", "modifiers": ["effusion", "fluid accumulation", "thickening", "fibrosis", "adhesion", "calcification"]},
            {"location": "pericardial", "modifiers": ["effusion", "thickening", "fluid", "fibrosis", "adhesions"]}
        ]
    },
    "pneumonia": {
        "alternatives": [
            {"location": "lung", "modifiers": ["infect", "pneumonitis", "bacterial", "viral", "aspiration"]},
            {"location": "pneumonia", "modifiers": ["acute", "pneumonitis", "bacterial", "viral", "aspiration"]}
        ],
        "gracious_alternatives": [{"location": "respiratory", "modifiers": ["infection", "bacterial", "viral", "fungal"]}]
    },
    "pulmonary embolism": {
        "alternatives": [
            {"location": "pulmonary", "modifiers": ["embolism", "embolus", "thrombus"]},
            {"location": "pe", "modifiers": []}
        ],
        "gracious_alternatives": [],
        "regex_patterns": [
            re.compile(r'\b(?:pulmonary\s+embolism|PE)\b', re.IGNORECASE),
            re.compile(r'\b(?:pulmonary|PE)\b.*?\b(?:embolus|thrombus)\b', re.IGNORECASE)
        ]
    }
}


def map_pathology_name(
    original_sentence: str,
    extracted_diagnosis: str,
    mapping_rules: dict
) -> str:
    """
    Perform global mapping of the extracted diagnosis name.
    It iterates through all known pathology rules to find the best match.
    """
    if not extracted_diagnosis:
        return ""

    # Iterate through each disease and its rules in the rule library
    for pathology_name, rules in mapping_rules.items():

        if 'regex_patterns' in rules:
            for pattern in rules['regex_patterns']:
                match = pattern.search(extracted_diagnosis)
                if match and keyword_positive(original_sentence, match.group(0)):
                    return pathology_name  

        answer_for_search = remove_punctuation(extracted_diagnosis.lower())
        for word in answer_for_search.split():
            if fuzz.ratio(word, pathology_name) > 90 and keyword_positive(original_sentence, word):
                return pathology_name 

        all_alternatives = rules.get("alternatives", []) + rules.get("gracious_alternatives", [])
        for alt_rule in all_alternatives:
            loc = alt_rule["location"]
            loc_pattern = r'\b' + re.escape(loc) + r'\b'
        
            if not alt_rule["modifiers"]:
                if re.search(loc_pattern, answer_for_search) and keyword_positive(original_sentence, loc):
                        return pathology_name
                continue
            
            for mod in alt_rule["modifiers"]:
                mod_pattern = r'\b' + re.escape(mod) + r'\b'
                if (re.search(loc_pattern, answer_for_search) and
                    re.search(mod_pattern, answer_for_search) and
                    keyword_positive(original_sentence, loc) and
                    keyword_positive(original_sentence, mod)):
                    return pathology_name

    return extracted_diagnosis


def read_from_pickle_file(filename):
    with open(filename, "rb") as f:
        while True:
            try: yield pickle.load(f)
            except EOFError: break


def load_data_from_pkl(path: str) -> dict:
    file_path = glob.glob(path)[0]
    results_list = [r for r in read_from_pickle_file(file_path)]
    return {k: v for d in results_list for k, v in d.items()}


class DiagnosisComparator:
    def __init__(self, model_path: str, threshold: float = 0.8):
        self.model_path = model_path
        self.threshold = threshold
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        if self.model is None and self.model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")


    def _encode_texts(self, texts: list, max_length: int = 128) -> np.ndarray:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def _normalize_for_comparison(self, diagnosis: str) -> str:
        if not diagnosis: return ""
        norm_dx = diagnosis.lower()
        norm_dx = re.sub(r'\([^)]*\)', '', norm_dx)
        split_phrases = [' with ', ' secondary to ', ' due to ', '/', ' vs ']
        for phrase in split_phrases:
            if phrase in norm_dx:
                norm_dx = norm_dx.split(phrase)[0]
        return norm_dx.strip()

    def compare(self, diagnoses: list) -> tuple[str, dict]:
        if len(diagnoses) < 2: return "Insufficient Information", {}
        if not self.model_path: return "Error: Model path not configured or failed to load", {}

        self._load_model()
        if not self.model: return "Error: Model failed to load successfully", {}
        normalized_diagnoses = [self._normalize_for_comparison(dx) for dx in diagnoses]
        scores = {}

        if len(diagnoses) == 2:
            if any(not dx for dx in normalized_diagnoses):
                return "❌ Inconsistent (Empty diagnosis exists)", {}
            embeddings = self._encode_texts(normalized_diagnoses)
            score = cosine_similarity(embeddings)[0, 1]
            scores['M1-M2'] = score
            status = "✅ Consistent" if score >= self.threshold else "❌ Inconsistent"
            return status, scores
        
        elif len(diagnoses) == 3:
            non_empty_diags = [(i, dx) for i, dx in enumerate(normalized_diagnoses) if dx]
            
            if len(non_empty_diags) == 3:
                embeddings = self._encode_texts(normalized_diagnoses)
                similarity_matrix = cosine_similarity(embeddings)
                s12 = similarity_matrix[0, 1]
                s13 = similarity_matrix[0, 2]
                s23 = similarity_matrix[1, 2]
                scores = {'M1-M2': s12, 'M1-M3': s13, 'M2-M3': s23}
                are_all_consistent = (s12 >= self.threshold and s13 >= self.threshold and s23 >= self.threshold)
                status = "✅ Consistent" if are_all_consistent else "❌ Inconsistent"
                return status, scores
            
            elif len(non_empty_diags) == 2:
                indices, texts_to_compare = zip(*non_empty_diags)
                embeddings = self._encode_texts(list(texts_to_compare))
                score = cosine_similarity(embeddings)[0, 1]
                score_key = f'M{indices[0]+1}-M{indices[1]+1}'
                scores[score_key] = score
                status = "✅ Consistent" if score >= self.threshold else "❌ Inconsistent"
                return status, scores
            
            else:
                return "❌ Inconsistent (Insufficient valid diagnoses)", {}
        
        return "Insufficient Information", {}


def run_pathology_comparison(
    base_dir: str,
    model_list: list,
    pathology_name: str,
    roundber: int,
    comparator: DiagnosisComparator,
    total_eval_log: str
) -> tuple[list, list]:
    
    print(f"\n--- Analyzing Pathology: {pathology_name} | Round: {roundber} ---")

    pkl_path_patterns = [os.path.join(base_dir, f"{pathology_name}_{model}_*FULL_INFO_*_results.pkl") for model in model_list]
    # pkl_path_patterns = [os.path.join(base_dir, f"{pathology_name}_{model}.pkl") for model in model_list]
    all_model_data = [load_data_from_pkl(path) for path in pkl_path_patterns]
    valid_model_data_map = {model_list[i]: data for i, data in enumerate(all_model_data) if data}

    if len(valid_model_data_map) < 2:
        print(f"Failed to load valid data for at least two models for pathology '{pathology_name}', skipping comparison.")
        return [], []
    common_ids = set(list(valid_model_data_map.values())[0].keys())

    for data in list(valid_model_data_map.values())[1:]:
        common_ids.intersection_update(data.keys())

    if not common_ids:
        print(f"No common case IDs found between successfully loaded models for pathology '{pathology_name}', skipping comparison.")
        return [], []
    print(f"Found {len(common_ids)} common case IDs among {len(valid_model_data_map)} models, starting analysis...")

    consistent_results_for_csv = []
    inconsistent_ids_for_patho = []
    all_results_for_csv = []
    model_names_with_data = list(valid_model_data_map.keys())
    def extract_diagnosis_with_regex(raw_text: str) -> str:
        if not isinstance(raw_text, str):
            return "Invalid Input Type"
        prediction = "Final Diagnosis: " + raw_text
        regex = r"(Final )?Diagnosis:([\s\S]*?)(?=[\.\n].*Treatment.*:|$)"
        match = re.search(regex, prediction, flags=re.IGNORECASE)
        diagnosis = match.group(2).strip() if match else raw_text
        cleanup_keywords = [
            "rationale", "note", "recommendation", "explanation", "finding",
            "other.*diagnos.*include", "other.*diagnos.*considered(?: were)?",
            "management", "action", "plan", "reasoning", "reasons", "assessment",
            "justification", "tests", "additional diagnoses", "notification",
            "impression", "background", "additional findings include",
            "diagnostic criteria.*", "diagnostic criteria met", "criteria met",
            "characterized by","rare", "Rare"
        ]
        for keyword in cleanup_keywords:
            diagnosis = re.sub(rf"{keyword}[s]?:.*", "", diagnosis, flags=re.IGNORECASE | re.DOTALL)
        diagnosis = re.sub(r":\n[\s\S]*", "", diagnosis)
        diagnosis = re.sub(r"^.*:\s*", "", diagnosis)
        diagnosis = re.sub(r"\s+-\s+.*", "", diagnosis, flags=re.DOTALL)
        diagnosis = re.sub(r":\n[\s\S]*", "", diagnosis)
        parts = re.split(r"[,.\n]|(?:\s*\b(?:and|or|vs[.]?)\b\s*)", diagnosis)
        diagnosis = parts[0].strip()
        diagnosis = re.sub(r'\*\*', '', diagnosis).strip()
        return diagnosis
    
    for case_id in sorted(list(common_ids)):
        original_diagnoses = [valid_model_data_map[model_name][case_id] for model_name in model_names_with_data]
        extracted_diagnoses = [extract_diagnosis_with_regex(text) for text in original_diagnoses]
        
        mapped_diagnoses = []
        for i in range(len(extracted_diagnoses)):
            mapped_diag = map_pathology_name(
                original_sentence=original_diagnoses[i],
                extracted_diagnosis=extracted_diagnoses[i],
                mapping_rules=PATHOLOGY_MAPPING_RULES
            )
            mapped_diagnoses.append(mapped_diag)

        status, scores = comparator.compare(mapped_diagnoses)
        scores_str = " | ".join([f"{key}: {value:.4f}" for key, value in scores.items()])
        all_results_for_csv.append({
            'Pathology': pathology_name,
            'Round': roundber,
            'Case_ID': int(case_id),
            'Status': status,
            'All_Diagnoses': " | ".join(mapped_diagnoses),
            'Similarity_Scores': scores_str
        })

        if status == "✅ Consistent":
            consistent_results_for_csv.append({
                'Pathology': pathology_name, 'Round': roundber, 'Case_ID': int(case_id),
                'All_Consistent_Diagnoses': " | ".join(mapped_diagnoses),
                'Similarity_Scores': scores_str
            })
        else:
            inconsistent_ids_for_patho.append(int(case_id))

    if all_results_for_csv:
        filename_csv = os.path.join(base_dir, f"{pathology_name}_round{roundber}_consistency_patient_list.csv")
        with open(filename_csv, 'w', newline='', encoding='utf-8') as f:
            header = ['Pathology', 'Round', 'Case_ID', 'Status', 'All_Diagnoses', 'Similarity_Scores']
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(all_results_for_csv)
        print(f"\n[✔] All case comparison results for pathology '{pathology_name}' have been saved to: {os.path.abspath(filename_csv)}")

    if inconsistent_ids_for_patho:
        print("\n[!] Inconsistent cases found, generating separate full data PKL files for them...")
        for model_name in model_names_with_data:
            inconsistent_data_for_model = {
                str(case_id): valid_model_data_map[model_name][str(case_id)]
                for case_id in inconsistent_ids_for_patho
                if str(case_id) in valid_model_data_map[model_name]
            }
            if inconsistent_data_for_model:
                new_filename_pkl = os.path.join(
                    base_dir,
                    f"{pathology_name}_{model_name}_inconsistent_FULL_INFO_round{roundber}_results.pkl"
                )
                with open(new_filename_pkl, 'wb') as f:
                    pickle.dump(inconsistent_data_for_model, f)
                print(f"    - Saved inconsistent case data for model '{model_name}' to: {os.path.abspath(new_filename_pkl)}")

    total_common = len(common_ids)
    consistent_count = len(consistent_results_for_csv)
    inconsistent_count = len(inconsistent_ids_for_patho)
    print(f"\n   --- {pathology_name} | Round: {roundber} | Analysis Summary ---")

    if total_common > 0:
        consistency_rate = (consistent_count / total_common) * 100
        print(f"   Consistency rate: {consistency_rate:.2f}% ({consistent_count} / {total_common})")
        with open(total_eval_log, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if os.path.getsize(total_eval_log) == 0:
                writer.writerow(['Timestamp', 'Pathology', 'Round', 'Consistent_Count', 'Inconsistent_Count', 'Total_Common_Cases'])
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), pathology_name, roundber,
                consistent_count, inconsistent_count, total_common
            ])
    else:
        print("   No common cases available for analysis.")
    return consistent_results_for_csv, inconsistent_ids_for_patho

def main():
    round = 3
    threshold = 0.83
    base_dir = f"/data2/kunzhang/MIMIC-CDM/MIMIC-Clinical-Decision-Making-Framework-llama3.1/eval_round2/professional/round{round}"

    models = [
        "Llama-3.1-8B-Instruct",
        "DeepSeek-R1-Distill-Llama-70B",
        "Llama-3.1-70B-Instruct",
    ]
    pathology_name = [
        "appendicitis",
        "cholecystitis",
        "diverticulitis",
        "pancreatitis",
        "pericarditis",
        "pneumonia",
        "pulmonary embolism",
    ]

    biobert_path = '/data2/kunzhang/MIMIC-CDM_Models/BioBert' 
    total_eval_log = "/data2/kunzhang/MIMIC-CDM/MIMIC-Clinical-Decision-Making-Framework-llama3.1/eval_round2/professional/total_eval.csv"
    all_cosistent_cases = "/data2/kunzhang/MIMIC-CDM/MIMIC-Clinical-Decision-Making-Framework-llama3.1/eval_round2/professional/all_pathologies_consistent_ids.pkl"
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.dirname(total_eval_log), exist_ok=True)

    main_comparator = DiagnosisComparator(biobert_path, threshold)
    all_consistent_ids_aggregator = defaultdict(lambda: defaultdict(list))

    for pathology in pathology_name:
        consistent_results, inconsistent_ids = run_pathology_comparison(
            base_dir=base_dir,
            model_list=models,
            pathology_name=pathology,
            roundber=round,
            comparator=main_comparator,
            total_eval_log=total_eval_log
        )
        if inconsistent_ids:
            filename_pkl = os.path.join(base_dir,f"{pathology}_round{round}_incosistent_patient_id_list.pkl")
            with open(filename_pkl, 'wb') as f:
                pickle.dump(inconsistent_ids, f)
        if consistent_results:
            if 'multi_diagnosis' not in all_consistent_ids_aggregator[pathology]:
                all_consistent_ids_aggregator[pathology]['multi_diagnosis'] = []
            consistent_ids_this_patho = [item['Case_ID'] for item in consistent_results]
            all_consistent_ids_aggregator[pathology]['multi_diagnosis'].extend(consistent_ids_this_patho)

    if all_consistent_ids_aggregator:
        try:
            with open(all_cosistent_cases, 'rb') as f: existing_data = pickle.load(f)
        except FileNotFoundError:
            existing_data = {}
        for pathology, data in all_consistent_ids_aggregator.items():
            if pathology not in existing_data: existing_data[pathology] = {'multi_diagnosis': []}
            new_ids = data['multi_diagnosis']
            existing_data[pathology]['multi_diagnosis'] = sorted(list(set(existing_data[pathology]['multi_diagnosis'] + new_ids)))
        with open(all_cosistent_cases, 'wb') as f:
            pickle.dump(existing_data, f)
        print(f"\nGlobal consistent IDs have been updated and saved to: {os.path.abspath(all_cosistent_cases)}")


if __name__ == '__main__':
    main()