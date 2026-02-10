from os.path import join

import pickle
import glob
import random

import hydra
from omegaconf import DictConfig

from dataset.utils import load_hadm_from_file
from utils.logging import read_from_pickle_file

evals_path = "" # path to the evaluation results witr model name
output_path = "" # path to save the correct patient IDs and diagnoses


def create_correct_ids():
    # Load patient data and create a list of correct patient IDs for each pathology
    # The diagnostic results are evaluated without the diagnostic criteria
    # for patho in ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]:
    for model in [
        "Llama-3.1-8B-Instruct"
        "Llama-3.1-70B-Instruct",
        "DeepSeek-R1-Distill-Llama-70B"
    ]:
    # load all evaluation results
        all_evals = load_hadm_from_file(
            f"{model}_evals", base_mimic=evals_path
        )

        # Iterate through the data to find patients with Diagnosis = 1
        for disease_list, cases in all_evals.items():
            num=0
            correct_patients = []
            for patient_id, patient_info in cases.items():
                # Extract 'scores' dictionary from patient_info
                scores = patient_info.get('scores', {})
                # Check if Diagnosis score is 1
                if scores.get('Diagnosis') == 1:
                    correct_patients.append(patient_id)
                    num+=1
            # Save the resulting patient IDs to a new .pkl file
            with open(join(output_path, f"{disease_list}_PLI_N.pkl"), 
                    'wb') as output_file:
                pickle.dump(correct_patients, output_file)
            print(f"{disease_list} has {num} correct patients")



def create_correct_id_info():
    # select the 30 cases correct diagnoses of patients based on the correct patient IDs
    # for patho in ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]:
    id_path = output_path
    result_path = evals_path
    for model in [
        "Llama-3.1-8B-Instruct"
        "Llama-3.1-70B-Instruct",
        "DeepSeek-R1-Distill-Llama-70B" 
    ]:
        run = f"_{model}_*_FULL_INFO_*results.pkl"
        assert "result" in run

        for patho in [
            "appendicitis", 
            "cholecystitis", 
            "diverticulitis", 
            "pancreatitis",
            "pericarditis",
            "pneumonia",
            "pulmonary embolism",
        ]:
            #load the diagnoses results path
            results_log_path = join(result_path, f"{patho}{run}")

            # Load the correct patient IDs
            with open(join(id_path, f"{patho}_PLI_N.pkl"), 'rb') as f:
                patients_ids = pickle.load(f)

            correct_diagnoses = []
            all_results = []
            # Load the diagnoses results of all patients
            for r in read_from_pickle_file(glob.glob(results_log_path)[0]):
                for k, v in r.items():
                    if k in patients_ids:
                        all_results.append(v)
            n = 3     
            for i in range(n):
                if len(all_results) >= 30:
                    correct_diagnoses = random.sample(all_results,30)
                else:
                    correct_diagnoses = all_results
                print(len(correct_diagnoses))
                
                # Save the resulting patient IDs to a new .pkl file
                with open(join(id_path, f"{patho}_{model}_PLI_N_INFO_CORRECT_30_{i}.pkl"), 'wb') as correct_f:
                    pickle.dump(correct_diagnoses, correct_f)


create_correct_ids()
create_correct_id_info()