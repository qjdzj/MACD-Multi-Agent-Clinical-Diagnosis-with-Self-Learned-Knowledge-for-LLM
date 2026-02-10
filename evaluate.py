import os
import pickle
import glob

import hydra

from dataset.utils import load_hadm_from_file
from utils.logging import read_from_pickle_file
from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from evaluators.cholecystitis_evaluator import CholecystitisEvaluator
from evaluators.diverticulitis_evaluator import DiverticulitisEvaluator
from evaluators.pancreatitis_evaluator import PancreatitisEvaluator
from evaluators.pneumonia_evaluator import PneumoniaEvaluator
from evaluators.pulmonary_embolism_evaluator import PulmonaryEmbolismEvaluator
from evaluators.pericarditis_evaluator import PericarditisEvaluator


def calculate_average(evals, field):
    average = 0
    for patient in evals.keys():
        average += evals[patient]["scores"][field]
    average /= len(evals)
    return average, len(evals)


def load_evaluator(pathology):
    # Load desired evaluator
    if pathology == "appendicitis":
        evaluator = AppendicitisEvaluator()
    elif pathology == "cholecystitis":
        evaluator = CholecystitisEvaluator()
    elif pathology == "diverticulitis":
        evaluator = DiverticulitisEvaluator()
    elif pathology == "pancreatitis":
        evaluator = PancreatitisEvaluator()
    elif pathology == "pneumonia":
        evaluator = PneumoniaEvaluator()
    elif pathology == "pulmonary embolism":
        evaluator = PulmonaryEmbolismEvaluator()
    elif pathology == "pericarditis":
        evaluator = PericarditisEvaluator()
    else:
        raise NotImplementedError
    return evaluator


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def evaluate(args):
    data_dir = args.data_dir
    print(args.run_name)
    result_dir = os.path.join(args.save_dir, args.run_name)
    difficulty = "dr_eval"

    path_list = [
        "appendicitis",
        "cholecystitis",
        "diverticulitis",
        "pancreatitis",
        "pericarditis",
        "pneumonia",
        "pulmonary embolism",
    ]

    all_evals = {}
    all_results = {}
    for pathology in path_list:
        print(f"Evaluating {pathology}...")
        if pathology in [
            "appendicitis",
            "cholecystitis",
            "diverticulitis",
            "pancreatitis",
        ]:
            region = "abdomen"
        elif pathology in ["pericarditis", "pneumonia", "pulmonary embolism"]:
            region = "chest"
        id_difficulty = pickle.load(
            open(os.path.join(data_dir, region + "_id_difficulty.pkl"), "rb")
        )
        # Load Doctor Diagnosis
        hadm_info_clean = load_hadm_from_file(
            f"{pathology}_hadm_info_clean",
            data_dir,
        )
        all_evals[pathology] = {}
        all_results[pathology] = {}
        # Load LLM Prediction
        result_path = os.path.join(result_dir, f"{pathology}.pkl")

        results = []
        for r in read_from_pickle_file(glob.glob(result_path)[0]):
            results.append(r)
        results = {k: v for d in results for k, v in d.items()}

        for _id in id_difficulty[pathology][difficulty]:
            if _id not in results:
                print(f"Skipping {_id} | {glob.glob(result_path)[0]}")
                continue

            result = "Final Diagnosis: " + results[_id]

            evaluator = load_evaluator(pathology)

            eval = evaluator._evaluate_agent_trajectory(
                prediction=result,
                input="",
                reference=(
                    hadm_info_clean[_id]["Discharge Diagnosis"],
                    hadm_info_clean[_id]["ICD Diagnosis"],
                    hadm_info_clean[_id]["Procedures ICD9"],
                    hadm_info_clean[_id]["Procedures ICD10"],
                    hadm_info_clean[_id]["Procedures Discharge"],
                ),
                agent_trajectory=[],
                diagnosis_probabilities=None,
            )

            all_evals[pathology][_id] = eval
            all_results[pathology][_id] = result

    avg_scores = {}
    avg_samples = {}

    for field in ["Diagnosis", "Gracious Diagnosis"]:
        avg_scores[field] = {}
        avg_samples[field] = {}
        for pathology in path_list:
            avg, n = calculate_average(all_evals[pathology], field)

            avg_scores[field][pathology] = round(avg, 4)
            avg_samples[field][pathology] = n

    evaluation_result = {"all": all_evals, "average": avg_scores}
    print(evaluation_result["average"]["Diagnosis"])
    pickle.dump(
        evaluation_result,
        open(os.path.join(result_dir, "evaluation.pkl"), "wb"),
    )


if __name__ == "__main__":
    evaluate()
