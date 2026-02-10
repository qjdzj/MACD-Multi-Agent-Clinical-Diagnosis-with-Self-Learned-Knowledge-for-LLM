import os
import re
from os.path import join
import json
import random
from datetime import datetime
import time
import pickle
import fcntl

import numpy as np
import hydra
from omegaconf import DictConfig
from loguru import logger
import langchain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from thefuzz import process

from utils.nlp import calculate_num_tokens, truncate_text, create_lab_test_string
from dataset.utils import load_hadm_from_file
from utils.logging import append_to_pickle_file
from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from evaluators.cholecystitis_evaluator import CholecystitisEvaluator
from evaluators.diverticulitis_evaluator import DiverticulitisEvaluator
from evaluators.pancreatitis_evaluator import PancreatitisEvaluator
from evaluators.pneumonia_evaluator import PneumoniaEvaluator
from evaluators.pulmonary_embolism_evaluator import PulmonaryEmbolismEvaluator
from evaluators.pericarditis_evaluator import PericarditisEvaluator
from models.models import CustomLLM

from agents.prompts import (
    FULL_INFO_TEMPLATE_DIAGSUM,
    FULL_INFO_TEMPLATE_COT,
    FULL_INFO_TEMPLATE_COT_FINAL_DIAGNOSIS,
    FI_FEWSHOT_TEMPLATE_COPD,
    FI_FEWSHOT_TEMPLATE_PNEUMONIA,
    FI_FEWSHOT_TEMPLATE_APPENDICITIS,
    FI_FEWSHOT_TEMPLATE_CHOLECYSTITIS,
    SUMMARIZE_OBSERVATION_TEMPLATE,
    WRITE_DIAG_CRITERIA_TEMPLATE,
    FULL_INFO_TEMPLATE_DIAGSUM_WITH_PAST,
)
from agents import reference_llama70B
from agents import reference_deepseek70B
from agents import reference_llama8B
from utils.logging import read_from_pickle_file


gpt_tags = {
    "system_tag_start": "<|im_start|>system",
    "user_tag_start": "<|im_start|>user",
    "ai_tag_start": "<|im_start|>assistant",
    "system_tag_end": "<|im_end|>",
    "user_tag_end": "<|im_end|>",
    "ai_tag_end": "<|im_end|>",
}

STOP_WORDS = []


def load_all_past_diagnosis_results(pathology, result_dir):
    pattern = re.compile(rf"{re.escape(pathology)}_(.*?)_FULL_INFO_.*results\.pkl")
    files = [f for f in os.listdir(result_dir) if pattern.match(f)]
    files.sort()
    all_results = {}
    for idx, fname in enumerate(files):
        model_match = pattern.match(fname)
        model_name = model_match.group(1) if model_match else f"model{idx+1}"
        file_path = os.path.join(result_dir, fname)
        data = {}
        for d in read_from_pickle_file(file_path):
            data.update(d)
        # 保证key为int
        data_int_keys = {int(k): v for k, v in data.items()}
        all_results[fname] = data_int_keys
    return all_results


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
def run(args: DictConfig):

    print(args)

    # Build run_name consistent with run_full_info.py
    args.model_name = args.model_name.replace("/", "_")
    run_name = f"{args.pathology}_{args.model_name}_FULL_INFO"
    # Order
    if getattr(args, 'order', None):
        run_name += f"_{str(args.order).upper()}"
    else:
        run_name += "_H"
    # Criteria letters: use F when criteria enabled, else N
    if getattr(args, 'criteria', False):
        run_name += "_F"
    else:
        run_name += "_N"
    # Fewshot and other flags
    if getattr(args, 'fewshot', False):
        run_name += "_FEWSHOT"
    if getattr(args, 'include_ref_range', False):
        run_name += "_REFRANGE"
    if getattr(args, 'only_abnormal_labs', False):
        run_name += "_ONLYABNORMAL"
    if getattr(args, 'bin_lab_results', False):
        run_name += "_BIN"
    if getattr(args, 'bin_lab_results_abnormal', False):
        run_name += "_BINABNORMAL"
    if not getattr(args, 'summarize', True):
        run_name += "_NOSUMMARY"
    if not getattr(args, 'abbreviated', True):
        run_name += "_NOABBR"
    if getattr(args, 'self_consistency', False):
        run_name += "_SELFCONSISTENCY"
    # Template suffixes like in run_full_info
    if getattr(args, 'prompt_template', None) == "DIAGSUM_WITH_PAST":
        run_name += "_DIAGSUM_WITH_PAST"
    elif getattr(args, 'prompt_template', None) == "DIAGSUM":
        run_name += "_DIAGSUM"
    # Optional run description
    if getattr(args, 'run_descr', None):
        run_name += str(args.run_descr)

    run_dir = args.local_logging_dir
    os.makedirs(run_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set stop words
    global STOP_WORDS
    STOP_WORDS = args.stop_words

    tags = {
        "system_tag_start": args.system_tag_start,
        "user_tag_start": args.user_tag_start,
        "ai_tag_start": args.ai_tag_start,
        "system_tag_end": args.system_tag_end,
        "user_tag_end": args.user_tag_end,
        "ai_tag_end": args.ai_tag_end,
    }

    # Load desired model
    llm = CustomLLM(
        model_name=args.model_name,
        openai_api_key=args.openai_api_key,
        openai_api_base=args.openai_api_base,
        tags=tags,
        max_context_length=args.max_context_length,
        exllama=args.exllama,
        seed=args.seed,
        self_consistency=args.self_consistency,
        lora_path=getattr(args, "lora_path", None),
        use_lora=getattr(args, "use_lora", False)
    )
    llm.load_model(args.base_models)

    if args.prompt_template == "DIAGSUM":
        basic_prompt = FULL_INFO_TEMPLATE_DIAGSUM
    elif args.prompt_template == "DIAGSUM_WITH_PAST":
        basic_prompt = FULL_INFO_TEMPLATE_DIAGSUM_WITH_PAST
    elif args.prompt_template == "COT":
        basic_prompt = FULL_INFO_TEMPLATE_COT
        final_diagnosis_prompt = PromptTemplate(
            template=FULL_INFO_TEMPLATE_COT_FINAL_DIAGNOSIS,
            input_variables=["cot"],
            partial_variables={
                "system_tag_start": tags["system_tag_start"],
                "system_tag_end": tags["system_tag_end"],
                "user_tag_start": tags["user_tag_start"],
                "user_tag_end": tags["user_tag_end"],
                "ai_tag_start": tags["ai_tag_start"],
            },
        )
        cot_conclude_chain = LLMChain(llm=llm, prompt=final_diagnosis_prompt)
    else:
        raise NotImplementedError
    print(basic_prompt)

    # 根据不同模板设置输入变量
    if args.prompt_template == "DIAGSUM_WITH_PAST":
        prompt_input_vars = [
            "input",
            "fewshot_examples",
            "diagnostic_guidelines",
            "past_diagnosis_results",
        ]
    else:
        prompt_input_vars = [
            "input",
            "fewshot_examples",
            "diagnostic_guidelines",
        ]

    prompt = PromptTemplate(
        template=basic_prompt,
        input_variables=prompt_input_vars,
        partial_variables={
            "system_tag_start": tags["system_tag_start"],
            "system_tag_end": tags["system_tag_end"],
            "user_tag_start": tags["user_tag_start"],
            "user_tag_end": tags["user_tag_end"],
            "ai_tag_start": tags["ai_tag_start"],
        },
    )
    langchain.debug = True

    diagnose_chain = LLMChain(llm=llm, prompt=prompt)

    if args.summarize:
        summarize_prompt = PromptTemplate(
            template=SUMMARIZE_OBSERVATION_TEMPLATE,
            input_variables=["observation"],
            partial_variables={
                "system_tag_start": tags["system_tag_start"],
                "system_tag_end": tags["system_tag_end"],
                "user_tag_start": tags["user_tag_start"],
                "user_tag_end": tags["user_tag_end"],
                "ai_tag_start": tags["ai_tag_start"],
            },
        )
        length_summary_prompt = calculate_num_tokens(
            llm.tokenizer,
            [
                summarize_prompt.format(
                    observation="",
                    system_tag_start=tags["system_tag_start"],
                    system_tag_end=tags["system_tag_end"],
                    user_tag_start=tags["user_tag_start"],
                    user_tag_end=tags["user_tag_end"],
                    ai_tag_start=tags["ai_tag_start"],
                )
            ],
        )
        summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

    args.model_name = args.model_name.replace("/", "_")

    # Setup logfile and results path (match run_full_info.py naming)
    results_path = join(run_dir, f"{run_name}_results.pkl")
    log_path = join(run_dir, f"{run_name}.log")
    logger.add(log_path, enqueue=True, backtrace=True, diagnose=True)
    logger.info(args)

    # Load lab test mapping
    with open(args.lab_test_mapping_path, "rb") as f:
        lab_test_mapping_df = pickle.load(f)

    knowledge_mapping = {
        "Llama-3.1-8B-Instruct": {
            "Abdomen": reference_llama8B.abdomen_reference,
            "Chest": reference_llama8B.chest_reference
        },
        "Llama-3.1-70B-Instruct": {
            "Abdomen": reference_llama70B.abdomen_reference,
            "Chest": reference_llama70B.chest_reference
        },
        "DeepSeek-R1-Distill-Llama-70B": {
            "Abdomen": reference_deepseek70B.abdomen_reference,
            "Chest": reference_deepseek70B.chest_reference
        },
        "deepseek-v3-1-250821": {
            "Abdomen": reference_llama70B.abdomen_reference,
            "Chest": reference_llama70B.chest_reference
        },
        "qwen3-235b-a22b": {
            "Abdomen": reference_llama70B.abdomen_reference,
            "Chest": reference_llama70B.chest_reference
        },
        "gpt-5": {
            "Abdomen": reference_llama70B.abdomen_reference,
            "Chest": reference_llama70B.chest_reference
        },
    }

    pathology = args.pathology
    if pathology in ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]:
        region = "Abdomen"
    elif pathology in ["pneumonia", "pulmonary embolism", "pericarditis"]:
        region = "Chest"

    model_name = args.model_name
    guideline_text_for_this_run = knowledge_mapping.get(model_name, {}).get(region, "")

    diagnostic_guidelines = "Based on the provided diagnostic criteria, please analyze and diagnose the disease. The diagnostic criteria are offered as a reference; however, it is essential to consider the actual condition of the patient comprehensively. Please provide a diagnosis that aligns with the given information and the patient's specific situation.\n"
    if args.guideline:
        diagnostic_guidelines += guideline_text_for_this_run 

    if args.guideline and not guideline_text_for_this_run:
        logger.warning(f"Guideline is enabled, but no knowledge found for model '{model_name}' and region '{region}'!")

    if region == "Abdomen":
        diagnosis_examples = [
            FI_FEWSHOT_TEMPLATE_COPD,
            FI_FEWSHOT_TEMPLATE_PNEUMONIA,
        ]
    elif region == "Chest":
        diagnosis_examples = [
            FI_FEWSHOT_TEMPLATE_APPENDICITIS,
            FI_FEWSHOT_TEMPLATE_CHOLECYSTITIS,
        ]

    hadm_info_clean = load_hadm_from_file(
        f"{pathology}_hadm_info_first_diag", base_mimic=args.base_mimic
    )

    # 历史诊断结果：按需加载
    if hasattr(args, 'use_past_diagnosis') and args.use_past_diagnosis:
        past_diagnosis_dir = getattr(args, 'past_diagnosis_dir', None)
        if past_diagnosis_dir is None and hasattr(args, 'path') and hasattr(args.path, 'past_diagnosis_dir'):
            past_diagnosis_dir = args.path.past_diagnosis_dir
        elif past_diagnosis_dir is None:
            past_diagnosis_dir = './'
        
        print(f"Loading past diagnosis results from: {past_diagnosis_dir}")
        all_past_results = load_all_past_diagnosis_results(pathology, past_diagnosis_dir)

    # # Load list of specific IDs if provided
    patient_list = hadm_info_clean.keys()
    # Support selecting a subset of patients via a pickle file list of IDs
    if hasattr(args, 'patient_list_path') and args.patient_list_path:
        with open(args.patient_list_path, "rb") as f:
            patient_list = pickle.load(f)

    for _id in patient_list:
        logger.info(f"Processing patient: {_id}")
        # continue
        hadm = hadm_info_clean[_id]

        # Fewshot
        fewshot_examples = ""
        if args.fewshot:
            for example in diagnosis_examples:
                fewshot_examples += example.format(
                    user_tag_start=tags["user_tag_start"],
                    user_tag_end=tags["user_tag_end"],
                    ai_tag_start=tags["ai_tag_start"],
                    ai_tag_end=tags["ai_tag_end"],
                )

        # Eval
        evaluator = load_evaluator(
            args.pathology
        )  # Reload every time to ensure no state is carried over

        input = ""
        rad_reports = ""

        input = add_patient_history(input, hadm, args.abbreviated)

        char_to_func = {
            "p": "include_physical_examination",
            "l": "include_laboratory_tests",
            "i": "include_imaging",
        }

        # Read desired order from mapping and args.order and then execute and parse result
        for char in args.order:
            func = char_to_func[char]
            # Must be within for loop to use updated input variable
            mapping_functions = {
                "include_imaging": (add_rad_reports, [input, hadm, region]),
                "include_physical_examination": (
                    add_physical_examination,
                    [input, hadm, args.abbreviated],
                ),
                "include_laboratory_tests": (
                    add_laboratory_tests,
                    [input, hadm, evaluator, lab_test_mapping_df, args],
                ),
            }

            function, input_params = mapping_functions[func]
            result = function(*input_params)

            if isinstance(result, tuple):
                input, rad_reports = result
            else:
                input = result

        # Escape previous curly brackets to avoid issues with format later
        input = input.replace("{", "{{").replace("}", "}}")
        # This we want to leave for future formatting
        input = input.replace("{{rad_reports}}", "{rad_reports}")

        # 历史诊断：不参与长度控制以保持一致性
        if hasattr(args, 'use_past_diagnosis') and args.use_past_diagnosis:
            past_diagnosis_results_for_id = ""
            idx_doc = 1
            for model_name, model_results in all_past_results.items():
                if int(_id) in model_results:
                    text = model_results[int(_id)]
                    if isinstance(text, dict) and 'Diagnosis' in text:
                        text = text['Diagnosis']
                    past_diagnosis_results_for_id += f"<Result from previous doctor {idx_doc}>\n{str(text).strip()}\n"
                    idx_doc += 1

        input, fewshot_examples, rad_reports = control_context_length(
            input,
            basic_prompt,
            fewshot_examples,
            args.include_ref_range,
            rad_reports,
            llm,
            args,
            tags,
            _id,
            hadm_info_clean,
            args.summarize,
            summarize_chain,
            length_summary_prompt,
            diagnostic_guidelines,
            region,
            # past_diagnosis_results=past_diagnosis_results_for_id,
        )

        if args.prompt_template == "DIAGSUM_WITH_PAST":
            result = diagnose_chain.predict(
                input=input.format(rad_reports=rad_reports),
                fewshot_examples=fewshot_examples,
                diagnostic_guidelines=diagnostic_guidelines,
                past_diagnosis_results=past_diagnosis_results_for_id,
                stop=STOP_WORDS,
            )
        else:
            result = diagnose_chain.predict(
                input=input.format(rad_reports=rad_reports),
                fewshot_examples=fewshot_examples,
                diagnostic_guidelines=diagnostic_guidelines,
                stop=STOP_WORDS,
            )

        if args.prompt_template == "COT":
            # input = input.format(rad_reports=rad_reports)
            prompt_tokens_final_diag = calculate_num_tokens(
                llm.tokenizer,
                [
                    final_diagnosis_prompt.format(
                        cot=result,
                    )
                ],
            )
            if prompt_tokens_final_diag > args.max_context_length - 25:
                to_remove = prompt_tokens_final_diag - args.max_context_length + 25
                input = truncate_text(
                    llm.tokenizer,
                    # input,
                    -to_remove,
                )
                # input = input.replace("{", "{{").replace("}", "}}")

            result = cot_conclude_chain.predict(
                # input=input,
                cot=result,
                stop=STOP_WORDS,
            )

        append_to_pickle_file(results_path, {_id: result})


def write_diagnostic_criteria(pathology, diag_crit_writer):
    global STOP_WORDS
    diag_crit_prompt = PromptTemplate(
        template=WRITE_DIAG_CRITERIA_TEMPLATE,
        input_variables=["pathology"],
        partial_variables={
            "system_tag_start": gpt_tags["system_tag_start"],
            "system_tag_end": gpt_tags["system_tag_end"],
            "user_tag_start": gpt_tags["user_tag_start"],
            "user_tag_end": gpt_tags["user_tag_end"],
        },
    )
    diag_crit_chain = LLMChain(llm=diag_crit_writer, prompt=diag_crit_prompt)
    return diag_crit_chain.predict(
        pathology=pathology,
        stop=STOP_WORDS,
    )


def read_dict(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_SH)
            data = json.load(file)
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            return data
    return {}


def write_dict(file_path, data):
    with open(file_path, "w") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        json.dump(data, file)
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def add_patient_history(input, hadm, abbreviated=True):
    input += "@@@ PATIENT HISTORY @@@\n"
    # input += "PATIENT HISTORY\n"
    input += (
        hadm["Patient History"].strip()
        if abbreviated
        else hadm["Patient History Unabbreviated"].strip()
    )
    return input


def add_physical_examination(input, hadm, abbreviated=True):
    input += "\n\n@@@ PHYSICAL EXAMINATION @@@\n"
    # input += "\n\nPHYSICAL EXAMINATION\n"
    input += (
        hadm["Physical Examination"].strip()
        if abbreviated
        else hadm["Physical Examination Unabbreviated"].strip()
    )
    return input


def add_laboratory_tests(input, hadm, evaluator, lab_test_mapping_df, args):
    input += "\n\n@@@ LABORATORY RESULTS @@@\n"
    # input += "\n\nLABORATORY RESULTS\n"
    if args.include_ref_range:
        input += (
            "(<FLUID>) <TEST>: <RESULT> | REFERENCE RANGE (RR): [LOWER RR - UPPER RR]\n"
        )

    else:
        input += "(<FLUID>) <TEST>: <RESULT>\n"
    lab_tests_to_include = []

    for test_name in evaluator.required_lab_tests:
        lab_tests_to_include = (
            lab_tests_to_include + evaluator.required_lab_tests[test_name]
        )
    lab_tests_to_include = lab_tests_to_include + evaluator.neutral_lab_tests

    for test in lab_tests_to_include:
        if test in hadm["Laboratory Tests"].keys():
            input += create_lab_test_string(
                test,
                lab_test_mapping_df,
                hadm,
                include_ref_range=args.include_ref_range,
                bin_lab_results=args.bin_lab_results,
                bin_lab_results_abnormal=args.bin_lab_results_abnormal,
                only_abnormal_labs=args.only_abnormal_labs,
            )

    return input


def add_rad_reports(input, hadm, region):
    rad_reports = ""
    input += "\n\n@@@ IMAGING RESULTS @@@\n{rad_reports}"
    # input += "\n\nIMAGING RESULTS\n{rad_reports}"
    for rad in hadm["Radiology"]:
        if rad["Region"] == region:
            rad_reports += f"\n{rad['Modality']} {rad['Region']}\n"
            rad_reports += f"{rad['Report']}".strip()
    return input, rad_reports


def control_context_length(
    input,
    prompt_template,
    fewshot_examples,
    include_ref_range,
    rad_reports,
    llm,
    args,
    tags,
    _id,
    hadm_info_clean,
    summarize,
    summarize_chain,
    length_summarize_prompt,
    diagnostic_guidelines,
    region,
    past_diagnosis_results=None,
):
    global STOP_WORDS
    max_context_length = args.max_context_length
    final_diagnosis_tokens = 25
    total_length = calculate_num_tokens(
        llm.tokenizer,
        [
            prompt_template.format(
                input=input.format(rad_reports=rad_reports),
                system_tag_start=tags["system_tag_start"],
                system_tag_end=tags["system_tag_end"],
                user_tag_start=tags["user_tag_start"],
                user_tag_end=tags["user_tag_end"],
                ai_tag_start=tags["ai_tag_start"],
                fewshot_examples=fewshot_examples,
                diagnostic_guidelines=diagnostic_guidelines,
                past_diagnosis_results=past_diagnosis_results,
            ),
        ],
    )
    # Check if our prompt would exceed the max context length and lead to truncation
    if total_length > max_context_length:
        # If fewshot can try taking some of the examples away
        if args.fewshot:
            fewshot_examples = FI_FEWSHOT_TEMPLATE_COPD.format(
                user_tag_start=tags["user_tag_start"],
                user_tag_end=tags["user_tag_end"],
                ai_tag_start=tags["ai_tag_start"],
                ai_tag_end=tags["ai_tag_end"],
            )
            total_length = calculate_num_tokens(
                llm.tokenizer,
                [
                    prompt_template.format(
                        input=input.format(rad_reports=rad_reports),
                        system_tag_start=tags["system_tag_start"],
                        system_tag_end=tags["system_tag_end"],
                        user_tag_start=tags["user_tag_start"],
                        user_tag_end=tags["user_tag_end"],
                        ai_tag_start=tags["ai_tag_start"],
                        fewshot_examples=fewshot_examples,
                        diagnostic_guidelines=diagnostic_guidelines,
                        past_diagnosis_results=past_diagnosis_results,
                    ),
                ],
            )

            # If we're still too long, completely remove examples
            if total_length > max_context_length:
                # logger.warning(
                #    "Prompt is still too long. Removing all fewshot examples."
                # )
                fewshot_examples = ""
                total_length = calculate_num_tokens(
                    llm.tokenizer,
                    [
                        prompt_template.format(
                            input=input.format(rad_reports=rad_reports),
                            system_tag_start=tags["system_tag_start"],
                            system_tag_end=tags["system_tag_end"],
                            user_tag_start=tags["user_tag_start"],
                            user_tag_end=tags["user_tag_end"],
                            ai_tag_start=tags["ai_tag_start"],
                            fewshot_examples=fewshot_examples,
                            diagnostic_guidelines=diagnostic_guidelines,
                            past_diagnosis_results=past_diagnosis_results,
                        ),
                    ],
                )

        # Before we start summarizing rad we should take a look if we are already over the context length
        prompt_tokens_no_rad = calculate_num_tokens(
            llm.tokenizer,
            [
                prompt_template.format(
                    input=input.format(rad_reports=""),
                    system_tag_start=tags["system_tag_start"],
                    system_tag_end=tags["system_tag_end"],
                    user_tag_start=tags["user_tag_start"],
                    user_tag_end=tags["user_tag_end"],
                    ai_tag_start=tags["ai_tag_start"],
                    fewshot_examples=fewshot_examples,
                    diagnostic_guidelines=diagnostic_guidelines,
                    past_diagnosis_results=past_diagnosis_results,
                )
            ],
        )
        max_new_tokens = max_context_length - prompt_tokens_no_rad
        if max_new_tokens < final_diagnosis_tokens:
            # Even without rad, we are hitting our limit or close to. Need to remove rad and possibly truncate.
            rad_reports = ""
            # No rad and still too long, so truncate to max context length - final_diagnosis_tokens
            # logger.warning("Prompt is still too long. Truncating prompt.")
            to_truncate_length = (
                max_new_tokens
                - final_diagnosis_tokens  # Give a little wiggle room for the transitions and diagnosis
            )
            input = truncate_text(
                llm.tokenizer,
                input.format(rad_reports=rad_reports),
                to_truncate_length,
            )
            # Need to re-escape curly brackets
            input = input.replace("{", "{{").replace("}", "}}")
            return input, fewshot_examples, rad_reports

        # If we're still too long, then case is just longer than max context length and we need to summarize imaging results
        if total_length > max_context_length:
            if summarize:
                seen_modalities = set()
                rad_reports = ""
                # Go through original imaging and summarize
                for rad in hadm_info_clean[_id]["Radiology"]:
                    if (
                        rad["Region"] == region
                        and rad["Modality"] not in seen_modalities
                    ):

                        summary = summarize_chain.predict(
                            observation=rad["Report"], stop=STOP_WORDS
                        )
                        rad_reports += f"\n {summary}"
                        seen_modalities.add(rad["Modality"])
                total_length = calculate_num_tokens(
                    llm.tokenizer,
                    [
                        prompt_template.format(
                            input=input.format(rad_reports=rad_reports),
                            system_tag_start=tags["system_tag_start"],
                            system_tag_end=tags["system_tag_end"],
                            user_tag_start=tags["user_tag_start"],
                            user_tag_end=tags["user_tag_end"],
                            ai_tag_start=tags["ai_tag_start"],
                            fewshot_examples=fewshot_examples,
                            diagnostic_guidelines=diagnostic_guidelines,
                            past_diagnosis_results=past_diagnosis_results,
                        ),
                    ],
                )

            # If we are still too long, summarize the summary and enforce max characters
            if total_length > max_context_length:
                if summarize:
                    # Make sure that the length of rad_reports summary prompt is less than max_context_length

                    length_rad_report = calculate_num_tokens(
                        llm.tokenizer,
                        [rad_reports],
                    )
                    if length_summarize_prompt + length_rad_report > max_context_length:
                        rad_reports = truncate_text(
                            llm.tokenizer,
                            rad_reports,
                            max_context_length
                            - length_summarize_prompt
                            - max_new_tokens,
                        )
                    rad_reports = summarize_chain.predict(
                        observation=rad_reports,
                        stop=STOP_WORDS,
                    )
                rad_reports = truncate_text(
                    llm.tokenizer,
                    rad_reports,
                    max_new_tokens - final_diagnosis_tokens,
                )  # give a little wiggle room for the transitions and diagnosis

    return input, fewshot_examples, rad_reports


if __name__ == "__main__":
    run()