#!/usr/bin/env python3
"""
Multi-Round Diagnosis System
Runs three models in parallel, evaluates consistency, and performs iterative diagnosis
"""

import os
import re
import json
import pickle
from os.path import join
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
import hydra
from omegaconf import DictConfig
from loguru import logger
import langchain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 启用 LangChain debug 模式，自动打印所有 prompt 和输出（与 run_full_info.py 保持一致）
langchain.debug = True

from models.models import CustomLLM
from agents.multi_round_orchestrator import MultiRoundDiagnosisOrchestrator
from agents.prompts import (
    FULL_INFO_TEMPLATE,
    FULL_INFO_TEMPLATE_DIAGSUM,
    FULL_INFO_TEMPLATE_DIAGSUM_WITH_PAST,
    FULL_INFO_TEMPLATE_DIAGSUM_WITH_DIFFERENCE,
    FULL_INFO_TEMPLATE_SECTION,
    FULL_INFO_TEMPLATE_NO_SYSTEM,
    FULL_INFO_TEMPLATE_NO_MEDICAL,
    FULL_INFO_TEMPLATE_SERIOUS,
    FULL_INFO_TEMPLATE_MINIMAL_SYSTEM,
    FULL_INFO_TEMPLATE_NO_USER,
    FULL_INFO_TEMPLATE_NO_SYSTEM_NO_USER,
    FULL_INFO_TEMPLATE_NO_PROMPT,
    FULL_INFO_TEMPLATE_NOFINAL,
    FULL_INFO_TEMPLATE_MAINDIAGNOSIS,
    FULL_INFO_TEMPLATE_PRIMARYDIAGNOSIS,
    FULL_INFO_TEMPLATE_ACUTE,
    FULL_INFO_TEMPLATE_COT,
    FULL_INFO_TEMPLATE_TOP3,
    FI_FEWSHOT_TEMPLATE_COPD,
    FI_FEWSHOT_TEMPLATE_PNEUMONIA,
    FI_FEWSHOT_TEMPLATE_COPD_RR,
    FI_FEWSHOT_TEMPLATE_PNEUMONIA_RR,
    SUMMARIZE_OBSERVATION_TEMPLATE
)
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
from agents import reference_llama8B, reference_llama70B, reference_deepseek70B
from agents import guidelines_profession

STOP_WORDS = []

# Knowledge mapping: 不同模型对应不同的diagnostic criteria
KNOWLEDGE_MAPPING = {
    "Llama-3.1-8B-Instruct": {
        "Abdomen": guidelines_profession.abdomen_guideline,
        "Chest": guidelines_profession.chest_guideline
    },
    "Llama-3.1-70B-Instruct": {
        "Abdomen": guidelines_profession.abdomen_guideline,
        "Chest": guidelines_profession.chest_guideline
    },
    "DeepSeek-R1-Distill-Llama-70B": {
        "Abdomen": guidelines_profession.abdomen_guideline,
        "Chest": guidelines_profession.chest_guideline
    }
}


def get_region_from_pathology(pathology: str) -> str:
    if pathology in ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]:
        return "Abdomen"
    elif pathology in ["pneumonia", "pulmonary embolism", "pericarditis"]:
        return "Chest"

def get_guideline_for_model(model_name: str, pathology: str, args: DictConfig) -> str:
    region = get_region_from_pathology(pathology)
    
    guideline_text = KNOWLEDGE_MAPPING.get(model_name, {}).get(region, "")
    
    diagnostic_guidelines = "Based on the provided diagnostic criteria, please analyze and diagnose the disease. The diagnostic criteria are offered as a reference; however, it is essential to consider the actual condition of the patient comprehensively. Please provide a diagnosis that aligns with the given information and the patient's specific situation.\n"
    
    if getattr(args, 'guideline', True):  # 默认启用guideline
        if guideline_text:
            diagnostic_guidelines += guideline_text + "\n\n"
        else:
            logger.warning(f"Guideline is enabled, but no knowledge found for model '{model_name}' and region '{region}'!")
    
    return diagnostic_guidelines


def load_evaluator(pathology):
    """Load the appropriate evaluator for the pathology"""
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


def add_patient_history(input_text, hadm, abbreviated=True):
    """Add patient history to input"""
    input_text += "@@@ PATIENT HISTORY @@@\n"
    input_text += (
        hadm["Patient History"].strip()
        if abbreviated
        else hadm["Patient History Unabbreviated"].strip()
    )
    return input_text


def add_physical_examination(input_text, hadm, abbreviated=True):
    """Add physical examination to input"""
    input_text += "\n\n@@@ PHYSICAL EXAMINATION @@@\n"
    input_text += (
        hadm["Physical Examination"].strip()
        if abbreviated
        else hadm["Physical Examination Unabbreviated"].strip()
    )
    return input_text


def add_laboratory_tests(input_text, hadm, evaluator, lab_test_mapping_df, args):
    """Add laboratory tests to input"""
    input_text += "\n\n@@@ LABORATORY RESULTS @@@\n"
    if args.include_ref_range:
        input_text += "(<FLUID>) <TEST>: <RESULT> | REFERENCE RANGE (RR): [LOWER RR - UPPER RR]\n"
    else:
        input_text += "(<FLUID>) <TEST>: <RESULT>\n"
    
    lab_tests_to_include = []
    for test_name in evaluator.required_lab_tests:
        lab_tests_to_include = lab_tests_to_include + evaluator.required_lab_tests[test_name]
    lab_tests_to_include = lab_tests_to_include + evaluator.neutral_lab_tests

    for test in lab_tests_to_include:
        if test in hadm["Laboratory Tests"].keys():
            test_string = create_lab_test_string(
                test_id=test,
                lab_test_mapping_df=lab_test_mapping_df,
                hadm_info=hadm,
                include_ref_range=args.include_ref_range
            )
            input_text += test_string
    
    return input_text


def add_rad_reports(input_text, hadm, pathology):
    """Add radiology reports to input"""
    rad_reports = ""
    input_text += "\n\n@@@ IMAGING RESULTS @@@\n{rad_reports}"
    target_region = get_region_from_pathology(pathology)
    for rad in hadm.get("Radiology", []):
        if rad.get("Region") == target_region:
            rad_reports += f"\n{rad.get('Modality', '')} {rad.get('Region', '')}\n"
            rad_reports += f"{rad.get('Report', '')}".strip()
    return input_text, rad_reports


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
    diagnostic_criteria,
    summarize,
    pathology,
    past_diagnosis_results="",
    difference="",
):
    global STOP_WORDS
    max_context_length = args.max_context_length
    final_diagnosis_tokens = 25
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

    # 准备完整的 format 参数字典，确保估算时包含所有可能的变量，防止后端“左截断”
    format_kwargs = {
        "input": input.format(rad_reports=rad_reports),
        "system_tag_start": tags["system_tag_start"],
        "system_tag_end": tags["system_tag_end"],
        "user_tag_start": tags["user_tag_start"],
        "user_tag_end": tags["user_tag_end"],
        "ai_tag_start": tags["ai_tag_start"],
        "fewshot_examples": fewshot_examples,
    }
    
    # 动态注入多轮变量（适配模板中的占位符）
    if "diagnostic_criteria" in prompt_template.input_variables:
        format_kwargs["diagnostic_criteria"] = diagnostic_criteria
    if "past_diagnosis_results" in prompt_template.input_variables:
        format_kwargs["past_diagnosis_results"] = past_diagnosis_results
    if "difference" in prompt_template.input_variables:
        format_kwargs["difference"] = difference

    prompt_tokens = calculate_num_tokens(
        llm.tokenizer,
        [
            prompt_template.format(**format_kwargs),
        ],
    )
    # Check if our prompt would exceed the max context length and lead to truncation
    if prompt_tokens > max_context_length:
        # If fewshot can try taking some of the examples away
        if args.fewshot:
            # logger.warning(
            #    f"Patient {_id} has too long prompt. Attempting to remedy by removing a fewshot example."
            # )
            # Only include shorter sample i.e. COPD
            if include_ref_range:
                fewshot_examples = FI_FEWSHOT_TEMPLATE_COPD_RR.format(
                    user_tag_start=tags["user_tag_start"],
                    user_tag_end=tags["user_tag_end"],
                    ai_tag_start=tags["ai_tag_start"],
                    ai_tag_end=tags["ai_tag_end"],
                )
            else:
                fewshot_examples = FI_FEWSHOT_TEMPLATE_COPD.format(
                    user_tag_start=tags["user_tag_start"],
                    user_tag_end=tags["user_tag_end"],
                    ai_tag_start=tags["ai_tag_start"],
                    ai_tag_end=tags["ai_tag_end"],
                )
            
            format_kwargs["fewshot_examples"] = fewshot_examples
            prompt_tokens = calculate_num_tokens(
                llm.tokenizer,
                [
                    prompt_template.format(**format_kwargs),
                ],
            )

            # If we're still too long, completely remove examples
            if prompt_tokens > max_context_length:
                # logger.warning(
                #    "Prompt is still too long. Removing all fewshot examples."
                # )
                fewshot_examples = ""
                format_kwargs["fewshot_examples"] = fewshot_examples
                prompt_tokens = calculate_num_tokens(
                    llm.tokenizer,
                    [
                        prompt_template.format(**format_kwargs),
                    ],
                )

        # Before we start summarizing rad we should take a look if we are already over the context length
        format_kwargs_no_rad = format_kwargs.copy()
        format_kwargs_no_rad["input"] = input.format(rad_reports="")
        prompt_tokens_no_rad = calculate_num_tokens(
            llm.tokenizer,
            [
                prompt_template.format(**format_kwargs_no_rad)
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
        if prompt_tokens > max_context_length:
            if summarize:
                seen_modalities = set()
                rad_reports = ""
                # 根据疾病类型动态确定影像报告区域，解决单 Agent 的硬编码问题
                target_region = get_region_from_pathology(pathology)
                # Go through original imaging and summarize
                for rad in hadm_info_clean[_id]["Radiology"]:
                    if (
                        rad.get("Region") == target_region
                        and rad.get("Modality") not in seen_modalities
                    ):
                        summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)
                        summary = summarize_chain.predict(
                            observation=rad.get("Report", ""), stop=STOP_WORDS
                        )
                        rad_reports += f"\n {summary}"
                        seen_modalities.add(rad.get("Modality"))
                
                format_kwargs["input"] = input.format(rad_reports=rad_reports)
                prompt_tokens = calculate_num_tokens(
                    llm.tokenizer,
                    [
                        prompt_template.format(**format_kwargs),
                    ],
                )

            # If we are still too long, summarize the summary and enforce max characters
            if prompt_tokens > max_context_length:
                if summarize:
                    summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)
                    # Make sure that the length of rad_reports summary prompt is less than max_context_length
                    prompt_tokens_summary = calculate_num_tokens(
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
                    prompt_tokens_rad = calculate_num_tokens(
                        llm.tokenizer,
                        [rad_reports],
                    )
                    if prompt_tokens_summary + prompt_tokens_rad > max_context_length:
                        rad_reports = truncate_text(
                            llm.tokenizer,
                            rad_reports,
                            max_context_length - prompt_tokens_summary - max_new_tokens,
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



class MultiRoundDiagnosisRunner:
    """
    多轮诊断系统的运行器
    """
    
    def __init__(
        self,
        pathology: str,
        model_names: List[str],
        model_configs: Dict,
        biobert_model_path: str,
        difference_json_path: str,
        prompt_template_name: str,
        max_rounds: int,
        consistency_threshold: float,
        output_base_dir: str,
        enable_differentiation_analysis: bool = True,
        judge_agent_config: Optional[Dict] = None
    ):
        """
        初始化多轮诊断运行器
        
        Args:
            pathology: 疾病类型
            model_names: 模型名称列表
            model_configs: 模型配置字典 {model_name: config}
            biobert_model_path: BioBERT模型路径
            difference_json_path: difference.json文件路径
            prompt_template_name: prompt模板名称（DIAGSUM、VANILLA等）
            max_rounds: 最大诊断轮次
            consistency_threshold: 一致性阈值
            output_base_dir: 输出基础目录
            enable_differentiation_analysis: 是否启用差异分析
            judge_agent_config: Judge Agent API配置（用于生成差异分析）
        """
        self.pathology = pathology
        self.model_names = model_names
        self.model_configs = model_configs
        self.max_rounds = max_rounds
        self.consistency_threshold = consistency_threshold
        
        # 创建输出目录
        self.output_dir = os.path.join(output_base_dir, pathology)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化编排器
        self.orchestrator = MultiRoundDiagnosisOrchestrator(
            model_names=model_names,
            biobert_model_path=biobert_model_path,
            difference_json_path=difference_json_path,
            max_rounds=max_rounds,
            consistency_threshold=consistency_threshold,
            output_dir=self.output_dir,
            enable_differentiation=enable_differentiation_analysis,
            judge_agent_config=judge_agent_config
        )
        
        # 模型缓存
        self.llm_cache = {}
        self.chain_cache = {}  # 缓存LLMChain
        
        # Prompt template - 根据名称选择相应的模板
        self.prompt_template_name = prompt_template_name
        self.prompt_template = self._select_prompt_template(prompt_template_name)
        
        # 日志配置
        log_path = os.path.join(self.output_dir, f"multi_round_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logger.add(log_path, enqueue=True, backtrace=True, diagnose=True)
        
        # 模型预加载标志
        self._models_preloaded = False
        # 线程锁，保护共享资源
        self._output_lock = threading.Lock()
    
    def preload_all_models(self):
        """
        预加载所有模型到各自的GPU上
        只在第一个case处理前调用一次
        """
        if self._models_preloaded:
            logger.info("Models already preloaded, skipping...")
            return
        
        logger.info(f"开始预加载 {len(self.model_names)} 个模型...")
        print(f"\n{'='*80}")
        print(f"预加载模型到GPU...")
        print(f"{'='*80}")
        
        for i, model_name in enumerate(self.model_names, 1):
            config = self.model_configs.get(model_name, {})
            gpu_id = config.get('gpu_id', 'auto')
            
            print(f"\n[{i}/{len(self.model_names)}] 加载 {model_name} 到 GPU {gpu_id}...")
            
            # 调用_get_or_create_llm会触发模型加载并缓存
            self._get_or_create_llm(model_name)
            
            print(f"✓ {model_name} 加载完成")
        
        self._models_preloaded = True
        logger.info("所有模型预加载完成")
        print(f"\n{'='*80}")
        print(f"✓ 所有模型加载完成，开始诊断...")
        print(f"{'='*80}\n")
    
    def _select_prompt_template(self, template_name: str, use_past: bool = False, use_diff: bool = False) -> str:
        """
        根据模板名称选择相应的prompt模板，支持多轮对话变体
        
        Args:
            template_name: 模板名称（DIAGSUM、VANILLA等）
            use_past: 是否包含历史诊断
            use_diff: 是否包含差异分析
        
        Returns:
            选中的prompt模板字符串
        """
        if template_name == "DIAGSUM":
            if use_past and use_diff:
                return FULL_INFO_TEMPLATE_DIAGSUM_WITH_DIFFERENCE
            elif use_past:
                return FULL_INFO_TEMPLATE_DIAGSUM_WITH_PAST
            return FULL_INFO_TEMPLATE_DIAGSUM
        elif template_name == "VANILLA":
            return FULL_INFO_TEMPLATE
        elif template_name == "NOSYSTEM":
            return FULL_INFO_TEMPLATE_NO_SYSTEM
        elif template_name == "NOUSER":
            return FULL_INFO_TEMPLATE_NO_USER
        elif template_name == "NOSYSTEMNOUSER":
            return FULL_INFO_TEMPLATE_NO_SYSTEM_NO_USER
        elif template_name == "NOMEDICAL":
            return FULL_INFO_TEMPLATE_NO_MEDICAL
        elif template_name == "SERIOUS":
            return FULL_INFO_TEMPLATE_SERIOUS
        elif template_name == "MINIMALSYSTEM":
            return FULL_INFO_TEMPLATE_MINIMAL_SYSTEM
        elif template_name == "NOPROMPT":
            return FULL_INFO_TEMPLATE_NO_PROMPT
        elif template_name == "NOFINAL":
            return FULL_INFO_TEMPLATE_NOFINAL
        elif template_name == "MAINDIAGNOSIS":
            return FULL_INFO_TEMPLATE_MAINDIAGNOSIS
        elif template_name == "PRIMARYDIAGNOSIS":
            return FULL_INFO_TEMPLATE_PRIMARYDIAGNOSIS
        elif template_name == "ACUTE":
            return FULL_INFO_TEMPLATE_ACUTE
        elif template_name == "SECTION":
            return FULL_INFO_TEMPLATE_SECTION
        elif template_name == "TOP3":
            return FULL_INFO_TEMPLATE_TOP3
        elif template_name == "COT":
            return FULL_INFO_TEMPLATE_COT
        else:
            raise NotImplementedError(f"Unknown prompt template: {template_name}")
    
    def _get_or_create_llm(self, model_name: str) -> CustomLLM:
        """
        获取或创建LLM实例（带缓存）
        """
        if model_name not in self.llm_cache:
            config = self.model_configs.get(model_name, {})
            
            llm = CustomLLM(
                model_name=config.get('model_name', model_name),
                openai_api_key=config.get('openai_api_key'),
                tags=config.get('tags'),
                max_context_length=config.get('max_context_length'),
                gpu_id=config.get('gpu_id'),
                exllama=config.get('exllama'),
                seed=config.get('seed'),
                model=None,
                generator=None,
                tokenizer=None,
                pipeline=None
            )
        
            llm.load_model(config.get('base_models'))
            self.llm_cache[model_name] = llm
            
            print(f"✓ Model {model_name} loaded successfully on GPU {config.get('gpu_id', 'auto')}")
            logger.info(f"模型 {model_name} 加载完成")
        
        return self.llm_cache[model_name]
    
    def _get_or_create_chain(self, model_name: str, tags: Dict, use_past_diagnosis: bool = False, use_difference: bool = False) -> LLMChain:
        """
        获取或创建LLMChain（带缓存）
        
        Args:
            model_name: 模型名称
            tags: 模型标签
            use_past_diagnosis: 是否使用past_diagnosis_results
            use_difference: 是否使用difference占位符
        """
        cache_key = f"{model_name}_past={use_past_diagnosis}_diff={use_difference}"
        
        if cache_key not in self.chain_cache:
            llm = self._get_or_create_llm(model_name)
            
            # 根据多轮状态动态选择模板，对齐单 Agent 的适配逻辑
            template_str = self._select_prompt_template(
                template_name=self.prompt_template_name,
                use_past=use_past_diagnosis,
                use_diff=use_difference
            )
            
            # 根据参数选择input_variables
            input_vars = ["input", "fewshot_examples", "diagnostic_criteria"]
            if use_past_diagnosis:
                input_vars.append("past_diagnosis_results")
            if use_difference:
                input_vars.append("difference")
            
            prompt = PromptTemplate(
                template=template_str,
                input_variables=input_vars,
                partial_variables={
                    "system_tag_start": tags["system_tag_start"],
                    "system_tag_end": tags["system_tag_end"],
                    "user_tag_start": tags["user_tag_start"],
                    "user_tag_end": tags["user_tag_end"],
                    "ai_tag_start": tags["ai_tag_start"],
                },
            )
            
            chain = LLMChain(llm=llm, prompt=prompt)
            self.chain_cache[cache_key] = chain
        
        return self.chain_cache[cache_key]
    
    def diagnosis_function(
        self,
        case_id: str,
        patient_data: Dict,
        model_name: str,
        past_diagnosis_results: str = "",
        difference_analysis: str = ""
    ) -> str:
        """
        诊断函数 - 实现严格的环境隔离与状态重置，确保与单模型独立运行一致
        """
        import random
        import torch
        import numpy as np
        global STOP_WORDS
        
        # 1. 强制环境隔离：清理缓存并准备环境
        config = self.model_configs.get(model_name, {})
        gpu_id = config.get('gpu_id', None)
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()

        # 2. 强制随机性隔离：重置所有种子，消除“上一轮”或“前一个模型”的影响
        seed = config.get('seed')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 3. 隔离处理：获取当前模型特定的 LLM 和 Tags
        llm = self._get_or_create_llm(model_name)
        tags = config.get('tags', {})
        
        # 判断多轮状态
        has_difference = bool(difference_analysis)
        use_past = bool(past_diagnosis_results)
        
        # 4. 隔离格式化：使用当前模型自己的标签处理 Fewshot，防止标签污染
        raw_fewshot = patient_data.get('raw_fewshot_template', '')
        fewshot_examples = ""
        if raw_fewshot:
            fewshot_examples = raw_fewshot.format(
                user_tag_start=tags["user_tag_start"],
                user_tag_end=tags["user_tag_end"],
                ai_tag_start=tags["ai_tag_start"],
                ai_tag_end=tags["ai_tag_end"],
            )

        # 5. 动态选择 Chain：状态管理下沉到 _get_or_create_chain，消除临时修改属性风险
        has_difference = bool(difference_analysis)
        use_past = bool(past_diagnosis_results)
        chain = self._get_or_create_chain(model_name, tags, use_past_diagnosis=use_past, use_difference=has_difference)

        # 6. 适配控制逻辑：传入所有影响长度的变量，并修正 hadm 数据传递
        model_specific_criteria = patient_data.get('model_specific_criteria', {}).get(model_name, '')
        
        final_input_text, fewshot_examples, rad_reports = control_context_length(
            patient_data['raw_input_text'],
            chain.prompt,
            fewshot_examples,
            patient_data['include_ref_range'],
            patient_data['raw_rad_reports'],
            llm,
            patient_data['args'],
            tags,
            case_id,
            {case_id: patient_data.get('hadm_info', {})}, # 确保摘要逻辑能拿到 Radiology 列表
            model_specific_criteria,
            patient_data['args'].summarize,
            self.pathology,
            past_diagnosis_results=past_diagnosis_results,
            difference=difference_analysis
        )

        # 7. 准备最终输入：使用当前模型的截断结果
        final_input = final_input_text.format(rad_reports=rad_reports)
        model_specific_criteria = patient_data.get('model_specific_criteria', {}).get(model_name, '')
        
        # 8. 执行推理（LangChain debug 模式会自动打印完整的 prompt 和输出）
        try:
            # 线程安全：打印输出时加锁
            with self._output_lock:
                logger.info(f"模型 {model_name} 开始推理 - 病例 {case_id}")
            
            if use_past:
                if has_difference:
                    result = chain.predict(
                        input=final_input,
                        fewshot_examples=fewshot_examples,
                        diagnostic_criteria=model_specific_criteria,
                        past_diagnosis_results=past_diagnosis_results,
                        difference=difference_analysis,
                        stop=STOP_WORDS,
                    )
                else:
                    result = chain.predict(
                        input=final_input,
                        fewshot_examples=fewshot_examples,
                        diagnostic_criteria=model_specific_criteria,
                        past_diagnosis_results=past_diagnosis_results,
                        stop=STOP_WORDS,
                    )
            else:
                result = chain.predict(
                    input=final_input,
                    fewshot_examples=fewshot_examples,
                    diagnostic_criteria=model_specific_criteria,
                    stop=STOP_WORDS,
                )
            
            # 强制等待 GPU 完成所有操作，防止 BitsAndBytes 量化端序竞争
            torch.cuda.synchronize()
            
            with self._output_lock:
                logger.info(f"模型 {model_name} 推理完成 - 病例 {case_id}")
            
            return result
        except Exception as e:
            with self._output_lock:
                logger.error(f"模型 {model_name} 诊断失败 - 病例 {case_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
    
    def prepare_patient_data(
        self,
        hadm: Dict,
        args: DictConfig,
        lab_test_mapping_df,
        evaluator,
        case_id: str,
        hadm_info_clean: Dict
    ) -> Dict:
        """
        准备患者原始数据 - 剥离所有与模型相关的处理，保证原始性，延迟到各模型内部执行截断和格式化
        """
        input_text = ""
        rad_reports = ""
        
        # 1. 构建原始文本
        input_text = add_patient_history(input_text, hadm, args.abbreviated)
        
        char_to_func = {
            "p": "include_physical_examination",
            "l": "include_laboratory_tests",
            "i": "include_imaging",
        }
        
        for char in args.order:
            func = char_to_func[char]
            mapping_functions = {
                "include_imaging": (add_rad_reports, [input_text, hadm, self.pathology]),
                "include_physical_examination": (
                    add_physical_examination,
                    [input_text, hadm, args.abbreviated],
                ),
                "include_laboratory_tests": (
                    add_laboratory_tests,
                    [input_text, hadm, evaluator, lab_test_mapping_df, args],
                ),
            }
            
            function, input_params = mapping_functions[func]
            result = function(*input_params)
            
            if isinstance(result, tuple):
                input_text, rad_reports = result
            else:
                input_text = result
        
        # 基础转义
        input_text = input_text.replace("{", "{{").replace("}", "}}")
        input_text = input_text.replace("{{rad_reports}}", "{rad_reports}")
        
        # 2. 准备 Fewshot 模板（对齐 run_full_info.py 的病种选择：统一使用 COPD/PNEUMONIA）
        raw_fewshot_template = ""
        if args.fewshot:
            if args.include_ref_range:
                raw_fewshot_template += FI_FEWSHOT_TEMPLATE_COPD_RR + FI_FEWSHOT_TEMPLATE_PNEUMONIA_RR
            else:
                raw_fewshot_template += FI_FEWSHOT_TEMPLATE_COPD + FI_FEWSHOT_TEMPLATE_PNEUMONIA
        
        # 3. 准备模型特定指南 (不再使用通用的 diagnostic_criteria)
        model_specific_criteria = {}
        for model_name in self.model_names:
            model_specific_criteria[model_name] = get_guideline_for_model(
                model_name=model_name,
                pathology=self.pathology,
                args=args
            )
        
        # 返回原始包裹，不进行 control_context_length
        return {
            'raw_input_text': input_text,
            'raw_rad_reports': rad_reports,
            'raw_fewshot_template': raw_fewshot_template,
            'model_specific_criteria': model_specific_criteria,
            'include_ref_range': args.include_ref_range,
            'args': args,
            'hadm_info': hadm # 适配：保留原始数据引用
        }
    
    def run(self, patient_list: List, hadm_info_clean: Dict, args: DictConfig, lab_test_mapping_df):
        """
        运行多轮诊断系统
        
        Args:
            patient_list: 患者ID列表
            hadm_info_clean: 患者信息字典
            args: 配置参数
            lab_test_mapping_df: 实验室测试映射
        """
        global STOP_WORDS
        STOP_WORDS = args.stop_words
        
        # 在处理第一个case前，预加载所有模型
        self.preload_all_models()
        
        # 加载评估器
        evaluator = load_evaluator(self.pathology)
        
        # 存储所有结果
        all_case_results = []
        
        for case_id in patient_list:
            logger.info(f"Processing case: {case_id}")
            hadm = hadm_info_clean[case_id]
            
            # 准备患者数据
            patient_data = self.prepare_patient_data(
                hadm=hadm,
                args=args,
                lab_test_mapping_df=lab_test_mapping_df,
                evaluator=evaluator,
                case_id=case_id,
                hadm_info_clean=hadm_info_clean
            )
            
            # 处理这个病例的多轮诊断
            case_result = self.orchestrator.process_case(
                case_id=str(case_id),
                pathology=self.pathology,
                patient_data=patient_data,
                diagnosis_function=self.diagnosis_function
            )
            
            all_case_results.append(case_result)
        
        # 保存结果
        self._save_results(all_case_results, args)
        
        return all_case_results
    
    def _save_results(self, case_results, args):
        """
        保存所有诊断结果到文件
        
        Args:
            case_results: 所有病例的诊断结果
            args: 配置参数
        """
        # 保存为JSON
        results_data = []
        for case_result in case_results:
            case_dict = {
                'case_id': str(case_result.case_id),
                'pathology': case_result.pathology,
                'final_status': case_result.final_status.value,
                'final_diagnosis': case_result.final_diagnosis,
                'max_rounds_completed': case_result.max_rounds_completed,
                'rounds': [round_diag.to_dict() for round_diag in case_result.rounds]
            }
            results_data.append(case_dict)
        
        results_json_path = os.path.join(self.output_dir, "multi_round_results.json")
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {results_json_path}")


@hydra.main(config_path="./configs", config_name="multi_round_config", version_base=None)
def main(args: DictConfig):
    """Main entry point for multi-round diagnosis system"""
    
    # 加载实验室测试映射
    with open(args.lab_test_mapping_path, "rb") as f:
        lab_test_mapping_df = pickle.load(f)
    
    # 加载患者数据
    hadm_info_clean = load_hadm_from_file(
        f"{args.pathology}_hadm_info_first_diag", base_mimic=args.base_mimic
    )
    
    # 获取患者列表
    patient_list = list(hadm_info_clean.keys())
    if args.patient_list_path:
        with open(args.patient_list_path, "rb") as f:
            patient_list = pickle.load(f)
    
    # 处理指定的开始患者
    if args.first_patient:
        first_found = False
        filtered_list = []
        for patient_id in patient_list:
            if patient_id == args.first_patient:
                first_found = True
            if first_found:
                filtered_list.append(patient_id)
        patient_list = filtered_list
    
    # 准备模型配置
    diagnosis_models_config = args.diagnosis_models
    model_names = [m['model_name'] for m in diagnosis_models_config]
    
    model_configs = {}
    for model_config in diagnosis_models_config:
        model_name = model_config['model_name']
        model_configs[model_name] = {
            'model_name': model_name,
            'gpu_id': model_config.get('gpu_id', None),
            'openai_api_key': model_config.get('openai_api_key', None),
            'tags': {
                'system_tag_start': model_config.get('system_tag_start', ''),
                'system_tag_end': model_config.get('system_tag_end', ''),
                'user_tag_start': model_config.get('user_tag_start', ''),
                'user_tag_end': model_config.get('user_tag_end', ''),
                'ai_tag_start': model_config.get('ai_tag_start', ''),
                'ai_tag_end': model_config.get('ai_tag_end', ''), # 补全 ai_tag_end
            },
            'max_context_length': model_config.get('max_context_length'),
            'exllama': model_config.get('exllama'),
            'seed': getattr(args, 'seed'),
            'base_models': args.base_models
        }
    
    # 创建运行器 - 指定prompt模板为DIAGSUM
    # 准备judge_agent_config
    judge_agent_config = None
    if hasattr(args, 'judge_agent') and args.judge_agent:
        judge_agent_config = {
            'api_base': args.judge_agent.get('api_base', ''),
            'api_key': args.judge_agent.get('api_key', ''),
            'model_name': args.judge_agent.get('model_name', 'gpt-3.5-turbo'),
            'temperature': args.judge_agent.get('temperature', 0.7)
        }
    
    runner = MultiRoundDiagnosisRunner(
        pathology=args.pathology,
        model_names=model_names,
        model_configs=model_configs,
        biobert_model_path=args.biobert_model_path,
        difference_json_path=args.difference_json_path,
        prompt_template_name=args.prompt_template,  # 从配置文件读取
        max_rounds=args.max_rounds,  # 直接使用配置值
        consistency_threshold=args.consistency_threshold,  # 直接使用配置值
        output_base_dir=args.local_logging_dir,
        enable_differentiation_analysis=args.enable_differentiation_analysis,
        judge_agent_config=judge_agent_config
    )
    
    # 运行诊断
    results = runner.run(
        patient_list=patient_list,
        hadm_info_clean=hadm_info_clean,
        args=args,
        lab_test_mapping_df=lab_test_mapping_df
    )
    
    print(f"\nProcessed {len(results)} cases")
    print(f"Results saved to {runner.output_dir}")


if __name__ == "__main__":
    main()
