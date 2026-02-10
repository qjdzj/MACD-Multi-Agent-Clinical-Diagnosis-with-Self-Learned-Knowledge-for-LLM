"""
Multi-round Diagnosis Orchestrator

Core module: manages the complete multi-round diagnosis process for individual cases
1. Round 1: Three models diagnose in parallel → Judge evaluates consistency
2. Consistent → Record result, process next case
   Inconsistent → Generate differentiation analysis → Enter next round
3. Repeat until consistent or maximum rounds reached
"""

import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.diagnosis_utils import (
    DiagnosisComparator,
    extract_diagnosis_from_text,
    map_pathology_name,
    PATHOLOGY_MAPPING_RULES
)
from agents.disease_differentiator import MultiModelDifferenceAnalyzer
from agents.judge_agent_prompts import get_judge_prompt


class DiagnosisStatus(Enum):
    """Diagnosis status enumeration"""
    CONSISTENT = "✅ Consistent"
    INCONSISTENT = "❌ Inconsistent"
    UNKNOWN = "❓ Unknown"


@dataclass
class RoundDiagnosis:
    """Single round diagnosis result"""
    round_num: int
    case_id: str
    model_diagnoses: Dict[str, str]  # {model_name: diagnosis}
    similarity_scores: Dict[str, float]  # {M1-M2: score, ...}
    consistency_status: DiagnosisStatus
    differentiations: str = ""  # 差异分析内容
    api_generated_differences: Optional[Dict[str, str]] = None  # API生成的差异内容 {pair_key: difference_text}
    timestamp: str = ""
    
    def to_dict(self):
        """Convert to dictionary format"""
        result = asdict(self)
        result['consistency_status'] = self.consistency_status.value
        return result


@dataclass
class CaseMultiRoundResult:
    """Complete multi-round diagnosis result for a case"""
    case_id: str
    pathology: str
    rounds: List[RoundDiagnosis]
    final_status: DiagnosisStatus
    final_diagnosis: str = ""
    max_rounds_completed: int = 0
    
    def get_current_round(self) -> Optional[RoundDiagnosis]:
        """Get current round diagnosis result"""
        if self.rounds:
            return self.rounds[-1]
        return None
    
    def add_round(self, round_diagnosis: RoundDiagnosis):
        """Add new round diagnosis"""
        self.rounds.append(round_diagnosis)
        self.max_rounds_completed = round_diagnosis.round_num


class MultiRoundDiagnosisOrchestrator:
    """
    Multi-round Diagnosis Orchestrator - Complete processing flow for individual cases
    """
    
    def __init__(
        self,
        model_names: List[str],
        biobert_model_path: str,
        difference_json_path: str,
        max_rounds: int,
        consistency_threshold: float,
        output_dir: str,
        enable_differentiation: bool = True,
        judge_agent_config: Optional[Dict] = None
    ):
        """
        初始化多轮诊断编排器
        
        Args:
            model_names: 模型名称列表
            biobert_model_path: BioBERT模型路径
            difference_json_path: 差异分析JSON文件路径
            max_rounds: 最大诊断轮次
            consistency_threshold: 一致性阈值
            output_dir: 输出目录
            enable_differentiation: 是否启用差异分析 (True: 完整模式, False: 简洁模式)
            judge_agent_config: Judge Agent API配置（用于生成差异分析）
        """
        self.model_names = model_names
        self.max_rounds = max_rounds
        self.consistency_threshold = consistency_threshold
        self.output_dir = output_dir
        self.enable_differentiation = enable_differentiation
        self.judge_agent_config = judge_agent_config or {}
        
        # Initialize components
        self.comparator = DiagnosisComparator(biobert_model_path, consistency_threshold)

        self.difference_analyzer = MultiModelDifferenceAnalyzer(difference_json_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load target pathology list
        self.target_pathologies = list(PATHOLOGY_MAPPING_RULES.keys())
        
        # Create model index mapping (for model anonymization)
        self.model_index_map = {
            "Llama-3.1-70B-Instruct": 1,
            "DeepSeek-R1-Distill-Llama-70B": 2,
            "Llama-3.1-8B-Instruct": 3
        }
    
    def process_case(
        self,
        case_id: str,
        pathology: str,
        patient_data: Dict,
        diagnosis_function: Callable
    ) -> CaseMultiRoundResult:
        """
        处理单个病例的完整多轮诊断流程
        
        Args:
            case_id: 病例ID
            pathology: 疾病类型
            patient_data: 患者数据
            diagnosis_function: 诊断函数，签名: (case_id, patient_data, model_name, context) -> diagnosis_text
        
        Returns:
            CaseMultiRoundResult: 病例的完整诊断结果
        """
        print(f"\n{'='*80}")
        print(f"Processing case: {case_id} | Disease: {pathology}")
        print(f"{'='*80}")
        
        # Initialize case result
        case_result = CaseMultiRoundResult(
            case_id=case_id,
            pathology=pathology,
            rounds=[],
            final_status=DiagnosisStatus.UNKNOWN
        )
        
        # Multi-round diagnosis loop
        for round_num in range(1, self.max_rounds + 1):
            print(f"\n[Round {round_num} Diagnosis]")
            print("-" * 80)
            
            # Get previous round diagnosis result (for building differentiation context for this round)
            previous_round = case_result.get_current_round()
            
            # Step 1: Execute three model diagnoses
            raw_diagnoses = self._run_three_models_diagnosis(
                case_id=case_id,
                patient_data=patient_data,
                diagnosis_function=diagnosis_function,
                previous_round=previous_round
            )
            
            # Step 2: Extract and map diagnoses
            mapped_diagnoses = self._extract_and_map_diagnoses(raw_diagnoses)
            
            # Step 3: Judge agent evaluates consistency
            consistency_status, similarity_scores = self._evaluate_consistency(mapped_diagnoses)
            
            print(f"Consistency evaluation result: {consistency_status.value}")
            print(f"Similarity scores: {similarity_scores}")
            
            # Step 4: Perform differentiation analysis
            differentiations = ""
            if consistency_status == DiagnosisStatus.INCONSISTENT and self.enable_differentiation:
                # 传递相似度得分，让其根据得分逐对判断是否需要生成差异
                differentiations = self._generate_differentiation_analysis(raw_diagnoses, mapped_diagnoses, similarity_scores)
            
            # Step 5: Create this round's diagnosis result
            round_diagnosis = RoundDiagnosis(
                round_num=round_num,
                case_id=case_id,
                model_diagnoses=raw_diagnoses,
                similarity_scores=similarity_scores,
                consistency_status=consistency_status,
                differentiations=differentiations,
                api_generated_differences={},
                timestamp=datetime.now().isoformat()
            )
            
            # Add to case result
            case_result.add_round(round_diagnosis)
            
            # Step 6: Determine whether to continue next round
            if consistency_status == DiagnosisStatus.CONSISTENT:
                print(f"\n✅ Diagnosis consistent! Case diagnosis complete")
                case_result.final_status = DiagnosisStatus.CONSISTENT
                case_result.final_diagnosis = " | ".join(mapped_diagnoses.values())
                break
            else:
                print(f"\n❌ Diagnosis inconsistent")
                
                if round_num == self.max_rounds:
                    print(f"\n⚠️  Maximum diagnosis rounds ({self.max_rounds}) reached, diagnosis ended")
                    case_result.final_status = DiagnosisStatus.INCONSISTENT
                    case_result.final_diagnosis = " | ".join(mapped_diagnoses.values())
                else:
                    print(f"\nPreparing for round {round_num + 1} diagnosis...")
        
        # Save diagnosis result
        self._save_case_result(case_result)
        
        print(f"\n{'='*80}")
        print(f"Case {case_id} processing complete")
        print(f"Final status: {case_result.final_status.value}")
        print(f"Final diagnosis: {case_result.final_diagnosis}")
        print(f"Completed rounds: {case_result.max_rounds_completed}")
        print(f"{'='*80}")
        
        return case_result
    
    def _run_three_models_diagnosis(
        self,
        case_id: str,
        patient_data: Dict,
        diagnosis_function: Callable,
        previous_round: Optional[RoundDiagnosis]
    ) -> Dict[str, str]:
        """
        运行三个模型的并行诊断
        
        Args:
            case_id: 病例ID
            patient_data: 患者数据
            diagnosis_function: 诊断函数
            previous_round: 上一轮诊断结果
        
        Returns:
            {model_name: diagnosis_text}
        """
        raw_diagnoses = {}
        
        # 如果存在上一轮诊断结果，获取格式化的past_diagnosis_results和差异分析
        past_diagnosis_results = ""
        difference_analysis = ""
        if previous_round:
            past_diagnosis_results = self._format_past_diagnosis_results(previous_round)
            # 上一轮的differentiations扥存放的就是此轮特别需要的差异分析内容
            difference_analysis = previous_round.differentiations
        
        # 使用线程池并行执行三个模型的诊断
        print(f"  → 启动并行推理...")
        
        with ThreadPoolExecutor(max_workers=len(self.model_names)) as executor:
            # 提交所有任务
            future_to_model = {}
            for model_name in self.model_names:
                future = executor.submit(
                    diagnosis_function,
                    case_id=case_id,
                    patient_data=patient_data,
                    model_name=model_name,
                    past_diagnosis_results=past_diagnosis_results,
                    difference_analysis=difference_analysis
                )
                future_to_model[future] = model_name
            
            # 收集结果
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    diagnosis_text = future.result()
                    raw_diagnoses[model_name] = diagnosis_text
                    print(f"  → {model_name} ✓")
                except Exception as e:
                    print(f"  → {model_name} ✗ (Error: {str(e)})")
                    raw_diagnoses[model_name] = ""
        
        print(f"  → 并行推理完成")
        return raw_diagnoses
    
    def _format_past_diagnosis_results(
        self,
        previous_round: Optional[RoundDiagnosis]
    ) -> str:
        """
        Format the raw diagnosis results from previous round for display in next round
        
        Returns model-anonymized diagnosis results without normalization or mapping
        """
        if previous_round is None:
            return ""
        
        # 使用匿名化的模型编号替代模型名称
        past_results_data = []
        for model_name in self.model_names:
            raw_diagnosis = previous_round.model_diagnoses.get(model_name, "")
            model_index = self.model_index_map.get(model_name, model_name)
            past_results_data.append((model_index, raw_diagnosis))
        
        # 按照 model_index 进行升序排序 (1, 2, 3)
        past_results_data.sort(key=lambda x: x[0] if isinstance(x[0], int) else float('inf'))
        
        # 格式化为最终字符串
        formatted_results = [f"Model {idx}: {diag}" for idx, diag in past_results_data]
        
        return "\n\n\n".join(formatted_results)
    
    def _generate_differentiation_analysis(
        self,
        raw_diagnoses: Dict[str, str],
        mapped_diagnoses: Dict[str, str],
        similarity_scores: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Generate differentiation analysis between different diagnoses from current round
        
        根据相似度得分对诊断对进行精细化控制：
        - 如果诊断对的相似度得分 > 0.82，认为高相似度，跳过该对的差异分析
        - 如果诊断对的相似度得分 <= 0.82，执行差异分析
        
        Args:
            raw_diagnoses: 模型的原始输出
            mapped_diagnoses: 映射后的正常化诊断，格式如 {'model1': 'disease_a', 'model2': 'disease_b', ...}
            similarity_scores: 诊断对的相似度得分，格式如 {'M1-M2': 0.85, 'M1-M3': 0.75, ...}
        
        Returns: 宜存放在 RoundDiagnosis 中，供下一轮使用
        Format: "Disease_A vs Disease_B:\n  1. [difference_point_1]\n  2. [difference_point_2]\n..."
        """
        if similarity_scores is None:
            similarity_scores = {}
        
        # 将所有诊断名称转为小写以统一判断，并去重
        unique_diagnoses = list(set(d.lower() for d in mapped_diagnoses.values()))
        
        # If all diagnoses are the same, no differentiation needed
        if len(unique_diagnoses) <= 1:
            return ""
        
        # 建立诊断名到模型名的映射（逆向映射），用于查找哪些模型输出了特定诊断
        diagnosis_to_models = {}
        for model_name, diagnosis in mapped_diagnoses.items():
            diagnosis_lower = diagnosis.lower()
            if diagnosis_lower not in diagnosis_to_models:
                diagnosis_to_models[diagnosis_lower] = []
            diagnosis_to_models[diagnosis_lower].append(model_name)
        
        # Generate differentiation analysis for each pair of different diagnoses
        analysis_text = ""
        processed_pairs = set()
        
        for i in range(len(unique_diagnoses)):
            for j in range(i + 1, len(unique_diagnoses)):
                # 使用排序后的元组作为键，确保 (A, B) 和 (B, A) 只处理一次
                pair = tuple(sorted([unique_diagnoses[i], unique_diagnoses[j]]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                disease_a, disease_b = pair
                
                # 获取输出这两个诊断的模型
                models_with_disease_a = diagnosis_to_models.get(disease_a, [])
                models_with_disease_b = diagnosis_to_models.get(disease_b, [])
                
                # 检查是否有高相似度得分的诊断对（> 0.82）
                should_skip = False
                for model_a_name in models_with_disease_a:
                    for model_b_name in models_with_disease_b:
                        # 转换模型名称为 M1/M2/M3 格式以匹配 similarity_scores 的键
                        idx_a = self.model_index_map.get(model_a_name, model_a_name)
                        idx_b = self.model_index_map.get(model_b_name, model_b_name)
                        
                        m_a = f"M{idx_a}" if isinstance(idx_a, int) else idx_a
                        m_b = f"M{idx_b}" if isinstance(idx_b, int) else idx_b
                        
                        # 生成可能的相似度得分键（处理可能的多种格式）
                        score_keys = [
                            f"{m_a}-{m_b}", 
                            f"{m_b}-{m_a}"
                        ]
                        
                        for key in score_keys:
                            if key in similarity_scores and similarity_scores[key] > 0.82:
                                print(f"  ℹ️  High similarity detected: {disease_a} vs {disease_b} = {similarity_scores[key]:.4f} (threshold: 0.82) [Key: {key}]")
                                print(f"  → Skipping differentiation analysis for this pair")
                                should_skip = True
                                break
                        if should_skip:
                            break
                    if should_skip:
                        break
                
                if should_skip:
                    continue
                
                # Get differentiation data
                difference_data = self.difference_analyzer.loader.get_differences(disease_a, disease_b)
                
                if difference_data:
                    # 输出格式: "Disease_A vs Disease_B: difference points"
                    analysis_text += f"{disease_a} vs {disease_b}:"
                    difference_points = difference_data.get('difference_points', [])
                    for idx, point in enumerate(difference_points, 1):
                        analysis_text += f"  {idx}. {point}\n"
                else:
                    # 如果在difference.json中找不到，尝试调用Judge Agent API生成
                    disease_pair = f"{disease_a} vs {disease_b}:"
                    api_generated_diff = self._call_judge_api_for_differences(disease_pair)
                    
                    if api_generated_diff:
                        analysis_text += f"{api_generated_diff}\n"
        
        return analysis_text
   
    def _extract_and_map_diagnoses(self, raw_diagnoses: Dict[str, str]) -> Dict[str, str]:
        """
        Extract and map diagnosis names
        """
        mapped_diagnoses = {}
        
        for model_name, diagnosis_text in raw_diagnoses.items():
            # Extract diagnosis
            extracted = extract_diagnosis_from_text(diagnosis_text)
            
            # Map diagnosis
            mapped = map_pathology_name(
                original_sentence=diagnosis_text,
                extracted_diagnosis=extracted,
                mapping_rules=PATHOLOGY_MAPPING_RULES
            )
            
            mapped_diagnoses[model_name] = mapped
        
        return mapped_diagnoses
    
    def _evaluate_consistency(
        self,
        mapped_diagnoses: Dict[str, str]
    ) -> Tuple[DiagnosisStatus, Dict[str, float]]:
        """
        Evaluate diagnosis consistency
        
        Returns:
            (consistency_status, similarity_scores)
        """
        # 按照固定的模型编号顺序(M1, M2, M3)构建诊断列表
        # M1=Llama-70B, M2=Deepseek-70B, M3=Llama-8B
        ordered_model_names = [
            "Llama-3.1-70B-Instruct",      # M1
            "DeepSeek-R1-Distill-Llama-70B",  # M2
            "Llama-3.1-8B-Instruct"        # M3
        ]
        
        diagnoses_list = [mapped_diagnoses.get(model_name, "") for model_name in ordered_model_names]
        
        # Check if all diagnoses are the same
        if len(set(diagnoses_list)) == 1:
            return DiagnosisStatus.CONSISTENT, {}
        
        # Use BioBERT to calculate similarity
        status, scores = self.comparator.compare(diagnoses_list)
        
        if "Consistent" in status:
            return DiagnosisStatus.CONSISTENT, scores
        else:
            return DiagnosisStatus.INCONSISTENT, scores
    
    def _call_judge_api_for_differences(self, disease_pair: str) -> Optional[str]:
        """
        Call Judge Agent API to generate disease differentiation
        
        Args:
            disease_pair: Disease pair string, e.g., "pericarditis vs pulmonary embolism"
        
        Returns:
            Generated differentiation text or None if API call fails
        """
        try:
            import requests
            
            if not self.judge_agent_config:
                return None
            
            api_base = self.judge_agent_config.get('api_base', '')
            api_key = self.judge_agent_config.get('api_key', '')
            model_name = self.judge_agent_config.get('model_name', 'gpt-3.5-turbo')
            temperature = self.judge_agent_config.get('temperature', 0.7)
            
            if not api_base or not api_key:
                return None
            
            # 使用judge_agent_prompts中定义的专业prompt
            diseases = disease_pair.split(' vs ')
            disease_a = diseases[0].strip()
            disease_b = diseases[1].strip() if len(diseases) > 1 else ""
            
            # 构建诊断文本
            diagnoses_text = f"- Disease A: {disease_a}\n- Disease B: {disease_b}"
            
            # 获取差异分析prompt
            user_prompt = get_judge_prompt(
                'differentiation',
                diagnoses_text=diagnoses_text
            )
            
            # 获取系统prompt
            system_prompt = get_judge_prompt('system')
            
            # Call API
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': model_name,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                'temperature': temperature
            }
            
            response = requests.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=data,
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # 尝试解析API返回的JSON格式内容，提取difference_points
                try:
                    api_response = json.loads(content)
                    difference_points = api_response.get('difference_points', [])
                    
                    # 格式化差异点为文本
                    formatted_text = f"{disease_pair}\n"
                    
                    for idx, point in enumerate(difference_points, 1):
                        formatted_text += f"{idx}. {point}\n"
                except (json.JSONDecodeError, TypeError):
                    # 如果无法解析JSON，直接使用原始内容
                    formatted_text = f"{disease_pair}\n"
                    formatted_text += content
                
                return formatted_text
            else:
                print(f"\n❌ API Error - Status Code: {response.status_code}")
                print(f"Response Body:\n{response.text}")
                return None
                
        except Exception as e:
            print(f"\n❌ Exception occurred:")
            print(f"Type: {type(e).__name__}")
            print(f"Detail: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            return None

    def _generate_differentiations(self, mapped_diagnoses: Dict[str, str]) -> Dict:
        """
        Generate differentiation analysis
        
        Returns:
            Dictionary with keys:
                - formatted_text: Formatted differentiation analysis text
                - api_generated_differences: API-generated differences {pair_key: difference_text}
        """
        
        analysis_result = self.difference_analyzer.analyze_inconsistency(
            mapped_diagnoses,
        )
        
        # 检查是否有需要API生成的差异分析
        api_generated_diffs = {}
        pairwise_analyses = analysis_result.get('pairwise_analyses', [])
        
        for pair_analysis in pairwise_analyses:
            if not pair_analysis.get('has_standard_analysis', True):
                # 没有标准分析，需要调用API
                disease_pair = pair_analysis.get('disease_pair', '')
                api_diff = self._call_judge_api_for_differences(disease_pair)
                if api_diff:
                    api_generated_diffs[disease_pair] = api_diff
        
        # Return both formatted text and api-generated differences
        return {
            'formatted_text': analysis_result.get('formatted_text', ''),
            'api_generated_differences': api_generated_diffs
        }
    
    
    def _save_case_result(self, case_result: CaseMultiRoundResult):
        """
        Save case diagnosis result to shared files (append mode)
        
        Saves to:
        1. PKL file: Complete multi-round diagnosis result objects
        2. JSON file: Readable diagnosis summary in append mode
        """
        case_id = case_result.case_id
        pathology = case_result.pathology
        
        # Save PKL in append mode
        pkl_path = os.path.join(self.output_dir, f"{pathology}_all_cases_diagnosis.pkl")
        cases_list = []
        
        # Load existing data if file exists
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    cases_list = pickle.load(f)
            except:
                cases_list = []
        
        # Append new case result
        cases_list.append(case_result)
        
        # Save updated list
        with open(pkl_path, 'wb') as f:
            pickle.dump(cases_list, f)
        
        # Save JSON summary in append mode
        json_path = os.path.join(self.output_dir, f"{pathology}_all_cases_summary.json")
        
        summary = {
            'case_id': str(case_id),  # 转换为字符串应对int64
            'pathology': pathology,
            'final_status': case_result.final_status.value,
            'final_diagnosis': case_result.final_diagnosis,
            'total_rounds': case_result.max_rounds_completed,
            'rounds': [r.to_dict() for r in case_result.rounds]
        }
        
        # 自定义JSON编码器以处理NumPy类型
        class NumpyEncoder(json.JSONEncoder):
            def default(self, o):
                import numpy as np
                if isinstance(o, np.integer):
                    return int(o)
                elif isinstance(o, np.floating):
                    return float(o)
                elif isinstance(o, np.ndarray):
                    return o.tolist()
                return super().default(o)
        
        # Append to JSON file
        cases_data = []
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    cases_data = json.load(f)
            except:
                cases_data = []
        
        cases_data.append(summary)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(cases_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        print(f"\nCase result saved (append mode):")
        print(f"  PKL: {pkl_path}")
        print(f"  JSON: {json_path}")
