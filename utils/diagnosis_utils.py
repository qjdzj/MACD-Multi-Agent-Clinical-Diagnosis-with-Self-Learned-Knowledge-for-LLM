"""
Diagnosis Utility Module

Provides diagnosis extraction, mapping, and consistency evaluation functions
"""

import re
import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from utils.nlp import keyword_positive, remove_punctuation
from thefuzz import fuzz


# Disease mapping rules
PATHOLOGY_MAPPING_RULES = {
    "appendicitis": {
        "alternatives": [{"location": "appendi", "modifiers": ["gangren", "infect", "inflam", "abscess", "rupture", "necros", "perf"]}],
        "gracious_alternatives": [
            {"location": "Appendercitis", "modifiers": []}
        ]
    },
    "cholecystitis": {
        "alternatives": [
            {"location": "gallbladder", "modifiers": ["gangren", "infect", "inflam", "abscess", "necros", "perf", "inflammation"]},
            {"location": "cholangitis", "modifiers": ["cholangitis"]}
        ],
        "gracious_alternatives": [
            {"location": "gallbladder", "modifiers": ["disease", "attack"]},
            {"location": "biliary", "modifiers": ["colic"]}
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
            {"location": "pericardial", "modifiers": ["effusion", "thickening", "fluid", "fibrosis", "adhesions"]},
            {"location": "Myopericarditis", "modifiers": []}
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
            {"location": "pulmonary", "modifiers": ["embolism", "embolus", "thrombus", "thromboembolism"]},
            {"location": "pe", "modifiers": []}
        ],
        "gracious_alternatives": [],
        "regex_patterns": [
            re.compile(r'\b(?:pulmonary\s+embolism|PE)\b', re.IGNORECASE),
            re.compile(r'\b(?:pulmonary|PE)\b.*?\b(?:embolus|thrombus)\b', re.IGNORECASE)
        ]
    }
}


def extract_diagnosis_from_text(raw_text: str) -> str:
    """
    从诊断文本中提取诊断名称
    
    Args:
        raw_text: 原始诊断文本
    
    Returns:
        提取的诊断名称
    """
    if not isinstance(raw_text, str):
        return "Invalid Input Type"
    
    prediction = "Final Diagnosis: " + raw_text
    regex = r"(Final )?Diagnosis:([\s\S]*?)(?=[\.\n].*Treatment.*:|$)"
    match = re.search(regex, prediction, flags=re.IGNORECASE)
    diagnosis = match.group(2).strip() if match else raw_text
    
    # 清理诊断文本
    cleanup_keywords = [
        "rationale", "note", "recommendation", "explanation", "finding",
        "other.*diagnos.*include", "other.*diagnos.*considered(?: were)?",
        "management", "action", "plan", "reasoning", "reasons", "assessment",
        "justification", "tests", "additional diagnoses", "notification",
        "impression", "background", "additional findings include",
        "diagnostic criteria.*", "diagnostic criteria met", "criteria met",
        "characterized by", "rare", "Rare"
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


def map_pathology_name(
    original_sentence: str,
    extracted_diagnosis: str,
    mapping_rules: Dict = None

) -> str:
    """
    对提取的诊断名进行全局映

    Args:
        original_sentence: 原始文本
        extracted_diagnosis: 提取的诊断
        mapping_rules: 映射规则（默认使用PATHOLOGY_MAPPING_RULES）

    Returns:
        映射后的疾病名称
    """
    if mapping_rules is None:
        mapping_rules = PATHOLOGY_MAPPING_RULES

    if not extracted_diagnosis:
        return ""

    # 新增逻辑：首先检查是否包含PATHOLOGY_MAPPING_RULES中的疾病关键字
    extracted_lower = extracted_diagnosis.lower()
    for pathology_name in mapping_rules.keys():
        if pathology_name.lower() in extracted_lower:
            # 如果找到匹配的关键字，直接返回基础疾病名称（不带形容词）
            return pathology_name

    # 如果没有找到关键字匹配，再使用原来的映射逻辑
    # 预处理：移除常见形容词前缀，使 "Acute Pancreatitis" -> "Pancreatitis"
    processed_diagnosis = re.sub(r'^(acute|chronic|severe|mild|moderate|recurrent|subacute|non[-\s]acute)\s+', '', extracted_diagnosis, flags=re.IGNORECASE)

    # 遍历规则库中的每一种疾病及其规则
    for pathology_name, rules in mapping_rules.items():
        # 层级 1: 专属正则表达式匹配 (如果有)
        if 'regex_patterns' in rules:
            for pattern in rules['regex_patterns']:
                match = pattern.search(extracted_diagnosis)
                if match and keyword_positive(original_sentence, match.group(0)):
                    return pathology_name

        # 层级 2: 模糊字符串匹配 - 针对原始诊断
        answer_for_search = remove_punctuation(extracted_diagnosis.lower())
        for word in answer_for_search.split():
            if fuzz.ratio(word, pathology_name) > 90 and keyword_positive(original_sentence, word):
                return pathology_name

        # 层级 2b: 模糊字符串匹配 - 针对预处理后的诊断
        processed_answer_for_search = remove_punctuation(processed_diagnosis.lower())
        for word in processed_answer_for_search.split():
            if fuzz.ratio(word, pathology_name) > 90 and keyword_positive(original_sentence, word):
                return pathology_name

        # 层级 3: 通用关键词组合匹配 - 针对原始诊断
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

        # 层级 3b: 通用关键词组合匹配 - 针对预处理后的诊断
        for alt_rule in all_alternatives:
            loc = alt_rule["location"]
            loc_pattern = r'\b' + re.escape(loc) + r'\b'

            if not alt_rule["modifiers"]:
                if re.search(loc_pattern, processed_answer_for_search) and keyword_positive(original_sentence, loc):
                    return pathology_name
                continue

            for mod in alt_rule["modifiers"]:
                mod_pattern = r'\b' + re.escape(mod) + r'\b'
                if (re.search(loc_pattern, processed_answer_for_search) and
                    re.search(mod_pattern, processed_answer_for_search) and
                    keyword_positive(original_sentence, loc) and
                    keyword_positive(original_sentence, mod)):
                    return pathology_name

    # 如果以上都没有匹配，尝试简单包含匹配（处理类似"Acute Pancreatitis"包含"pancreatitis"的情况）
    processed_lower = processed_diagnosis.lower()
    for pathology_name in mapping_rules.keys():
        if pathology_name.lower() in processed_lower:
            return pathology_name



    return extracted_diagnosis


class DiagnosisComparator:
    """
    使用BioBERT进行诊断一致性评估
    """
    
    def __init__(self, model_path: str, threshold: float = 0.8):
        """
        初始化诊断比较器
        
        Args:
            model_path: BioBERT模型路径
            threshold: 一致性阈值
        """
        self.model_path = model_path
        self.threshold = threshold
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        """Load BioBERT model"""
        if self.model is None and self.model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("BioBERT model loaded successfully.")

    def _encode_texts(self, texts: List[str], max_length: int = 128) -> np.ndarray:
        """Encode texts to vectors"""
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def _normalize_for_comparison(self, diagnosis: str) -> str:
        """Normalize diagnosis text for comparison"""
        if not diagnosis:
            return ""
        norm_dx = diagnosis.lower()
        norm_dx = re.sub(r'\([^)]*\)', '', norm_dx)
        split_phrases = [' with ', ' secondary to ', ' due to ', '/', ' vs ']
        for phrase in split_phrases:
            if phrase in norm_dx:
                norm_dx = norm_dx.split(phrase)[0]
        return norm_dx.strip()

    def compare(self, diagnoses: List[str]) -> Tuple[str, Dict[str, float]]:
        """
        比较诊断的一致性
        
        Args:
            diagnoses: 诊断列表
        
        Returns:
            (一致性状态, 相似度分数字典)
        """
        if len(diagnoses) < 2:
            return "Insufficient information", {}
        if not self.model_path:
            return "Error: Model path not configured or failed to load", {}

        self._load_model()
        if not self.model:
            return "Error: Model failed to load successfully", {}
        
        normalized_diagnoses = [self._normalize_for_comparison(dx) for dx in diagnoses]
        scores = {}

        if len(diagnoses) == 2:
            if any(not dx for dx in normalized_diagnoses):
                return "❌ Inconsistent (empty diagnosis exists)", {}
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
                return "❌ Inconsistent (insufficient valid diagnoses)", {}
        
        return "Insufficient information", {}
