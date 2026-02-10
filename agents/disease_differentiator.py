"""
Disease Differentiation Analyzer Module

Provides clinical feature differentiation analysis between diseases
based on JSON database for multi-model diagnosis system
"""

import json
import os
from typing import Dict, List, Optional


class DifferenceDatabaseLoader:
    """
    Load disease differentiation database from JSON file
    """
    
    def __init__(self, json_path: str):
        """
        初始化差异数据库加载器
        
        Args:
            json_path: difference.json文件的路径
        """
        self.json_path = json_path
        self.difference_db: Dict = {}
        self.disease_pair_map: Dict = {}
        self._load_database()
    
    def _load_database(self):
        """从JSON文件加载差异数据库"""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"差异数据库文件不存在: {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.difference_db = json.load(f)
        
        if self.difference_db is None:
            raise ValueError("差异数据库加载失败")
        
        # 构建疾病对映射（用于快速查找）
        self._build_disease_pair_map()
    
    def _build_disease_pair_map(self):
        """Build mapping from disease pairs to differentiation data"""
        for category, pairs in self.difference_db.items():
            if category == 'meta_info':
                continue
            
            for pair_id, pair_data in pairs.items():
                # Add forward and reverse mappings
                disease_a = pair_data['disease_A'].lower().strip()
                disease_b = pair_data['disease_B'].lower().strip()
                
                self.disease_pair_map[f"{disease_a}_vs_{disease_b}"] = pair_data
                self.disease_pair_map[f"{disease_b}_vs_{disease_a}"] = self._reverse_pair_data(pair_data)
    
    def _reverse_pair_data(self, pair_data: Dict) -> Dict:
        """Reverse disease pair data (for reverse lookup)"""
        reversed_data = pair_data.copy()
        reversed_data['disease_A'], reversed_data['disease_B'] = pair_data['disease_B'], pair_data['disease_A']
        return reversed_data
    
    def _normalize_disease_name(self, disease_name: str) -> str:
        """
        Normalize disease name for querying
        输入已经是经过map_pathology_name处理的标准疾病名
        只需要转小写和去除首尾空格即可
        
        Args:
            disease_name: Already mapped disease name (from mapped_diagnoses)
        
        Returns:
            Normalized disease name (lowercase and stripped)
        """
        # 直接转小写并去除空格，不再调用map_pathology_name
        # 因为输入已经是规范化的疾病名称
        return disease_name.lower().strip()
    
    def get_differences(self, disease_a: str, disease_b: str) -> Optional[Dict]:
        """
        Get differences between two diseases
        
        Args:
            disease_a: Name of disease A (already mapped)
            disease_b: Name of disease B (already mapped)
        
        Returns:
            Dictionary containing differentiation information, or None if not found
        """
        norm_a = self._normalize_disease_name(disease_a)
        norm_b = self._normalize_disease_name(disease_b)
        
        # 添加详细调试信息
        print(f"    [DIFF LOOKUP] Input: '{disease_a}' <-> '{disease_b}'")
        print(f"    [DIFF LOOKUP] Normalized: '{norm_a}' <-> '{norm_b}'")
        
        # Try to find disease pair
        pair_key = f"{norm_a}_vs_{norm_b}"
        print(f"    [DIFF LOOKUP] Query key: '{pair_key}'")
        
        if pair_key in self.disease_pair_map:
            print(f"    [DIFF LOOKUP] ✓ FOUND in database!")
            return self.disease_pair_map[pair_key]
        
        # 显示可能匹配的键
        print(f"    [DIFF LOOKUP] ✗ NOT FOUND in database")
        related_keys = [k for k in self.disease_pair_map.keys() if norm_a in k or norm_b in k]
        if related_keys:
            print(f"    [DIFF LOOKUP] Related keys in database:")
            for key in related_keys[:5]:
                print(f"      - '{key}'")
        else:
            print(f"    [DIFF LOOKUP] No related keys found. Sample keys:")
            for key in list(self.disease_pair_map.keys())[:10]:
                print(f"      - '{key}'")
        
        # Return None if no exact match found
        return None
    
    def format_difference_points(self, difference_data: Dict) -> str:
        """
        Format differentiation data into readable text
        
        Args:
            difference_data: Differentiation data dictionary
        
        Returns:
            Formatted differentiation text
        """
        if not difference_data:
            return ""
        
        disease_a = difference_data.get('disease_A', '')
        disease_b = difference_data.get('disease_B', '')
        difference_points = difference_data.get('difference_points', [])
        
        text = f"Disease Differentiation Analysis: {disease_a} vs {disease_b}\n"
        
        for i, point in enumerate(difference_points, 1):
            text += f"{i}. {point}\n"
        
        return text


class MultiModelDifferenceAnalyzer:
    """
    Differentiation analyzer for multi-model diagnosis results
    """
    
    def __init__(self, difference_json_path: str):
        """
        初始化多模型差异分析器
        
        Args:
            difference_json_path: difference.json文件的路径
        """
        self.loader = DifferenceDatabaseLoader(difference_json_path)
    
    def analyze_inconsistency(
        self,
        model_diagnoses: Dict[str, str],
    ) -> Dict:
        """
        Analyze diagnostic inconsistencies among multiple models
        
        Args:
            model_diagnoses: Diagnosis results in {model_name: diagnosis} format
            model_names: List of model names
        
        Returns:
            Dictionary containing analysis results
        """
        analysis_result = {
            'is_consistent': False,
            'unique_diagnoses': [],
            'pairwise_analyses': [],
            'formatted_text': ""
        }
        
        # Get unique diagnoses
        unique_diagnoses = list(set(model_diagnoses.values()))
        analysis_result['unique_diagnoses'] = unique_diagnoses
        
        if len(unique_diagnoses) <= 1:
            analysis_result['is_consistent'] = True
            return analysis_result
        
        # Pairwise comparison and differentiation analysis
        formatted_text = ""
        all_difference_analyses = []
        
        for i in range(len(unique_diagnoses)):
            for j in range(i + 1, len(unique_diagnoses)):
                disease_a = unique_diagnoses[i]
                disease_b = unique_diagnoses[j]
                
                # Get differentiation data
                difference_data = self.loader.get_differences(disease_a, disease_b)
                
                if difference_data:
                    # Standard differentiation analysis exists
                    formatted_text += self.loader.format_difference_points(difference_data)
                    all_difference_analyses.append({
                        'disease_pair': f"{disease_a} vs {disease_b}",
                        'has_standard_analysis': True,
                        'analysis': difference_data
                    })
                else:
                    all_difference_analyses.append({
                        'disease_pair': f"{disease_a} vs {disease_b}",
                        'has_standard_analysis': False,
                        'analysis': None
                    })
        
        analysis_result['pairwise_analyses'] = all_difference_analyses
        analysis_result['formatted_text'] = formatted_text
        
        return analysis_result
    
    def generate_model_specific_context(
        self,
        current_diagnosis: str,
        other_diagnoses: Dict[str, str],
        model_index_map: Dict[str, int] = None,
        api_generated_differences: Dict[str, str] = None,
        enable_differentiation: bool = True
    ) -> str:
        """
        Generate diagnosis context with differentiation analysis for specific model
        
        Args:
            model_name: Current model name
            current_diagnosis: Current model's diagnosis
            other_diagnoses: Diagnoses from other models {model_name: diagnosis}
            model_index_map: Mapping of model names to indices {model_name: index}
            api_generated_differences: API-generated difference content {pair_key: difference_text}
            enable_differentiation: Whether to enable differentiation analysis (True: full mode, False: simple mode)
        
        Returns:
            Formatted context text
        """
        context = "=" * 80 + "\n"
        context += "[Differentiation Analysis Context from Previous Round]\n"
        context += f"Your diagnosis in the previous round was: {current_diagnosis}\n\n"
        
        
        for other_model_name, other_diagnosis in other_diagnoses.items():
            
            # Use model index if available, otherwise use model name
            if model_index_map and other_model_name in model_index_map:
                other_model_display = f"Model {model_index_map[other_model_name]}"
            else:
                other_model_display = other_model_name
            
            context += f"Comparison Model: {other_model_display}\n"
            context += f"Its Diagnosis: {other_diagnosis}\n"
            
            if other_diagnosis != current_diagnosis:
                # 如果禁用差异分析，直接跳过
                if not enable_differentiation:
                    context += "\n"  # 只添加空行
                    continue
                
                # Get differentiation explanation
                difference_data = self.loader.get_differences(current_diagnosis, other_diagnosis)
                
                if difference_data:
                    context += f"\nKey Differences:\n"
                    # Show all difference points (no limit)
                    for idx, point in enumerate(difference_data.get('difference_points', []), 1):
                        context += f"  {idx}. {point}\n"
                else:
                    # Try to use API-generated difference if available
                    if api_generated_differences:
                        # 修正键的匹配逻辑 - 使用 " vs " 格式（带空格）
                        pair_key = f"{current_diagnosis} vs {other_diagnosis}"
                        reverse_pair_key = f"{other_diagnosis} vs {current_diagnosis}"
                    
                        
                        api_diff = api_generated_differences.get(pair_key) or api_generated_differences.get(reverse_pair_key)
                        
                        if api_diff:
                            context += f"\nKey Differences: \n"
                            context += f"{api_diff}\n"
                        else:
                            context += f"\nNote: One of the above diagnoses may be outside the standard range. Judge Agent analysis pending.\n"
                    else:
                        context += f"\nNote: One of the above diagnoses may be outside the standard range\n"
            
            context += "\n"
        
        context += "Please re diagnose this disease based on the previous diagnosis results and the differences between the diseases currently provided. Please note that the results provided by other models may not be accurate. All the above information is only for your reference, and you still need to make the correct judgment based on the actual situation of the patient\n"
        context += "=" * 80 + "\n"
        
        return context
