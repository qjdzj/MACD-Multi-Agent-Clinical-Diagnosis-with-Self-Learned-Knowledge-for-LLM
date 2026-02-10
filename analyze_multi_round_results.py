#!/usr/bin/env python3
"""
实验结果分析脚本
分析multi_round_results目录下各疾病的诊断一致性结果
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from thefuzz import fuzz
import pandas as pd

import sys
sys.path.append('/data2/kunzhang/MIMIC-CDM/MIMIC-Clinical-Decision-Making-Framework-llama3.1_copy_copy')

from utils.nlp import keyword_positive, remove_punctuation


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
            {"location": "pericardial", "modifiers": ["effusion", "thickening", "fluid", "fibrosis", "adhesions"]},
            {"location": "myopericarditis", "modifiers": []}
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


def _normalize_disease_name(diagnosis: str) -> str:
    """
    归一化疾病名称，移除括号和特殊格式
    """
    if not diagnosis:
        return ""
    normalized = diagnosis.strip().lower()
    # 移除括号及其内容
    normalized = re.sub(r'\s*\([^)]*\)', '', normalized)
    return normalized


def map_pathology_name(original_sentence: str, extracted_diagnosis: str, mapping_rules: Optional[Dict[str, Any]] = None) -> str:
    """
    对提取的诊断名进行全局映射
    
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

    # 遍历规则库中的每一种疾病及其规则
    for pathology_name, rules in mapping_rules.items():
        # 层级 1: 专属正则表达式匹配 (如果有)
        if 'regex_patterns' in rules:
            for pattern in rules['regex_patterns']:
                match = pattern.search(extracted_diagnosis)
                if match and keyword_positive(original_sentence, match.group(0)):
                    return pathology_name

        # 层级 2: 模糊字符串匹配
        answer_for_search = remove_punctuation(extracted_diagnosis.lower())
        for word in answer_for_search.split():
            if fuzz.ratio(word, pathology_name) > 90 and keyword_positive(original_sentence, word):
                return pathology_name

        # 层级 3: 通用关键词组合匹配
        all_alternatives = rules.get("alternatives", []) + rules.get("gracious_alternatives", [])
        for alt_rule in all_alternatives:
            loc = alt_rule["location"]
            loc_pattern = r'\b' + re.escape(loc) + r'\b'
        
            if not alt_rule["modifiers"]:
                if re.search(loc_pattern, answer_for_search, re.IGNORECASE) and keyword_positive(original_sentence, loc):
                    return pathology_name
                continue
            
            for mod in alt_rule["modifiers"]:
                mod_pattern = r'\b' + re.escape(mod) + r'\b'
                if (re.search(loc_pattern, answer_for_search, re.IGNORECASE) and
                    re.search(mod_pattern, answer_for_search, re.IGNORECASE) and
                    keyword_positive(original_sentence, loc) and
                    keyword_positive(original_sentence, mod)):
                    return pathology_name

    return extracted_diagnosis


def check_diagnosis_match(diagnosis: str, pathology: str) -> bool:
    """
    检查诊断是否与目标疾病匹配
    """
    normalized_diagnosis = _normalize_disease_name(diagnosis)
    mapped_diagnosis = map_pathology_name(diagnosis, normalized_diagnosis)
    mapped_diagnosis_lower = _normalize_disease_name(mapped_diagnosis)
    pathology_lower = pathology.lower()
    
    return mapped_diagnosis_lower == pathology_lower


def analyze_case(case_data: Dict[str, Any], pathology: str) -> Dict[str, Any]:
    """
    分析单个病例的诊断结果
    
    Returns:
        包含分析结果的字典
    """
    case_id = case_data.get('case_id', 'Unknown')
    final_status = case_data.get('final_status', '')
    final_diagnosis = case_data.get('final_diagnosis', '')
    total_rounds = case_data.get('total_rounds', 0)
    
    is_consistent = '✅' in final_status or 'Consistent' in final_status
    
    # 确定在哪一轮达成一致
    consistent_at_round = None
    if is_consistent:
        rounds = case_data.get('rounds', [])
        for round_data in rounds:
            round_status = round_data.get('consistency_status', '')
            if '✅' in round_status or 'Consistent' in round_status:
                consistent_at_round = round_data.get('round_num', None)
                break
    
    result = {
        'case_id': case_id,
        'is_consistent': is_consistent,
        'final_diagnosis': final_diagnosis,
        'pathology': pathology,
        'total_rounds': total_rounds,
        'consistent_at_round': consistent_at_round
    }
    
    # 统一检查逻辑：只要有一个诊断匹配pathology，则记为any_match
    diagnoses = [d.strip() for d in final_diagnosis.split('|')]
    matches = [check_diagnosis_match(d, pathology) for d in diagnoses if d]
    result['any_match'] = any(matches) if matches else False
    
    return result


def analyze_disease_directory(disease_dir: Path, pathology: str) -> Optional[Dict[str, Any]]:
    """
    分析单个疾病目录的所有病例
    """
    summary_file = None
    for file in disease_dir.glob('*all_cases_summary.json'):
        summary_file = file
        break
    
    if not summary_file or not summary_file.exists():
        return None
    
    print(f"  读取文件: {summary_file}")
    with open(summary_file, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    
    total_cases = len(cases)
    consistent_cases = []
    inconsistent_cases = []
    
    # 按轮次统计一致性
    round_1_consistent = 0
    round_2_consistent = 0
    round_3_consistent = 0
    
    # 分析每个病例
    for case in cases:
        result = analyze_case(case, pathology)
        if result['is_consistent']:
            consistent_cases.append(result)
            # 统计在哪一轮达成一致
            consistent_round = result.get('consistent_at_round')
            if consistent_round == 1:
                round_1_consistent += 1
            elif consistent_round == 2:
                round_2_consistent += 1
            elif consistent_round == 3:
                round_3_consistent += 1
        else:
            inconsistent_cases.append(result)
    
    consistent_count = len(consistent_cases)
    inconsistent_count = len(inconsistent_cases)
    
    # 统计一致病例中的匹配情况 (现在改为只要有一个匹配就记为匹配)
    consistent_match = sum(1 for c in consistent_cases if c.get('any_match', False))
    consistent_no_match = consistent_count - consistent_match
    
    # 统计不一致病例中的匹配情况
    inconsistent_any_match = sum(1 for c in inconsistent_cases if c.get('any_match', False))
    inconsistent_no_match = inconsistent_count - inconsistent_any_match
    
    # 计算有效意见率和无效意见率
    effective_opinions = consistent_match + inconsistent_any_match
    ineffective_opinions = consistent_no_match + inconsistent_no_match
    
    results = {
        'pathology': pathology,
        'total_cases': total_cases,
        # 维度 1: 模型间一致性
        'model_consistent': consistent_count,
        'model_inconsistent': inconsistent_count,
        # 维度 2: 与病理一致性 (只要有一个匹配即记为一致，其余记为不一致)
        'pathology_consistent': effective_opinions,
        'pathology_inconsistent': ineffective_opinions,
        # 维度 3: 一致且有效的统计
        'consistent_and_effective': consistent_match,
        'consistent_but_ineffective': consistent_no_match,
        # 维度 4: 按轮次统计一致性
        'round_1_consistent': round_1_consistent,
        'round_2_consistent': round_2_consistent,
        'round_3_consistent': round_3_consistent,
        # 保持旧字段兼容性
        'effective_opinions': effective_opinions,
        'ineffective_opinions': ineffective_opinions,
        'effective_opinion_rate': effective_opinions,
        'ineffective_opinion_rate': ineffective_opinions
    }
    
    # 计算百分比（基于总病例数）
    if total_cases > 0:
        results['model_consistent_percent'] = (consistent_count / total_cases) * 100
        results['model_inconsistent_percent'] = (inconsistent_count / total_cases) * 100
        results['pathology_consistent_percent'] = (effective_opinions / total_cases) * 100
        results['pathology_inconsistent_percent'] = (ineffective_opinions / total_cases) * 100
        results['consistent_and_effective_percent'] = (consistent_match / total_cases) * 100
        results['consistent_but_ineffective_percent'] = (consistent_no_match / total_cases) * 100
        results['effective_opinion_rate'] = (effective_opinions / total_cases) * 100
        results['ineffective_opinion_rate'] = (ineffective_opinions / total_cases) * 100
        # 按轮次的百分比
        results['round_1_consistent_percent'] = (round_1_consistent / total_cases) * 100
        results['round_2_consistent_percent'] = (round_2_consistent / total_cases) * 100
        results['round_3_consistent_percent'] = (round_3_consistent / total_cases) * 100
    else:
        results['model_consistent_percent'] = 0.0
        results['model_inconsistent_percent'] = 0.0
        results['pathology_consistent_percent'] = 0.0
        results['pathology_inconsistent_percent'] = 0.0
        results['consistent_and_effective_percent'] = 0.0
        results['consistent_but_ineffective_percent'] = 0.0
        results['effective_opinion_rate'] = 0.0
        results['ineffective_opinion_rate'] = 0.0
        results['round_1_consistent_percent'] = 0.0
        results['round_2_consistent_percent'] = 0.0
        results['round_3_consistent_percent'] = 0.0
    
    return results


def main():
    """
    主函数：分析所有疾病的诊断结果
    """
    base_dir = Path('/data2/kunzhang/MIMIC-CDM/MIMIC-Clinical-Decision-Making-Framework-llama3.1_copy_copy/multi_round_results_human')
    
    if not base_dir.exists():
        print(f"错误：目录不存在 {base_dir}")
        return
    
    print(f"开始分析目录: {base_dir}\n")
    
    all_results = []
    
    # 获取所有疾病子目录
    disease_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    for disease_dir in sorted(disease_dirs):
        pathology = disease_dir.name
        print(f"\n正在分析疾病: {pathology}")
        print(f"{'='*60}")
        
        results = analyze_disease_directory(disease_dir, pathology)
        
        if results:
            all_results.append(results)
            print(f"  总病例数: {results['total_cases']}")
            print(f"  一致病例: {results['model_consistent']}")
            print(f"  不一致病例: {results['model_inconsistent']}")
        else:
            print(f"  警告：未找到summary文件")
    
    if not all_results:
        print("\n错误：未找到任何数据")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(all_results)
    
    # 计算总体统计数据
    total_all_cases = df['total_cases'].sum()
    total_effective_opinions = df['effective_opinions'].sum()
    total_ineffective_opinions = df['ineffective_opinions'].sum()
    total_consistent_and_effective = df['consistent_and_effective'].sum()
    total_consistent_but_ineffective = df['consistent_but_ineffective'].sum()
    total_round_1_consistent = df['round_1_consistent'].sum()
    total_round_2_consistent = df['round_2_consistent'].sum()
    total_round_3_consistent = df['round_3_consistent'].sum()
    
    # 有效意见率的平均值（所有疾病的平均）
    avg_effective_opinion_rate = df['effective_opinion_rate'].mean()
    # 有效意见占所有病例总和的百分比
    effective_opinion_percent_of_all = (total_effective_opinions / total_all_cases) * 100 if total_all_cases > 0 else 0.0
    
    # 无效意见率的平均值（所有疾病的平均）
    avg_ineffective_opinion_rate = df['ineffective_opinion_rate'].mean()
    # 无效意见占所有病例总和的百分比
    ineffective_opinion_percent_of_all = (total_ineffective_opinions / total_all_cases) * 100 if total_all_cases > 0 else 0.0
    
    # 一致且有效的统计
    avg_consistent_and_effective_rate = df['consistent_and_effective_percent'].mean()
    consistent_and_effective_percent_of_all = (total_consistent_and_effective / total_all_cases) * 100 if total_all_cases > 0 else 0.0
    
    # 打印详细报告
    print("\n" + "="*100)
    print("实验结果汇总报告")
    print("="*100 + "\n")
    
    for _, row in df.iterrows():
        print(f"\n{'='*80}")
        print(f"疾病: {row['pathology']}")
        print(f"{'='*80}")
        print(f"总病例数: {row['total_cases']}")
        
        print(f"\n【评估维度 1：模型间诊断一致性统计】")
        print(f"  一致 (Consistent): {row['model_consistent']} ({row['model_consistent_percent']:.2f}%)")
        print(f"  不一致 (Inconsistent): {row['model_inconsistent']} ({row['model_inconsistent_percent']:.2f}%)")
        
        print(f"\n【评估维度 2：与病理诊断一致性统计 (有效性评估)】")
        print(f"  有效(Matched/Effective): {row['pathology_consistent']} ({row['pathology_consistent_percent']:.2f}%)")
        print(f"  无效 (Unmatched/Ineffective): {row['pathology_inconsistent']} ({row['pathology_inconsistent_percent']:.2f}%)")
        
        print(f"\n【评估维度 3：模型一致 + 病理有效的统计】")
        print(f"  一致且有效: {row['consistent_and_effective']} ({row['consistent_and_effective_percent']:.2f}%)")
        print(f"  一致但无效: {row['consistent_but_ineffective']} ({row['consistent_but_ineffective_percent']:.2f}%)")
        
        print(f"\n【评估维度 4：按轮次统计一致性】")
        print(f"  第1轮达成一致: {row['round_1_consistent']} ({row['round_1_consistent_percent']:.2f}%)")
        print(f"  第2轮达成一致: {row['round_2_consistent']} ({row['round_2_consistent_percent']:.2f}%)")
        print(f"  第3轮达成一致: {row['round_3_consistent']} ({row['round_3_consistent_percent']:.2f}%)")
    
    # 打印总体统计数据
    print(f"\n\n{'='*100}")
    print("总体统计数据")
    print(f"{'='*100}")
    print(f"\n所有疾病总病例数: {total_all_cases}")
    
    print(f"\n【有效性总体统计 (方法 2 - 与病理一致性)】")
    print(f"  所有疾病有效一致率的平均值: {avg_effective_opinion_rate:.2f}%")
    print(f"  有效一致数占所有病例总和的百分比: {total_effective_opinions}/{total_all_cases} ({effective_opinion_percent_of_all:.2f}%)")
    
    print(f"\n【无效性总体统计 (方法 2 - 与病理不一致)】")
    print(f"  所有疾病无效不一致率的平均值: {avg_ineffective_opinion_rate:.2f}%")
    print(f"  无效不一致数占所有病例总和的百分比: {total_ineffective_opinions}/{total_all_cases} ({ineffective_opinion_percent_of_all:.2f}%)")
    
    print(f"\n【一致且有效统计 (模型一致 + 病理匹配)】")
    print(f"  所有疾病一致且有效率的平均值: {avg_consistent_and_effective_rate:.2f}%")
    print(f"  一致且有效数占所有病例总和的百分比: {total_consistent_and_effective}/{total_all_cases} ({consistent_and_effective_percent_of_all:.2f}%)")
    print(f"  一致但无效数占所有病例总和的百分比: {total_consistent_but_ineffective}/{total_all_cases} ({(total_consistent_but_ineffective / total_all_cases * 100) if total_all_cases > 0 else 0.0:.2f}%)")
    
    # 计算按轮次一致性的统计
    avg_round_1_rate = df['round_1_consistent_percent'].mean()
    avg_round_2_rate = df['round_2_consistent_percent'].mean()
    avg_round_3_rate = df['round_3_consistent_percent'].mean()
    round_1_percent_of_all = (total_round_1_consistent / total_all_cases) * 100 if total_all_cases > 0 else 0.0
    round_2_percent_of_all = (total_round_2_consistent / total_all_cases) * 100 if total_all_cases > 0 else 0.0
    round_3_percent_of_all = (total_round_3_consistent / total_all_cases) * 100 if total_all_cases > 0 else 0.0
    
    print(f"\n【按轮次达成一致统计】")
    print(f"  第1轮达成一致:")
    print(f"    - 所有疾病平均值: {avg_round_1_rate:.2f}%")
    print(f"    - 占所有病例总和: {total_round_1_consistent}/{total_all_cases} ({round_1_percent_of_all:.2f}%)")
    print(f"  第2轮达成一致:")
    print(f"    - 所有疾病平均值: {avg_round_2_rate:.2f}%")
    print(f"    - 占所有病例总和: {total_round_2_consistent}/{total_all_cases} ({round_2_percent_of_all:.2f}%)")
    print(f"  第3轮达成一致:")
    print(f"    - 所有疾病平均值: {avg_round_3_rate:.2f}%")
    print(f"    - 占所有病例总和: {total_round_3_consistent}/{total_all_cases} ({round_3_percent_of_all:.2f}%)")
    
    # 保存CSV文件
    output_csv = base_dir / 'analysis_summary.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n{'='*80}")
    print(f"详细结果已保存至: {output_csv}")
    
    # 保存总体统计数据
    consistent_but_ineffective_percent_of_all = (total_consistent_but_ineffective / total_all_cases) * 100 if total_all_cases > 0 else 0.0
    summary_stats = {
        'total_all_cases': [total_all_cases],
        'total_effective_opinions': [total_effective_opinions],
        'total_ineffective_opinions': [total_ineffective_opinions],
        'total_consistent_and_effective': [total_consistent_and_effective],
        'total_consistent_but_ineffective': [total_consistent_but_ineffective],
        'avg_effective_opinion_rate': [avg_effective_opinion_rate],
        'effective_opinion_percent_of_all': [effective_opinion_percent_of_all],
        'avg_ineffective_opinion_rate': [avg_ineffective_opinion_rate],
        'ineffective_opinion_percent_of_all': [ineffective_opinion_percent_of_all],
        'avg_consistent_and_effective_rate': [avg_consistent_and_effective_rate],
        'consistent_and_effective_percent_of_all': [consistent_and_effective_percent_of_all],
        'consistent_but_ineffective_percent_of_all': [consistent_but_ineffective_percent_of_all],
        'total_round_1_consistent': [total_round_1_consistent],
        'total_round_2_consistent': [total_round_2_consistent],
        'total_round_3_consistent': [total_round_3_consistent],
        'avg_round_1_rate': [avg_round_1_rate],
        'avg_round_2_rate': [avg_round_2_rate],
        'avg_round_3_rate': [avg_round_3_rate],
        'round_1_percent_of_all': [round_1_percent_of_all],
        'round_2_percent_of_all': [round_2_percent_of_all],
        'round_3_percent_of_all': [round_3_percent_of_all]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_csv = base_dir / 'overall_summary.csv'
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f"总体统计数据已保存至: {summary_csv}")
    
    
    print(f"{'='*80}\n")
    print("分析完成！")


if __name__ == '__main__':
    main()
