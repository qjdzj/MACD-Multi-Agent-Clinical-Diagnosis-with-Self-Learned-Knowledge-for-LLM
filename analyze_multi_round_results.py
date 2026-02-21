#!/usr/bin/env python3
"""
Experimental Results Analysis Script
Analyze diagnosis consistency results for each disease in the multi_round_results directory
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
    Normalize disease name, remove parentheses and special formats
    """
    if not diagnosis:
        return ""
    normalized = diagnosis.strip().lower()
    # Remove parentheses and their content
    normalized = re.sub(r'\s*\([^)]*\)', '', normalized)
    return normalized


def map_pathology_name(original_sentence: str, extracted_diagnosis: str, mapping_rules: Optional[Dict[str, Any]] = None) -> str:
    """
    Global mapping for extracted diagnosis names
    
    Args:
        original_sentence: Original text
        extracted_diagnosis: Extracted diagnosis
        mapping_rules: Mapping rules (default uses PATHOLOGY_MAPPING_RULES)
    
    Returns:
        Mapped disease name
    """
    if mapping_rules is None:
        mapping_rules = PATHOLOGY_MAPPING_RULES
    
    if not extracted_diagnosis:
        return ""

    # 遍历规则库中的每一种疾病及其规则
    for pathology_name, rules in mapping_rules.items():
        # Level 1: Dedicated regex matching (if exists)
        if 'regex_patterns' in rules:
            for pattern in rules['regex_patterns']:
                match = pattern.search(extracted_diagnosis)
                if match and keyword_positive(original_sentence, match.group(0)):
                    return pathology_name

        # Level 2: Fuzzy string matching
        answer_for_search = remove_punctuation(extracted_diagnosis.lower())
        for word in answer_for_search.split():
            if fuzz.ratio(word, pathology_name) > 90 and keyword_positive(original_sentence, word):
                return pathology_name

        # Level 3: Generic keyword combination matching
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
    Check if diagnosis matches target disease
    """
    normalized_diagnosis = _normalize_disease_name(diagnosis)
    mapped_diagnosis = map_pathology_name(diagnosis, normalized_diagnosis)
    mapped_diagnosis_lower = _normalize_disease_name(mapped_diagnosis)
    pathology_lower = pathology.lower()
    
    return mapped_diagnosis_lower == pathology_lower


def analyze_case(case_data: Dict[str, Any], pathology: str) -> Dict[str, Any]:
    """
    Analyze individual case diagnosis results
    
    Returns:
        Dictionary containing analysis results
    """
    case_id = case_data.get('case_id', 'Unknown')
    final_status = case_data.get('final_status', '')
    final_diagnosis = case_data.get('final_diagnosis', '')
    total_rounds = case_data.get('total_rounds', 0)
    
    is_consistent = '✅' in final_status or 'Consistent' in final_status
    
    # Determine at which round consistency was achieved
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
    
    # Unified check logic: record as any_match if any diagnosis matches pathology
    diagnoses = [d.strip() for d in final_diagnosis.split('|')]
    matches = [check_diagnosis_match(d, pathology) for d in diagnoses if d]
    result['any_match'] = any(matches) if matches else False
    
    return result


def analyze_disease_directory(disease_dir: Path, pathology: str) -> Optional[Dict[str, Any]]:
    """
    Analyze all cases in a single disease directory
    """
    summary_file = None
    for file in disease_dir.glob('*all_cases_summary.json'):
        summary_file = file
        break
    
    if not summary_file or not summary_file.exists():
        return None
    
    print(f"  Reading file: {summary_file}")
    with open(summary_file, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    
    total_cases = len(cases)
    consistent_cases = []
    inconsistent_cases = []
    
    # Count consistency by round
    round_1_consistent = 0
    round_2_consistent = 0
    round_3_consistent = 0
    
    # Analyze each case
    for case in cases:
        result = analyze_case(case, pathology)
        if result['is_consistent']:
            consistent_cases.append(result)
            # Count at which round consistency was achieved
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
    
    # Count matches in consistent cases (now changed to record as match if any one matches)
    consistent_match = sum(1 for c in consistent_cases if c.get('any_match', False))
    consistent_no_match = consistent_count - consistent_match
    
    # Count matches in inconsistent cases
    inconsistent_any_match = sum(1 for c in inconsistent_cases if c.get('any_match', False))
    inconsistent_no_match = inconsistent_count - inconsistent_any_match
    
    # Calculate effective opinion rate and ineffective opinion rate
    effective_opinions = consistent_match + inconsistent_any_match
    ineffective_opinions = consistent_no_match + inconsistent_no_match
    
    results = {
        'pathology': pathology,
        'total_cases': total_cases,
        # Dimension 1: Consistency between models
        'model_consistent': consistent_count,
        'model_inconsistent': inconsistent_count,
        # Dimension 2: Consistency with pathology (record as consistent if any one matches, otherwise as inconsistent)
        'pathology_consistent': effective_opinions,
        'pathology_inconsistent': ineffective_opinions,
        # Dimension 3: Statistics of consistent and effective
        'consistent_and_effective': consistent_match,
        'consistent_but_ineffective': consistent_no_match,
        # Dimension 4: Count consistency by round
        'round_1_consistent': round_1_consistent,
        'round_2_consistent': round_2_consistent,
        'round_3_consistent': round_3_consistent,
        # Keep old field compatibility
        'effective_opinions': effective_opinions,
        'ineffective_opinions': ineffective_opinions,
        'effective_opinion_rate': effective_opinions,
        'ineffective_opinion_rate': ineffective_opinions
    }
    
    # Calculate percentages (based on total cases)
    if total_cases > 0:
        results['model_consistent_percent'] = (consistent_count / total_cases) * 100
        results['model_inconsistent_percent'] = (inconsistent_count / total_cases) * 100
        results['pathology_consistent_percent'] = (effective_opinions / total_cases) * 100
        results['pathology_inconsistent_percent'] = (ineffective_opinions / total_cases) * 100
        results['consistent_and_effective_percent'] = (consistent_match / total_cases) * 100
        results['consistent_but_ineffective_percent'] = (consistent_no_match / total_cases) * 100
        results['effective_opinion_rate'] = (effective_opinions / total_cases) * 100
        results['ineffective_opinion_rate'] = (ineffective_opinions / total_cases) * 100
        # Percentage by round
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
    Main function: Analyze diagnosis results for all diseases
    """
    base_dir = Path('/data2/kunzhang/MIMIC-CDM/MIMIC-Clinical-Decision-Making-Framework-llama3.1_copy_copy/multi_round_results_human')
    
    if not base_dir.exists():
        print(f"Error: Directory does not exist {base_dir}")
        return
    
    print(f"Starting analysis of directory: {base_dir}\n")
    
    all_results = []
    
    # Get all disease subdirectories
    disease_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    for disease_dir in sorted(disease_dirs):
        pathology = disease_dir.name
        print(f"\nAnalyzing disease: {pathology}")
        print(f"{'='*60}")
        
        results = analyze_disease_directory(disease_dir, pathology)
        
        if results:
            all_results.append(results)
            print(f"  Total cases: {results['total_cases']}")
            print(f"  Consistent cases: {results['model_consistent']}")
            print(f"  Inconsistent cases: {results['model_inconsistent']}")
        else:
            print(f"  Warning: summary file not found")
    
    if not all_results:
        print("\nError: No data found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate overall statistics
    total_all_cases = df['total_cases'].sum()
    total_effective_opinions = df['effective_opinions'].sum()
    total_ineffective_opinions = df['ineffective_opinions'].sum()
    total_consistent_and_effective = df['consistent_and_effective'].sum()
    total_consistent_but_ineffective = df['consistent_but_ineffective'].sum()
    total_round_1_consistent = df['round_1_consistent'].sum()
    total_round_2_consistent = df['round_2_consistent'].sum()
    total_round_3_consistent = df['round_3_consistent'].sum()
    
    # Average effective opinion rate (average across all diseases)
    avg_effective_opinion_rate = df['effective_opinion_rate'].mean()
    # Percentage of effective opinions out of total cases
    effective_opinion_percent_of_all = (total_effective_opinions / total_all_cases) * 100 if total_all_cases > 0 else 0.0
    
    # Average ineffective opinion rate (average across all diseases)
    avg_ineffective_opinion_rate = df['ineffective_opinion_rate'].mean()
    # Percentage of ineffective opinions out of total cases
    ineffective_opinion_percent_of_all = (total_ineffective_opinions / total_all_cases) * 100 if total_all_cases > 0 else 0.0
    
    # Statistics of consistent and effective
    avg_consistent_and_effective_rate = df['consistent_and_effective_percent'].mean()
    consistent_and_effective_percent_of_all = (total_consistent_and_effective / total_all_cases) * 100 if total_all_cases > 0 else 0.0
    
    # Print detailed report
    print("\n" + "="*100)
    print("Experimental Results Summary Report")
    print("="*100 + "\n")
    
    for _, row in df.iterrows():
        print(f"\n{'='*80}")
        print(f"Disease: {row['pathology']}")
        print(f"{'='*80}")
        print(f"Total cases: {row['total_cases']}")
        
        print(f"\n[Evaluation Dimension 1: Inter-Model Diagnostic Consistency Statistics]")
        print(f"  Consistent: {row['model_consistent']} ({row['model_consistent_percent']:.2f}%)")
        print(f"  Inconsistent: {row['model_inconsistent']} ({row['model_inconsistent_percent']:.2f}%)")
        
        print(f"\n[Evaluation Dimension 2: Consistency with Pathology Statistics (Effectiveness Evaluation)]")
        print(f"  Effective (Matched/Effective): {row['pathology_consistent']} ({row['pathology_consistent_percent']:.2f}%)")
        print(f"  Ineffective (Unmatched/Ineffective): {row['pathology_inconsistent']} ({row['pathology_inconsistent_percent']:.2f}%)")
        
        print(f"\n[Evaluation Dimension 3: Model Consistent + Pathologically Effective Statistics]")
        print(f"  Consistent and effective: {row['consistent_and_effective']} ({row['consistent_and_effective_percent']:.2f}%)")
        print(f"  Consistent but ineffective: {row['consistent_but_ineffective']} ({row['consistent_but_ineffective_percent']:.2f}%)")
        
        print(f"\n[Evaluation Dimension 4: Consistency Statistics by Round]")
        print(f"  Round 1 consistent: {row['round_1_consistent']} ({row['round_1_consistent_percent']:.2f}%)")
        print(f"  Round 2 consistent: {row['round_2_consistent']} ({row['round_2_consistent_percent']:.2f}%)")
        print(f"  Round 3 consistent: {row['round_3_consistent']} ({row['round_3_consistent_percent']:.2f}%)")
    
    # Print overall statistics
    print(f"\n\n{'='*100}")
    print("Overall Statistics")
    print(f"{'='*100}")
    print(f"\nTotal cases for all diseases: {total_all_cases}")
    
    print(f"\n[Overall Effectiveness Statistics (Method 2 - Consistency with Pathology)]")
    print(f"  Average effective consistency rate across all diseases: {avg_effective_opinion_rate:.2f}%")
    print(f"  Percentage of effective consistent cases out of total: {total_effective_opinions}/{total_all_cases} ({effective_opinion_percent_of_all:.2f}%)")
    
    print(f"\n[Overall Ineffectiveness Statistics (Method 2 - Inconsistency with Pathology)]")
    print(f"  Average ineffective inconsistency rate across all diseases: {avg_ineffective_opinion_rate:.2f}%")
    print(f"  Percentage of ineffective inconsistent cases out of total: {total_ineffective_opinions}/{total_all_cases} ({ineffective_opinion_percent_of_all:.2f}%)")
    
    print(f"\n[Consistent and Effective Statistics (Model Consistent + Pathology Match)]")
    print(f"  Average consistent and effective rate across all diseases: {avg_consistent_and_effective_rate:.2f}%")
    print(f"  Percentage of consistent and effective cases out of total: {total_consistent_and_effective}/{total_all_cases} ({consistent_and_effective_percent_of_all:.2f}%)")
    print(f"  Percentage of consistent but ineffective cases out of total: {total_consistent_but_ineffective}/{total_all_cases} ({(total_consistent_but_ineffective / total_all_cases * 100) if total_all_cases > 0 else 0.0:.2f}%)")
    
    # Calculate consistency statistics by round
    avg_round_1_rate = df['round_1_consistent_percent'].mean()
    avg_round_2_rate = df['round_2_consistent_percent'].mean()
    avg_round_3_rate = df['round_3_consistent_percent'].mean()
    round_1_percent_of_all = (total_round_1_consistent / total_all_cases) * 100 if total_all_cases > 0 else 0.0
    round_2_percent_of_all = (total_round_2_consistent / total_all_cases) * 100 if total_all_cases > 0 else 0.0
    round_3_percent_of_all = (total_round_3_consistent / total_all_cases) * 100 if total_all_cases > 0 else 0.0
    
    print(f"\n[Consistency Statistics by Round]")
    print(f"  Round 1 consistent:")
    print(f"    - Average across all diseases: {avg_round_1_rate:.2f}%")
    print(f"    - Out of total cases: {total_round_1_consistent}/{total_all_cases} ({round_1_percent_of_all:.2f}%)")
    print(f"  Round 2 consistent:")
    print(f"    - Average across all diseases: {avg_round_2_rate:.2f}%")
    print(f"    - Out of total cases: {total_round_2_consistent}/{total_all_cases} ({round_2_percent_of_all:.2f}%)")
    print(f"  Round 3 consistent:")
    print(f"    - Average across all diseases: {avg_round_3_rate:.2f}%")
    print(f"    - Out of total cases: {total_round_3_consistent}/{total_all_cases} ({round_3_percent_of_all:.2f}%)")
    
    # Save CSV file
    output_csv = base_dir / 'analysis_summary.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n{'='*80}")
    print(f"Detailed results saved to: {output_csv}")
    
    # Save overall statistics
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
    print(f"Overall statistics saved to: {summary_csv}")
    
    
    print(f"{'='*80}\n")
    print("Analysis complete!")


if __name__ == '__main__':
    main()
