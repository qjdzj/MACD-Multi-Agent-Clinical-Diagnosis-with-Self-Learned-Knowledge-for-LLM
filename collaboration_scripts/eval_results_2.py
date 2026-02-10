import pandas as pd
import os
import re
import sys

CSV_DIRECTORY = 
OUTPUT_DIRECTORY = CSV_DIRECTORY


def analyze_diagnosis_correctness(df: pd.DataFrame, target_disease: str):
    """
    Apply diagnostic correctness judgment logic to the incoming DataFrame.
    
    New logic:
    Determine whether there is 【any】diagnosis name in the 'All_Diagnoses' string
    that contains the keyword of the target disease.
    """
    results = []
    
    # Prepare for case-insensitive comparison
    target_keyword = target_disease.lower()

    for index, row in df.iterrows():
        all_diagnoses_str = row.get('All_Diagnoses', '')

        # Check if the 'All_Diagnoses' cell is a valid string
        if not isinstance(all_diagnoses_str, str) or not all_diagnoses_str.strip():
            results.append(False)
            continue
            
        # Split the string by '|' and get all diagnosis results
        diagnoses = [d.strip() for d in all_diagnoses_str.split('|')]
        
        # Use the any() function to check if any diagnosis in the list contains the target keyword
        # any() will iterate through all diagnoses, and as long as one satisfies the condition (target_keyword in d.lower()), it returns True
        is_correct = any(target_keyword in d.lower() for d in diagnoses)
        results.append(is_correct)
            
    return results


def main(data_path, output_path):
    """
    Main function to perform file search, processing, statistics, and saving.
    (No modification needed for this function)
    """
    os.makedirs(output_path, exist_ok=True)

    # Find all CSV files that meet the criteria
    try:
        files_to_process = [f for f in os.listdir(data_path) if f.endswith('_combined_consistent.csv')]
    except OSError as e:
        print(f"Error: Cannot access directory {data_path}: {e}")
        sys.exit(1)

    if not files_to_process:
        print(f"Error: No `..._combined_consistent.csv` files found in directory '{data_path}'.")
        sys.exit(1)

    print(f"Found {len(files_to_process)} files to process in '{data_path}'...\n")

    overall_correct = 0
    overall_total = 0

    # Iterate and process each file
    for filename in files_to_process:
        full_file_path = os.path.join(data_path, filename)
        try:
            target_disease = filename.replace('_combined_consistent.csv', '')
            
            df = pd.read_csv(full_file_path)
            
            correctness_column = analyze_diagnosis_correctness(df, target_disease)
            df['is_diagnosis_correct'] = correctness_column
            
            correct_count = df['is_diagnosis_correct'].sum()
            total_count = len(df)
            overall_correct += correct_count
            overall_total += total_count
            
            percentage = (correct_count / total_count * 100) if total_count > 0 else 0
            
            print(f"Statistics: Correct diagnoses {correct_count} / Total cases {total_count} ({percentage:.2f}%)")
            
            output_filename = filename.replace('_combined_consistent.csv', '_analysis_results.csv')

            full_output_path = os.path.join(output_path, output_filename)
            df.to_csv(full_output_path, index=False, encoding='utf-8-sig')
            print(f"Analysis results saved to: {full_output_path}\n")

        except FileNotFoundError:
            print(f"Error: File {full_file_path} not found, skipping.")

    print("="*50)
    overall_percentage = (overall_correct / overall_total * 100) if overall_total > 0 else 0
    print(f"Total: Correct diagnoses {overall_correct} / Total cases {overall_total} ({overall_percentage:.2f}%)")
    print("="*50)


if __name__ == '__main__':
    main(CSV_DIRECTORY, OUTPUT_DIRECTORY)