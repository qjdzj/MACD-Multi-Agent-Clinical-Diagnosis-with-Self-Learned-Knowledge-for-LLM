import pandas as pd
import os
import sys
from collections import defaultdict

base_path = 
output_path = base_path

def merge_by_disease(base_path, output_path):
    """
    Group and merge CSV files by disease name, and save the results in the specified output path.
    Merge all extracted data by disease name, and generate a separate output CSV file for each disease.
    """
    data_by_disease = defaultdict(list)

    os.makedirs(output_path, exist_ok=True)

    for round_folder in sorted(os.listdir(base_path)):
        round_path = os.path.join(base_path, round_folder)
        
        # Only process directories named round1, round2, round3
        if os.path.isdir(round_path) and round_folder in ['round1', 'round2', 'round3']:
            for filename in sorted(os.listdir(round_path)):
                if filename.endswith('.csv'):
                    try:
                        disease_name = filename.split('_')[0]
                    except IndexError:
                        print(f"  - Warning: Unable to extract disease name from {filename}, skipped.")
                        continue

                    file_path = os.path.join(round_path, filename)
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')

                        data_to_add = None
                        
                        # Rule 1: For round1 and round2, only extract consistent results
                        if round_folder in ['round1', 'round2']:
                            if 'Status' in df.columns:
                                data_to_add = df[df['Status'] == '✅ Consistent'].copy()
                                print(f"  - Found {len(data_to_add)} rows in {filename} based on [Consistent] rule")
                            else:
                                print(f"  - Warning: 'Status' column not found in {filename}, skipped.")
                        
                        # Rule 2: For round3, extract all results
                        elif round_folder == 'round3':
                            data_to_add = df.copy()
                            print(f"  - Found {len(data_to_add)} rows in {filename} based on [Extract All] rule")
                        
                        # Add the extracted data to the corresponding disease list
                        if data_to_add is not None and not data_to_add.empty:
                            data_by_disease[disease_name].append(data_to_add)

                    except Exception as e:
                        print(f"  - Error: An error occurred while processing file {filename}: {e}")

    if not data_by_disease:
        print("\nProcessing complete, but no data available for merging was found.")
        return


    for disease_name, dfs_list in data_by_disease.items():
        if not dfs_list:
            continue

        combined_df = pd.concat(dfs_list, ignore_index=True)
        
        # Note: The output filename can be adjusted as needed, currently remains unchanged
        output_filename = f"{disease_name}_combined_consistent.csv"
        full_output_path = os.path.join(output_path, output_filename)

        try:
            combined_df.to_csv(full_output_path, index=False, encoding='utf-8-sig')
            print(f"✅ Success: {len(combined_df)} rows of data for '{disease_name}' have been saved to '{full_output_path}'")
        except Exception as e:
            print(f"❌ Failure: An error occurred while saving data for '{disease_name}' to '{full_output_path}': {e}")

if __name__ == '__main__':
    merge_by_disease(base_path, output_path)