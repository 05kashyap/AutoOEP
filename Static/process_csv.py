# %%
import pandas as pd
import numpy as np
import os
import glob

# %%
def convert_timestamp(ts):
    """Convert timestamp from various formats to seconds"""
    # Handle colon-separated format (0:00:00.100000)
    if ':' in ts:
        parts = ts.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
    
    # Handle hyphen-separated formats
    parts = ts.split('-')
    
    if len(parts) == 4:  # Format: 0-01-06-066617
        _, hh, mm, sec = parts
    elif len(parts) == 3:  # Format: 0-07-13.866465
        _, hh, rest = parts
        mm, sec = rest.split('.')
    else:
        return None  # Handle unexpected formats

    # Convert to seconds
    return int(hh) * 3600 + int(mm) * 60 + float(sec)

def process_single_csv(input_path, output_path):
    """Process a single CSV file"""
    print(f"Processing {os.path.basename(input_path)}...")
    
    # %%
    df = pd.read_csv(input_path)
    
    # %%
    drop_cols = [
        "H-Hand Detected",
        "F-Hand Detected", 
        "face_is_cheating",
        "Cheat Score",
        "hand_is_cheating",
    ]
    # Only drop columns that exist
    drop_cols = [col for col in drop_cols if col in df.columns]
    df1 = df.drop(columns=drop_cols)
    
    # %%
    if 'verification_result' in df1.columns:
        df1.verification_result = df1.verification_result.astype(np.int64)
    
    # %%
    if 'timestamp' in df1.columns:
        df1["timestamp"] = df1['timestamp'].apply(convert_timestamp)
    
    # %%
    all_objects = {'cell phone', 'chits', 'closedbook', 'earpiece', 'headphone', 'openbook', 'sheet', 'watch'}

    def one_hot_encode(row, all_objects):
        observed = set()
        if 'F-Prohibited Item' in row and pd.notna(row['F-Prohibited Item']):
            observed.update(set(row['F-Prohibited Item']))
        if 'H-Prohibited Item' in row and pd.notna(row['H-Prohibited Item']):
            observed.update(set(row['H-Prohibited Item']))
        return {obj: int(obj in observed) for obj in all_objects}
    
    # %%
    if 'F-Prohibited Item' in df1.columns or 'H-Prohibited Item' in df1.columns:
        one_hot_df = df1.apply(lambda row: one_hot_encode(row, all_objects), axis=1)
        one_hot_df = pd.DataFrame(one_hot_df.tolist())
        df2 = pd.concat([df1, one_hot_df], axis=1)
        
        # Drop original prohibited item columns
        prohibited_cols = ['F-Prohibited Item', 'H-Prohibited Item', 'H-Illegal Objects', "F-Illegal Objects"]
        prohibited_cols = [col for col in prohibited_cols if col in df2.columns]
        df2.drop(columns=prohibited_cols, inplace=True)
    else:
        df2 = df1.copy()
        # Add missing one-hot columns with zeros
        for obj in all_objects:
            df2[obj] = 0
    
    # %%
    mappings = {
        'iris_pos': {'center': 0, 'left': 1, 'right': 2},
        'mouth_zone': {'GREEN': 0, 'YELLOW': 1, 'ORANGE': 2, 'RED': 3},
        'gaze_direction': {'forward': 0, 'left': 1, 'right': 2, 'up': 3, 'down': 4},
        'gaze_zone': {'white': 0, 'yellow': 1, 'red': 2}
    }
    
    # %%
    df2 = df2.replace(mappings)
    
    # %%
    # nan mapping dictionary
    nan_mappings = {
        'iris_pos': -1,
        'mouth_zone': -1,
        'gaze_direction': -1,
        'gaze_zone': -1,
        "H-Distance": 10000,
        "F-Distance": 10000,
    }
    
    # %%
    df2.fillna(nan_mappings, inplace=True)
    
    # %%
    # Define desired column order
    desired_columns = [
        'timestamp',
        'verification_result',
        'num_faces',
        'iris_pos', 
        'iris_ratio', 
        'mouth_zone', 
        'mouth_area',
        'x_rotation', 
        'y_rotation', 
        'z_rotation', 
        'radial_distance',
        'gaze_direction', 
        'gaze_zone',
        'watch', 
        'headphone', 
        'closedbook', 
        'earpiece', 
        'cell phone',
        'openbook', 
        'chits', 
        'sheet',
        'H-Distance',
        'F-Distance',
        'split',
        'video',
        'is_cheating'
    ]
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in desired_columns if col in df2.columns]
    df3 = df2[available_columns]

    # --- DEBUG: Print missing/extra columns ---
    missing = set(desired_columns) - set(df2.columns)
    extra = set(df2.columns) - set(desired_columns)
    if missing:
        print(f"WARNING: Missing columns in processed CSV: {missing}")
    if extra:
        print(f"NOTE: Extra columns in processed CSV: {extra}")

    # %%
    df3 = df3.sort_values(by='timestamp')
    
    # %%
    df3.to_csv(output_path, index=False)
    print(f"Saved processed data to {os.path.basename(output_path)}")
    print(f"  Shape: {df3.shape}")
    if 'is_cheating' in df3.columns:
        print(f"  Cheating distribution: {df3['is_cheating'].value_counts().to_dict()}")
    
    return df3

def process_all_csvs(results_dir, output_dir):
    """Process all CSV files from save_features.py results"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files (exclude combined file to avoid processing twice)
    csv_files = glob.glob(os.path.join(results_dir, "*_proctor_results.csv"))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith('combined_')]
    
    combined_csv = os.path.join(results_dir, "combined_proctor_results.csv")
    if os.path.exists(combined_csv):
        csv_files.append(combined_csv)
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    all_processed_data = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        # Create output filename
        if filename.startswith('combined_'):
            output_filename = "combined_processed_proctor_results.csv"
        else:
            # Replace _proctor_results.csv with _processed.csv
            output_filename = filename.replace('_proctor_results.csv', '_processed.csv')
        
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            processed_df = process_single_csv(csv_file, output_path)
            
            # Collect for combined processing (exclude the already combined file)
            if not filename.startswith('combined_'):
                all_processed_data.append(processed_df)
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Create a new combined processed file
    if all_processed_data:
        combined_processed = pd.concat(all_processed_data, ignore_index=True)
        combined_processed = combined_processed.sort_values(by=['split', 'video', 'timestamp'])
        
        combined_output_path = os.path.join(output_dir, "final_combined_processed.csv")
        combined_processed.to_csv(combined_output_path, index=False)
        
        print(f"\nCreated final combined processed file: final_combined_processed.csv")
        print(f"  Total shape: {combined_processed.shape}")
        if 'is_cheating' in combined_processed.columns:
            print(f"  Overall cheating distribution: {combined_processed['is_cheating'].value_counts().to_dict()}")

if __name__ == "__main__":
    # Configuration
    results_dir = "Proctor/results"  # Directory containing CSV files from save_features.py
    output_dir = "processed_results"    # Directory for processed CSV files
    
    # Process all CSV files
    process_all_csvs(results_dir, output_dir)



