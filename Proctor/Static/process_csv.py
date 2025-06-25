# %%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv("Proctor/Datasets/raw_proctor_results.csv")

# %%
df

# %%
df.dtypes.sort_values()

# %%
drop_cols = [
    "H-Hand Detected",
    "F-Hand Detected",
    "face_is_cheating",
    "Cheat Score",
    "hand_is_cheating",
]
df1 = df.drop(columns=drop_cols)

# %%
df1.dtypes.sort_values()

# %%
df1.verification_result = df1.verification_result.astype(np.int64)

# %%
df1.dtypes.sort_values()

# %%
def convert_timestamp(ts):
    parts = ts.split('-')
    
    if len(parts) == 4:  # Format: 0-01-06-066617
        _, hh, mm, sec = parts
    elif len(parts) == 3:  # Format: 0-07-13.866465
        _, hh, rest = parts
        mm, sec = rest.split('.')
    else:
        return None  # Handle unexpected formats

    # Convert to HH:MM:SS format
    return f"{int(hh):02}:{int(mm):02}:{int(float(sec)):02}"


# %%
df1["timestamp"] = pd.to_timedelta(df1['timestamp'].apply(convert_timestamp)).dt.total_seconds()

# %%
obj_cols = df1.select_dtypes(include=[object]).columns
obj_cols

# %%
df1["H-Prohibited Item"].unique()

# %%
all_objects = {'cell phone', 'chits', 'closedbook', 'earpiece', 'headphone', 'openbook', 'sheet', 'watch'}

def one_hot_encode(row, all_objects):
    observed = set(row['F-Prohibited Item']) | set(row['H-Prohibited Item'])  # Merge both columns
    return {obj: int(obj in observed) for obj in all_objects}

# %%
one_hot_df = df1.apply(lambda row: one_hot_encode(row, all_objects), axis=1)

# %%
one_hot_df = pd.DataFrame(one_hot_df.tolist())

# %%
df2 = pd.concat([df1, one_hot_df], axis=1)

# %%
df2.drop(columns=['F-Prohibited Item', 'H-Prohibited Item', 'H-Illegal Objects', "F-Illegal Objects"], inplace=True)

# %%
df2

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
df2.dtypes.sort_values()

# %%
df2.isna().sum()

# %%
# nan mapping dictionary
nan_mappings = {
    'iris_pos': -1,
    'mouth_zone': -1,
    'gaze_direction': -1,
    'gaze_zone': -1,
    "H-Distance": 1000,
    "F-Distance": 1000,
}

# %%
df2.fillna(nan_mappings, inplace=True)

# %%
df2.is_cheating.value_counts()

# %%
df2.columns

# %%
columns = [
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
    'is_cheating'
]

# %%
df3 = df2[columns]

# %%
# df3.to_csv("Proctor/Datasets/proctor_results_final_cleaned.csv", index=False)

# %%
df3["is_cheating"].value_counts()

# %%
df3 = df3.sort_values(by='timestamp')

# %%
df3.head()

# %%
df3.to_csv("Proctor/Datasets/training_proctor_results.csv", index=False)

# %%



