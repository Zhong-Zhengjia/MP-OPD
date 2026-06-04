import numpy as np
import pandas as pd
from rich import print

from typing import Dict, List, Set

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 0)

original_data = '/mnt/phwfile/datafrontier/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet'

output_train_path = '/mnt/phwfile/datafrontier/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/train_80_percent.parquet'
output_val_path = '/mnt/phwfile/datafrontier/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/val_1000.parquet'

seed = 42

df = pd.read_parquet(original_data)

df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

train_size = int(len(df) * 0.8)

train_df = df.iloc[:train_size].reset_index(drop=True)
remaining_df = df.iloc[train_size:].reset_index(drop=True)

val_df = remaining_df.sample(n=1000, random_state=seed).reset_index(drop=True)

train_df.to_parquet(output_train_path, index=False)
val_df.to_parquet(output_val_path, index=False)

print(f"Original data size: {len(df)}")
print(f"Train data size: {len(train_df)}")
print(f"Remaining data size: {len(remaining_df)}")
print(f"Validation data size: {len(val_df)}")
print(f"Train saved to: {output_train_path}")
print(f"Validation saved to: {output_val_path}")