import json
import pandas as pd

model_name = 'Qwen3-1.7B'
sample_nums = 16

p_file = "/mnt/phwfile/datafrontier/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet"
j_file = f"/mnt/phwfile/datafrontier/fudaocheng/datasets/G-OPD-Training-Data/offline_rollout_results/{model_name}_DeepMath-103K_pass@{sample_nums}.json"

df = pd.read_parquet(p_file)
print(len(df))

print(df.columns)

with open(j_file, 'r', encoding='utf-8') as rf:
    data = json.load(rf)    


print(len(data))
print(data['DeepMath-103K_0'])