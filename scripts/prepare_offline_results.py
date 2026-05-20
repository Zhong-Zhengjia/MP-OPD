import os
import sqlite3
import json
from collections import defaultdict
from rich import print

model_name = 'Qwen3-1.7B'
sample_nums = 16

db_path = f"/mnt/petrelfs/fudaocheng/codes/G-OPD/eval_outputs/{model_name}_DeepMath-103K_pass@{sample_nums}.sqlite"
output_path = f"/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/offline_rollout_results/{model_name}_DeepMath-103K_pass@{sample_nums}.json"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# 读取每个 question_id 的 ground_truth（取任意一条非空即可）
cur.execute("""
    SELECT question_id, ground_truth
    FROM generations
    WHERE ground_truth IS NOT NULL AND ground_truth != ''
""")
gt_map = {}
for row in cur.fetchall():
    qid = row["question_id"]
    if qid not in gt_map:
        gt_map[qid] = row["ground_truth"]

# 读取 acc（0/1）作为 teacher_rewards
cur.execute("""
    SELECT question_id, rollout_id, acc
    FROM generations
    ORDER BY question_id ASC, rollout_id ASC
""")

data = defaultdict(lambda: {"teacher_rewards": [], "ground_truth": ""})

missing_acc = 0
for row in cur.fetchall():
    qid = row["question_id"]
    acc = row["acc"]
    if acc is None:
        missing_acc += 1
        continue
    data[qid]["teacher_rewards"].append(int(acc))

# 写入 ground_truth
missing_gt = 0
for qid in data.keys():
    gt = gt_map.get(qid, "")
    if not gt:
        missing_gt += 1
    data[qid]["ground_truth"] = gt

conn.close()

# 转成普通 dict 并保存
out_obj = dict(data)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(out_obj, f, ensure_ascii=False, indent=2)

# 简单统计输出
num_q = len(out_obj)
num_rewards = sum(len(v["teacher_rewards"]) for v in out_obj.values())
has_correct = sum(1 for v in out_obj.values() if any(r > 0 for r in v["teacher_rewards"]))
all_wrong = sum(1 for v in out_obj.values() if v["teacher_rewards"] and all(r == 0 for r in v["teacher_rewards"]))
empty_rewards = sum(1 for v in out_obj.values() if len(v["teacher_rewards"]) == 0)

print({
    "db_path": db_path,
    "output_path": output_path,
    "num_questions": num_q,
    "num_total_rewards": num_rewards,
    "avg_rollouts_per_question": (num_rewards / num_q) if num_q else 0.0,
    "questions_has_correct": has_correct,
    "questions_all_wrong": all_wrong,
    "questions_empty_rewards": empty_rewards,
    "missing_ground_truth_questions": missing_gt,
    "missing_acc_rows": missing_acc,
})
print("Saved example (first 1 item):")
if out_obj:
    first_k = next(iter(out_obj.keys()))
    print({first_k: out_obj[first_k]})

