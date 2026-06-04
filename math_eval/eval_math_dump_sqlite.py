import os
import json
import copy
import math
import sqlite3
import torch
import pandas as pd
import numpy as np

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from math_verify import parse, verify


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx: right_brace_idx + 1]


def remove_boxed(s):
    if s is None:
        return None
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left): -1]
    except Exception:
        return None


def apply_chat_template(toker, messages, chat_template=None, enable_thinking=False):
    if chat_template is None:
        return toker.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
    else:
        return chat_template.format(prompt=messages[0]["content"])


def ensure_obj(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()

    if isinstance(x, dict):
        return {k: ensure_obj(v) for k, v in x.items()}
    if isinstance(x, list):
        return [ensure_obj(v) for v in x]

    if isinstance(x, str):
        try:
            return ensure_obj(json.loads(x))
        except Exception:
            return x

    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)

    if isinstance(x, (np.bool_,)):
        return bool(x)

    return x


DEFAULT_CFG = {
    "model_name": None,
    "input_file": None,          # required: parquet
    "model_path": None,          # required
    "output_file": None,         # kept for compatibility, not used for jsonl anymore
    "db_path": None,             # required: sqlite db path
    "max_tokens": 9216,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "do_sample": True,
    "max_num_seqs": 128,
    "n": 1,
    "begin_idx": -1,
    "end_idx": -1,
    "seed": 42,
    "enable_thinking": False,
    "chat_template": None,
    "gpu_memory_utilization": 0.95,
    "tensor_parallel_size": None,  # None -> torch.cuda.device_count()
    "batch_size": 512,             # number of prompts per generation batch
}


def _merge_cfg(user_cfg: dict) -> dict:
    cfg = dict(DEFAULT_CFG)
    if user_cfg:
        cfg.update(user_cfg)

    for k in ["model_name", "input_file", "model_path"]:
        if not cfg.get(k):
            raise ValueError(f"Missing required config: `{k}`")

    if not cfg.get("db_path"):
        if cfg.get("output_file"):
            base, _ = os.path.splitext(cfg["output_file"])
            cfg["db_path"] = base + ".sqlite"
        else:
            raise ValueError("Missing required config: `db_path`")

    if cfg["tensor_parallel_size"] is None:
        cfg["tensor_parallel_size"] = torch.cuda.device_count()

    return cfg


def _load_parquet_as_records(path: str):
    df = pd.read_parquet(path)

    required_cols = {"prompt", "reward_model", "data_source", "extra_info"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}")

    def _ensure_obj_local(x):
        if isinstance(x, (dict, list)):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return x
        return x

    df["prompt"] = df["prompt"].apply(_ensure_obj_local)
    df["reward_model"] = df["reward_model"].apply(_ensure_obj_local)
    df["extra_info"] = df["extra_info"].apply(_ensure_obj_local)

    records = df.to_dict(orient="records")
    return records


def build_question_id(item):
    data_source = str(item.get("data_source", "unknown"))
    extra_info = item.get("extra_info", {})
    if not isinstance(extra_info, dict):
        extra_info = {}
    index = extra_info.get("index", None)
    if index is None:
        raise ValueError(f"Missing extra_info['index'] in item: {item}")
    return f"{data_source}_{index}"


def init_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS generations (
            question_id TEXT NOT NULL,
            rollout_id INTEGER NOT NULL,
            prompt TEXT NOT NULL,
            ground_truth TEXT,
            reward_style TEXT,
            response TEXT,
            pred_ans TEXT,
            acc INTEGER,
            model TEXT,
            PRIMARY KEY (question_id, rollout_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_generations_question_id
        ON generations(question_id)
        """
    )
    conn.commit()
    return conn


def get_existing_rollouts(conn, question_id):
    cur = conn.execute(
        "SELECT rollout_id FROM generations WHERE question_id = ?",
        (question_id,)
    )
    return {row[0] for row in cur.fetchall()}


def insert_generations(conn, rows):
    if not rows:
        return 0

    values = []
    for r in rows:
        values.append(
            (
                r["question_id"],
                r["rollout_id"],
                json.dumps(ensure_obj(r["prompt"]), ensure_ascii=False),
                None if r["ground_truth"] is None else str(r["ground_truth"]),
                None if r["reward_style"] is None else str(r["reward_style"]),
                r["response"],
                r["pred_ans"],
                None if r["acc"] is None else int(bool(r["acc"])),
                r["model"],
            )
        )

    before_changes = conn.total_changes

    with conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO generations (
                question_id, rollout_id, prompt, ground_truth, reward_style,
                response, pred_ans, acc, model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )

    inserted = conn.total_changes - before_changes
    return inserted


def compute_metrics_from_db(conn, model_name=None):
    if model_name is None:
        rows = conn.execute(
            """
            SELECT question_id, rollout_id, response, acc
            FROM generations
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT question_id, rollout_id, response, acc
            FROM generations
            WHERE model = ?
            """
            , (model_name,)
        ).fetchall()

    by_question = {}
    total_preds = 0
    correct_preds = 0
    total_resp_len = 0
    total_question_cnt_for_len = 0

    for question_id, rollout_id, response, acc in rows:
        by_question.setdefault(question_id, {"accs": [], "responses": []})
        by_question[question_id]["accs"].append(bool(acc) if acc is not None else False)
        if response is not None:
            by_question[question_id]["responses"].append(response)

        total_preds += 1
        if acc:
            correct_preds += 1

    pass_at_k_cnt = 0
    for question_id, info in by_question.items():
        if any(info["accs"]):
            pass_at_k_cnt += 1
        if info["responses"]:
            total_resp_len += sum(len(r) for r in info["responses"]) / len(info["responses"])
            total_question_cnt_for_len += 1

    accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
    pass_at_k = pass_at_k_cnt / len(by_question) if by_question else 0.0
    avg_length = total_resp_len / total_question_cnt_for_len if total_question_cnt_for_len > 0 else 0.0

    return {
        "total_questions_with_any_saved_rollout": len(by_question),
        "total_predictions": total_preds,
        "correct_predictions": correct_preds,
        "accuracy": accuracy,
        "pass_at_k": pass_at_k,
        "avg_char_length": avg_length,
    }


def main(cfg: dict):
    cfg = _merge_cfg(cfg)

    toker = AutoTokenizer.from_pretrained(cfg["model_path"])
    model_name = cfg["model_name"]

    llm = LLM(
        model=cfg["model_path"],
        tokenizer=cfg["model_path"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        tensor_parallel_size=cfg["tensor_parallel_size"],
        max_num_seqs=cfg["max_num_seqs"],
    )

    input_data = _load_parquet_as_records(cfg["input_file"])

    if cfg["begin_idx"] >= 0 and cfg["end_idx"] >= 0:
        input_data = input_data[cfg["begin_idx"]: cfg["end_idx"]]

    conn = init_db(cfg["db_path"])

    processed_items = []
    for item in input_data:
        qid = build_question_id(item)
        existing_rollouts = get_existing_rollouts(conn, qid)

        missing_rollouts = [rid for rid in range(cfg["n"]) if rid not in existing_rollouts]
        if not missing_rollouts:
            print(f"[Skip] {qid}: all {cfg['n']} rollouts already saved.")
            continue

        item_copy = copy.deepcopy(item)
        item_copy["_question_id"] = qid
        item_copy["_missing_rollouts"] = missing_rollouts
        processed_items.append(item_copy)

    pending_rollout_keys = set()
    for item in processed_items:
        for rollout_id in item["_missing_rollouts"]:
            pending_rollout_keys.add((item["_question_id"], rollout_id))

    print(f"Total input questions: {len(input_data)}")
    print(f"Questions needing generation: {len(processed_items)}")
    print(f"Rollout prompts needing generation: {len(pending_rollout_keys)}")
    print(f"Database path: {cfg['db_path']}")

    batch_size = cfg["batch_size"]

    sampling_params = SamplingParams(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_tokens=cfg["max_tokens"],
        n=1,
        seed=cfg["seed"],
        top_k=cfg["top_k"],
    )

    for batch_start in range(0, len(processed_items), batch_size):
        batch_items = processed_items[batch_start: batch_start + batch_size]
        if not batch_items:
            continue

        expanded_prompts = []
        expanded_meta = []

        for item in batch_items:
            messages = item["prompt"]
            prompt_text = apply_chat_template(
                toker,
                messages,
                chat_template=cfg["chat_template"],
                enable_thinking=cfg["enable_thinking"],
            )

            gt = item.get("reward_model", {}).get("ground_truth", None)
            if gt is None:
                raise ValueError(f"Missing reward_model.ground_truth at question_id={item['_question_id']}")
            reward_style = item.get("reward_model", {}).get("style", None)

            for rollout_id in item["_missing_rollouts"]:
                expanded_prompts.append(prompt_text)
                expanded_meta.append({
                    "question_id": item["_question_id"],
                    "rollout_id": rollout_id,
                    "prompt": messages,
                    "ground_truth": str(gt),
                    "reward_style": reward_style,
                })

        if not expanded_prompts:
            continue

        print(
            f"[Batch {batch_start // batch_size + 1}] "
            f"questions={len(batch_items)}, generations={len(expanded_prompts)}"
        )

        generations = llm.generate(expanded_prompts, sampling_params=sampling_params)

        batch_rows = []
        batch_rollout_keys = []
        batch_correct = 0
        batch_no_answer = 0

        for meta, gen in zip(expanded_meta, generations):
            if len(gen.outputs) != 1:
                raise ValueError(
                    f"Expected exactly 1 output per expanded prompt, got {len(gen.outputs)}"
                )

            response = gen.outputs[0].text.strip()
            boxed_answer = remove_boxed(last_boxed_only_string(response))

            if boxed_answer is None:
                acc = False
                batch_no_answer += 1
            else:
                try:
                    acc = verify(
                        parse("\\boxed{" + meta["ground_truth"] + "}"),
                        parse("\\boxed{" + boxed_answer + "}"),
                    )
                except Exception:
                    acc = False

            if acc:
                batch_correct += 1

            batch_rows.append(
                {
                    "question_id": meta["question_id"],
                    "rollout_id": meta["rollout_id"],
                    "prompt": meta["prompt"],
                    "ground_truth": meta["ground_truth"],
                    "reward_style": meta["reward_style"],
                    "response": response,
                    "pred_ans": boxed_answer,
                    "acc": acc,
                    "model": model_name,
                }
            )

            batch_rollout_keys.append((meta["question_id"], meta["rollout_id"]))

        inserted = insert_generations(conn, batch_rows)

        for key in batch_rollout_keys:
            pending_rollout_keys.discard(key)

        remaining_rollouts = len(pending_rollout_keys)
        remaining_questions = len({qid for qid, _ in pending_rollout_keys})

        batch_total = len(batch_rows)
        batch_acc = batch_correct / batch_total if batch_total > 0 else 0.0

        print(
            f"[Inserted Batch {batch_start // batch_size + 1}] "
            f"attempted={batch_total}, "
            f"inserted={inserted}, "
            f"correct={batch_correct}, "
            f"acc={batch_acc:.4f}, "
            f"no_boxed_answer={batch_no_answer}, "
            f"remaining_rollout_prompts={remaining_rollouts}, "
            f"remaining_questions={remaining_questions}, "
            f"model={cfg['model_name']}"
        )

    metrics = compute_metrics_from_db(conn, model_name=model_name)

    print(f"dataset: {cfg['input_file']}")
    print(f"db_path: {cfg['db_path']}")
    print(f"Total predictions: {metrics['total_predictions']}")
    print(f"Accurate predictions: {metrics['correct_predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"pass@k: {metrics['pass_at_k']:.4f}")
    print(f"avg_char_length: {metrics['avg_char_length']:.4f}")

    conn.close()


if __name__ == "__main__":
    model_name = 'Skywork-OR1-Math-7B'  # Skywork-OR1-Math-7B, DeepSeek-R1-Distill-Qwen-7B
    if 'deepseek' in model_name.lower():
        enable_thinking = 'r1' in model_name.lower()
    elif 'skywork' in model_name.lower():
        enable_thinking = 'r1' in model_name.lower()
    else:
        enable_thinking = 'thinking' in model_name.lower()
    

    dataset_name = 'DeepMath-103K'
    split_name = 'train_80_percent'   # train_filtered_level6, val_1000
    num_outputs = 16

    print('=' * 30)
    print(model_name, enable_thinking, num_outputs)
    print('=' * 30)

    cfg = {
        "model_name": model_name,
        "input_file": f"/mnt/phwfile/datafrontier/fudaocheng/datasets/G-OPD-Training-Data/{dataset_name}/{split_name}.parquet",
        "model_path": f"/mnt/phwfile/datafrontier/public_models/{model_name}",
        "db_path": f"/mnt/phwfile/datafrontier/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/{split_name}_{model_name}_pass@{num_outputs}.sqlite",
        "n": num_outputs,
        "enable_thinking": enable_thinking,
        "batch_size": 128,
    }
    main(cfg)