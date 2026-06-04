import os
import json
import copy
import time
import sqlite3
import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from math_verify import parse, verify


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i, right_brace_idx, num_left_braces_open = idx, None, 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx:right_brace_idx + 1]


def remove_boxed(s):
    if s is None:
        return None
    left = "\\boxed{"
    return s[len(left):-1] if s.startswith(left) and s.endswith("}") else None


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
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


DEFAULT_CFG = {
    "model_name": None,
    "input_file": None,
    "output_file": None,
    "db_path": None,
    "base_url": None,
    "api_key": None,
    "max_tokens": 9216,
    "temperature": 1.0,
    "top_p": 1.0,
    "n": 1,
    "begin_idx": -1,
    "end_idx": -1,
    "seed": 8962,
    "batch_size": 512,
    "max_workers": 32,
    "timeout": 300,
    "max_retries": 8,
    "retry_sleep": 2,
    "extra_body": None,
}


def _merge_cfg(user_cfg):
    cfg = dict(DEFAULT_CFG)
    if user_cfg:
        cfg.update(user_cfg)

    cfg["api_key"] = cfg["api_key"] or os.getenv("OPENAI_API_KEY")
    cfg["base_url"] = cfg["base_url"] or os.getenv("OPENAI_BASE_URL")

    for k in ["model_name", "input_file", "api_key", "base_url"]:
        if not cfg.get(k):
            raise ValueError(f"Missing required config: `{k}`")

    if not cfg.get("db_path"):
        if cfg.get("output_file"):
            cfg["db_path"] = os.path.splitext(cfg["output_file"])[0] + ".sqlite"
        else:
            raise ValueError("Missing required config: `db_path`")

    return cfg


def _load_parquet_as_records(path):
    df = pd.read_parquet(path)
    required_cols = {"prompt", "reward_model", "data_source", "extra_info"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}")

    def load(x):
        if isinstance(x, (dict, list)):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return x
        return x

    df["prompt"] = df["prompt"].apply(load)
    df["reward_model"] = df["reward_model"].apply(load)
    df["extra_info"] = df["extra_info"].apply(load)
    return df.to_dict(orient="records")


def build_question_id(item):
    data_source = str(item.get("data_source", "unknown"))
    extra_info = item.get("extra_info", {})
    if not isinstance(extra_info, dict):
        extra_info = {}
    index = extra_info.get("index")
    if index is None:
        raise ValueError(f"Missing extra_info['index'] in item: {item}")
    return f"{data_source}_{index}"


def init_db(db_path):
    dirname = os.path.dirname(db_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
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
    rows = conn.execute(
        "SELECT rollout_id FROM generations WHERE question_id = ?",
        (question_id,),
    ).fetchall()
    return {r[0] for r in rows}


def insert_generation(conn, question_id, rollout_id, prompt, ground_truth, reward_style, response, pred_ans, acc, model):
    conn.execute(
        """
        INSERT OR IGNORE INTO generations (
            question_id, rollout_id, prompt, ground_truth, reward_style,
            response, pred_ans, acc, model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            question_id,
            rollout_id,
            json.dumps(ensure_obj(prompt), ensure_ascii=False),
            None if ground_truth is None else str(ground_truth),
            None if reward_style is None else str(reward_style),
            response,
            pred_ans,
            None if acc is None else int(bool(acc)),
            model,
        ),
    )
    conn.commit()


def compute_metrics_from_db(conn, model_name=None):
    if model_name is None:
        rows = conn.execute(
            "SELECT question_id, rollout_id, response, acc FROM generations"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT question_id, rollout_id, response, acc FROM generations WHERE model = ?",
            (model_name,),
        ).fetchall()

    by_question = {}
    total_preds = correct_preds = 0
    total_resp_len = total_question_cnt_for_len = 0

    for question_id, _, response, acc in rows:
        by_question.setdefault(question_id, {"accs": [], "responses": []})
        by_question[question_id]["accs"].append(bool(acc) if acc is not None else False)
        if response is not None:
            by_question[question_id]["responses"].append(response)
        total_preds += 1
        correct_preds += int(bool(acc))

    pass_at_k_cnt = 0
    for info in by_question.values():
        pass_at_k_cnt += int(any(info["accs"]))
        if info["responses"]:
            total_resp_len += sum(len(r) for r in info["responses"]) / len(info["responses"])
            total_question_cnt_for_len += 1

    return {
        "total_questions_with_any_saved_rollout": len(by_question),
        "total_predictions": total_preds,
        "correct_predictions": correct_preds,
        "accuracy": correct_preds / total_preds if total_preds else 0.0,
        "pass_at_k": pass_at_k_cnt / len(by_question) if by_question else 0.0,
        "avg_char_length": total_resp_len / total_question_cnt_for_len if total_question_cnt_for_len else 0.0,
    }


def normalize_messages(prompt):
    prompt = ensure_obj(prompt)
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, list):
        return prompt
    if isinstance(prompt, dict):
        return [prompt]
    raise ValueError(f"Unsupported prompt format: {type(prompt)}")


def content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, dict):
                parts.append(str(x.get("text", x.get("content", ""))))
            else:
                parts.append(str(x))
        return "".join(parts)
    return str(content)


def call_openai(client, cfg, messages, rollout_id):
    params = {
        "model": cfg["model_name"],
        "messages": normalize_messages(messages),
        "max_tokens": cfg["max_tokens"],
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "n": 1,
    }
    if cfg["seed"] is not None:
        params["seed"] = int(cfg["seed"]) + int(rollout_id)
    if cfg["extra_body"]:
        params["extra_body"] = cfg["extra_body"]

    last_err = None
    for i in range(cfg["max_retries"]):
        try:
            resp = client.chat.completions.create(**params)
            return content_to_text(resp.choices[0].message.content).strip()
        except Exception as e:
            last_err = e
            time.sleep(cfg["retry_sleep"] * (2 ** min(i, 5)))
    raise last_err


def score_response(response, ground_truth):
    boxed_answer = remove_boxed(last_boxed_only_string(response))
    if boxed_answer is None:
        return boxed_answer, False
    try:
        acc = verify(
            parse("\\boxed{" + str(ground_truth) + "}"),
            parse("\\boxed{" + str(boxed_answer) + "}"),
        )
    except Exception:
        acc = False
    return boxed_answer, acc


def main(cfg):
    cfg = _merge_cfg(cfg)
    model_name = cfg["model_name"]
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=cfg["timeout"], max_retries=0)

    input_data = _load_parquet_as_records(cfg["input_file"])
    if cfg["begin_idx"] >= 0 and cfg["end_idx"] >= 0:
        input_data = input_data[cfg["begin_idx"]:cfg["end_idx"]]

    conn = init_db(cfg["db_path"])
    processed_items = []

    for item in input_data:
        qid = build_question_id(item)
        existing = get_existing_rollouts(conn, qid)
        missing = [rid for rid in range(cfg["n"]) if rid not in existing]
        if not missing:
            print(f"[Skip] {qid}: all {cfg['n']} rollouts already saved.")
            continue
        item = copy.deepcopy(item)
        item["_question_id"] = qid
        item["_missing_rollouts"] = missing
        processed_items.append(item)

    print(f"Total input questions: {len(input_data)}")
    print(f"Questions needing generation: {len(processed_items)}")
    print(f"Database path: {cfg['db_path']}")

    for batch_start in range(0, len(processed_items), cfg["batch_size"]):
        batch_items = processed_items[batch_start:batch_start + cfg["batch_size"]]
        jobs = []

        for item in batch_items:
            gt = item.get("reward_model", {}).get("ground_truth")
            if gt is None:
                raise ValueError(f"Missing reward_model.ground_truth at question_id={item['_question_id']}")
            reward_style = item.get("reward_model", {}).get("style")
            for rollout_id in item["_missing_rollouts"]:
                jobs.append({
                    "question_id": item["_question_id"],
                    "rollout_id": rollout_id,
                    "prompt": item["prompt"],
                    "ground_truth": str(gt),
                    "reward_style": reward_style,
                })

        if not jobs:
            continue

        print(f"[Batch {batch_start // cfg['batch_size'] + 1}] questions={len(batch_items)}, generations={len(jobs)}")

        with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as ex:
            futures = {
                ex.submit(call_openai, client, cfg, job["prompt"], job["rollout_id"]): job
                for job in jobs
            }

            for fut in as_completed(futures):
                meta = futures[fut]
                try:
                    response = fut.result()
                    pred_ans, acc = score_response(response, meta["ground_truth"])
                    insert_generation(
                        conn=conn,
                        question_id=meta["question_id"],
                        rollout_id=meta["rollout_id"],
                        prompt=meta["prompt"],
                        ground_truth=meta["ground_truth"],
                        reward_style=meta["reward_style"],
                        response=response,
                        pred_ans=pred_ans,
                        acc=acc,
                        model=model_name,
                    )
                    print(f"[Saved] question_id={meta['question_id']} rollout_id={meta['rollout_id']} acc={acc}")
                except Exception as e:
                    print(f"[Error] question_id={meta['question_id']} rollout_id={meta['rollout_id']} error={repr(e)}")

    metrics = compute_metrics_from_db(conn, model_name=model_name)

    print(f"dataset: {cfg['input_file']}")
    print(f"db_path: {cfg['db_path']}")
    print(f"Total predictions: {metrics['total_predictions']}")
    print(f"Accurate predictions: {metrics['correct_predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"pass@k: {metrics['pass_at_k']:.4f}")
    print(f"avg_char_length: {metrics['avg_char_length']:.4f}")

    conn.close()


if __name__ == "__main__":
    model_name = "gpt-5.1"
    dataset_name = "MathTestTotal"
    num_outputs = 8

    print("=" * 30)
    print(model_name, num_outputs)
    print("=" * 30)

    cfg = {
        "model_name": model_name,
        "base_url": "http://35.220.164.252:3888/v1",
        "api_key": "sk-LCNRSkN5fnAsRTJ8a5VUvyQznlWR2LJEpVCAoRhhodxx8Ls2",
        "input_file": f"/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/{dataset_name}/test.parquet",
        "db_path": f"/mnt/petrelfs/fudaocheng/codes/G-OPD/eval_outputs/{model_name}_{dataset_name}_pass@{num_outputs}.sqlite",
        "n": num_outputs,
        "batch_size": 8,
        "max_workers": 4,
    }
    main(cfg)