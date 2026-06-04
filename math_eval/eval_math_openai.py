import os
import json
import copy
import time
import random
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
    return string[idx:right_brace_idx + 1]


def remove_boxed(s):
    if s is None:
        return None
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except Exception:
        return None


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
    return x


def count_tokens(text):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(text)


DEFAULT_CFG = {
    "model_name": None,
    "input_file": None,
    "output_file": None,
    "base_url": None,
    "api_key": None,
    "max_tokens": 8192,
    "temperature": 1.0,
    "top_p": 1.0,
    "n": 1,
    "begin_idx": -1,
    "end_idx": -1,
    "seed": 42,
    "max_workers": 16,
    "max_retries": 5,
    "retry_sleep": 2.0,
    "timeout": 300,
    "extra_body": None,
    "token_param": "max_tokens",
}


def _merge_cfg(user_cfg):
    cfg = dict(DEFAULT_CFG)
    if user_cfg:
        cfg.update(user_cfg)

    for k in ["model_name", "input_file", "output_file"]:
        if not cfg.get(k):
            raise ValueError(f"Missing required config: `{k}`")

    cfg["api_key"] = cfg["api_key"] or os.getenv("OPENAI_API_KEY")
    cfg["base_url"] = cfg["base_url"] or os.getenv("OPENAI_BASE_URL")

    if not cfg["api_key"]:
        raise ValueError("Missing required config: `api_key` or env OPENAI_API_KEY")
    if not cfg["base_url"]:
        raise ValueError("Missing required config: `base_url` or env OPENAI_BASE_URL")

    return cfg


def _load_parquet_as_records(path):
    df = pd.read_parquet(path)

    required_cols = {"prompt", "reward_model"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}")

    df["prompt"] = df["prompt"].apply(ensure_obj)
    df["reward_model"] = df["reward_model"].apply(ensure_obj)
    return df.to_dict(orient="records")


def normalize_messages(messages):
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if isinstance(messages, list):
        return messages
    raise ValueError(f"Invalid prompt format: {type(messages)}")


def call_openai(client, cfg, messages):
    kwargs = {
        "model": cfg["model_name"],
        "messages": normalize_messages(messages),
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "n": cfg["n"],
        "seed": cfg["seed"],
        cfg["token_param"]: cfg["max_tokens"],
        "timeout": cfg["timeout"],
    }

    if cfg.get("extra_body") is not None:
        kwargs["extra_body"] = cfg["extra_body"]

    for t in range(cfg["max_retries"]):
        try:
            resp = client.chat.completions.create(**kwargs)
            return [(c.message.content or "").strip() for c in resp.choices]
        except Exception:
            if t == cfg["max_retries"] - 1:
                raise
            time.sleep(cfg["retry_sleep"] * (2 ** t) + random.random())


def process_one(idx, item, responses, model_name):
    d = copy.deepcopy(item)

    gt = d.get("reward_model", {}).get("ground_truth", None)
    if gt is None:
        raise ValueError(f"Missing reward_model.ground_truth at row {idx}")
    gt = str(gt)

    boxed_answers = []
    acc_list = []

    for response in responses:
        boxed_answer = remove_boxed(last_boxed_only_string(response))
        boxed_answers.append(boxed_answer)

        if boxed_answer is None:
            acc = False
        else:
            acc = verify(
                parse("\\boxed{" + gt + "}"),
                parse("\\boxed{" + boxed_answer + "}"),
            )
        acc_list.append(bool(acc))

    d["pred_answers"] = boxed_answers
    d["responses"] = responses
    d["acc_list"] = acc_list
    d["model"] = model_name
    return d


def main(cfg):
    cfg = _merge_cfg(cfg)
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

    input_data = _load_parquet_as_records(cfg["input_file"])

    if cfg["begin_idx"] >= 0 and cfg["end_idx"] >= 0:
        input_data = input_data[cfg["begin_idx"]:cfg["end_idx"]]

    if input_data:
        print(normalize_messages(input_data[0]["prompt"]))

    all_responses = [None] * len(input_data)

    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as executor:
        futures = {
            executor.submit(call_openai, client, cfg, item["prompt"]): i
            for i, item in enumerate(input_data)
        }

        for future in as_completed(futures):
            i = futures[future]
            all_responses[i] = future.result()
            print(f"finished {i + 1}/{len(input_data)}")

    if all_responses:
        print(all_responses[0][0] if all_responses[0] else "")

    res_data = [
        process_one(i, input_data[i], all_responses[i], cfg["model_name"])
        for i in range(len(input_data))
    ]

    total_preds = 0
    correct_preds = 0
    pass_at_k = 0
    avg_length = 0.0

    for d in res_data:
        accs = d.get("acc_list", [])
        total_preds += len(accs)
        correct_preds += sum(1 for acc in accs if acc)
        pass_at_k += int(any(accs))

        responses = d.get("responses", [])
        if responses:
            avg_length += sum(count_tokens(r) for r in responses) / len(responses)

    accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
    pass_at_k = pass_at_k / len(res_data) if res_data else 0.0
    avg_length = avg_length / len(res_data) if res_data else 0.0

    print(f"dataset: {cfg['input_file']}")
    print(f"Total predictions: {total_preds}")
    print(f"Accurate predictions: {correct_preds}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"pass@k: {pass_at_k:.4f}")
    print(f"avg_length: {avg_length:.4f}")

    os.makedirs(os.path.dirname(cfg["output_file"]), exist_ok=True)

    with open(cfg["output_file"], "w", encoding="utf-8") as f:
        for d in res_data:
            f.write(json.dumps(ensure_obj(d), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    model_name = "gpt-5.2"
    dataset_name = "MathTestTotal"
    num_outputs = 8

    cfg = {
        "model_name": model_name,
        "base_url": "http://35.220.164.252:3888/v1",
        "api_key": "sk-LCNRSkN5fnAsRTJ8a5VUvyQznlWR2LJEpVCAoRhhodxx8Ls2",
        "input_file": f"/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/{dataset_name}/test.parquet",
        "output_file": f"/mnt/petrelfs/fudaocheng/codes/G-OPD/eval_outputs/{model_name}_{dataset_name}_pass@{num_outputs}.jsonl",
        "n": num_outputs,
    }

    main(cfg)