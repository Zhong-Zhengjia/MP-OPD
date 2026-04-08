import os
import json
import copy
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


def find_bad(obj, path="root"):
    if isinstance(obj, np.ndarray):
        print("ndarray at", path, "shape=", obj.shape, "dtype=", obj.dtype)
        return True
    if isinstance(obj, (np.integer, np.floating)):
        print("numpy scalar at", path, "type=", type(obj), "value=", obj)
        return True
    if torch.is_tensor(obj):
        print("tensor at", path, "shape=", tuple(obj.shape), "dtype=", obj.dtype)
        return True
    if isinstance(obj, dict):
        for k, v in obj.items():
            if find_bad(v, f"{path}.{k}"):
                return True
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            if find_bad(v, f"{path}[{i}]"):
                return True
    return False


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

    return x


DEFAULT_CFG = {
    "model_name": None,
    "input_file": None,          # required: parquet
    "model_path": None,          # required
    "output_file": None,         # required: jsonl
    "max_tokens": 8192,
    "temperature": 1.0,
    "top_p": 1.0,
    "max_num_seqs": 256,
    "n": 1,
    "begin_idx": -1,
    "end_idx": -1,
    "seed": 42,
    "enable_thinking": False,
    "chat_template": None,
    "gpu_memory_utilization": 0.95,
    "tensor_parallel_size": None,  # None -> torch.cuda.device_count()
}


def _merge_cfg(user_cfg: dict) -> dict:
    cfg = dict(DEFAULT_CFG)
    if user_cfg:
        cfg.update(user_cfg)

    for k in ["model_name", "input_file", "model_path", "output_file"]:
        if not cfg.get(k):
            raise ValueError(f"Missing required config: `{k}`")

    if cfg["tensor_parallel_size"] is None:
        cfg["tensor_parallel_size"] = torch.cuda.device_count()

    return cfg


def _load_parquet_as_records(path: str):
    df = pd.read_parquet(path)

    required_cols = {"prompt", "reward_model"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}")

    def ensure_obj(x):
        if isinstance(x, (dict, list)):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return x
        return x

    df["prompt"] = df["prompt"].apply(ensure_obj)
    df["reward_model"] = df["reward_model"].apply(ensure_obj)

    records = df.to_dict(orient="records")
    return records


def main(cfg: dict):
    cfg = _merge_cfg(cfg)

    toker = AutoTokenizer.from_pretrained(cfg["model_path"])
    model_name = cfg['model_name']

    llm = LLM(
        model=cfg["model_path"],
        tokenizer=cfg["model_path"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        tensor_parallel_size=cfg["tensor_parallel_size"],
        max_num_seqs=cfg["max_num_seqs"],
    )

    sampling_params = SamplingParams(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_tokens=cfg["max_tokens"],
        n=cfg["n"],
        seed=cfg["seed"],
    )

    input_data = _load_parquet_as_records(cfg["input_file"])

    if cfg["begin_idx"] >= 0 and cfg["end_idx"] >= 0:
        input_data = input_data[cfg["begin_idx"]: cfg["end_idx"]]

    prompt_texts = []
    for item in input_data:
        messages = item["prompt"]
        prompt_texts.append(
            apply_chat_template(
                toker,
                messages,
                chat_template=cfg["chat_template"],
                enable_thinking=cfg["enable_thinking"],
            )
        )

    generations = llm.generate(prompt_texts, sampling_params=sampling_params)

    res_data = []
    for i in range(len(input_data)):
        d = copy.deepcopy(input_data[i])

        gt = d.get("reward_model", {}).get("ground_truth", None)
        if gt is None:
            raise ValueError(f"Missing reward_model.ground_truth at row {i}")
        gt = str(gt)

        responses = []
        boxed_answers = []
        acc_list = []

        for j in range(len(generations[i].outputs)):
            response = generations[i].outputs[j].text.strip()
            responses.append(response)

            boxed_answer = remove_boxed(last_boxed_only_string(response))
            boxed_answers.append(boxed_answer)

            if boxed_answer is None:
                acc = False
            else:
                acc = verify(
                    parse("\\boxed{" + gt + "}"),
                    parse("\\boxed{" + boxed_answer + "}"),
                )
            acc_list.append(acc)

        d["pred_answers"] = boxed_answers
        d["responses"] = responses
        d["acc_list"] = acc_list
        d["model"] = model_name
        res_data.append(d)

    total_preds = 0
    correct_preds = 0
    pass_at_k = 0
    avg_length = 0.0

    for d in res_data:
        accs = d.get("acc_list", [])
        total_preds += len(accs)
        correct_preds += sum(1 for acc in accs if acc)
        if any(acc for acc in accs):
            pass_at_k += 1

        responses = d.get("responses", [])
        if responses:
            avg_length += sum(
                len(toker.encode(r, add_special_tokens=False)) for r in responses
            ) / len(responses)

    accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
    pass_at_k = pass_at_k / len(res_data) if res_data else 0.0
    avg_length = avg_length / len(res_data) if res_data else 0.0

    print(f"dataset: {cfg['input_file']}")
    print(f"Total predictions: {total_preds}")
    print(f"Accurate predictions: {correct_preds}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"pass@k: {pass_at_k:.4f}")
    print(f"avg_length: {avg_length:.4f}")


    with open(cfg["output_file"], "w", encoding="utf-8") as f:
        for idx, d in enumerate(res_data):
            f.write(json.dumps(ensure_obj(d), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    model_name = 'Qwen3-4B-Thinking'
    if 'thinking' in model_name.lower():
        enable_thinking = True
    else:
        enable_thinking = False
    dataset_name = 'MathTestTotal'
    num_outputs = 64
    print('='*30)
    print(model_name, enable_thinking, num_outputs)
    print('='*30)
    cfg = {
        "model_name": model_name,
        "input_file": f"/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/{dataset_name}/test.parquet",
        "model_path": f"/mnt/petrelfs/fudaocheng/checkpoints/huggingface/{model_name}",
        "output_file": f"/mnt/petrelfs/fudaocheng/codes/G-OPD/eval_outputs/{model_name}_{dataset_name}_pass@{num_outputs}.jsonl",
        "n": num_outputs,
        "enable_thinking": enable_thinking
    }
    main(cfg)