import json


def read_jsonl(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            response = obj['pred_answers']

            first = response[0]
            print(idx, all(s == first for s in response))


def read_first_line(file_path: str):
    with open(file_path, "r", encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            print(obj['responses'][1])
            break


if __name__ == '__main__':
    file_path = 'eval_outputs/Qwen3-4B-Thinking_MathTestTotal_pass@8.jsonl'
    # read_jsonl(file_path)
    read_first_line(file_path)
