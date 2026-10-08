"""JSONL source-to-normalized conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def convert_jsonl(
    source_path: Path,
    output_path: Path,
    adapter: Callable[[dict], dict],
) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                source_row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source_path}:{line_number}: invalid JSON") from exc
            if not isinstance(source_row, dict):
                raise ValueError(f"{source_path}:{line_number}: row must be an object")
            output.write(json.dumps(adapter(source_row), ensure_ascii=False) + "\n")
            count += 1
    return count
