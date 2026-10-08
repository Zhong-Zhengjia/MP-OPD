"""Write normalized rows to Parquet for the verl dataloader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def write_parquet(rows: Iterable[dict], output_path: Path) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet output requires the optional 'pyarrow' dependency") from exc

    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty MP-OPD parquet dataset")

    # Keep nested contexts as JSON strings for broad Arrow/verl compatibility;
    # the runtime adapter decodes this field before prompt packing.
    parquet_rows = []
    for row in materialized:
        converted = dict(row)
        extra_info = dict(converted["extra_info"])
        extra_info["expert_contexts"] = json.dumps(
            extra_info["expert_contexts"], ensure_ascii=False
        )
        converted["extra_info"] = extra_info
        parquet_rows.append(converted)

    table = pa.Table.from_pylist(parquet_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    return len(materialized)
