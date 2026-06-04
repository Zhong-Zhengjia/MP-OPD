import os
import sqlite3
from pathlib import Path
from contextlib import closing

import pandas as pd

model_name = 'Qwen3-1.7B'
num_rollouts = 16

original_database = f"/mnt/phwfile/datafrontier/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6_{model_name}_pass@16.sqlite"
val_parquet = "/mnt/phwfile/datafrontier/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/val_1000.parquet"
output_path = f"/mnt/phwfile/datafrontier/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/val_1000_{model_name}_pass@16.sqlite"


def build_question_id(item):
    data_source = str(item.get("data_source", "unknown"))
    extra_info = item.get("extra_info", {})
    if not isinstance(extra_info, dict):
        extra_info = {}
    index = extra_info.get("index", None)
    if index is None:
        raise ValueError(f"Missing extra_info['index'] in item: {item}")
    return f"{data_source}_{index}"


def quote_ident(name):
    return '"' + name.replace('"', '""') + '"'


def chunks(xs, size):
    for i in range(0, len(xs), size):
        yield xs[i:i + size]


if os.path.abspath(original_database) == os.path.abspath(output_path):
    raise ValueError("output_path must be different from original_database")

df = pd.read_parquet(val_parquet)
question_ids = [build_question_id(item) for item in df.to_dict("records")]
question_ids = list(dict.fromkeys(question_ids))

os.makedirs(os.path.dirname(output_path), exist_ok=True)

if os.path.exists(output_path):
    os.remove(output_path)

src_uri = Path(original_database).resolve().as_uri() + "?mode=ro"

with closing(sqlite3.connect(src_uri, uri=True)) as src, closing(sqlite3.connect(output_path)) as dst:
    schema_row = src.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'generations'"
    ).fetchone()

    if schema_row is None or schema_row[0] is None:
        raise ValueError("Table generations not found in original database")

    dst.execute(schema_row[0])

    columns = [
        row[1]
        for row in src.execute("PRAGMA table_info(generations)").fetchall()
    ]

    columns_sql = ", ".join(quote_ident(c) for c in columns)
    placeholders = ", ".join("?" for _ in columns)

    insert_sql = f"INSERT INTO generations ({columns_sql}) VALUES ({placeholders})"

    source_counts = {}
    total_source_rows = 0

    for batch in chunks(question_ids, 500):
        in_sql = ", ".join("?" for _ in batch)
        rows = src.execute(
            f"""
            SELECT question_id, COUNT(*)
            FROM generations
            WHERE question_id IN ({in_sql})
            GROUP BY question_id
            """,
            batch,
        ).fetchall()

        for question_id, count in rows:
            source_counts[question_id] = count
            total_source_rows += count

    missing_question_ids = [qid for qid in question_ids if qid not in source_counts]

    if missing_question_ids:
        raise ValueError(f"{len(missing_question_ids)} question_ids not found in original database")

    with dst:
        for batch in chunks(question_ids, 500):
            in_sql = ", ".join("?" for _ in batch)
            rows = src.execute(
                f"""
                SELECT {columns_sql}
                FROM generations
                WHERE question_id IN ({in_sql})
                ORDER BY question_id, rollout_id
                """,
                batch,
            ).fetchall()

            dst.executemany(insert_sql, rows)

    copied_rows = dst.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
    copied_questions = dst.execute("SELECT COUNT(DISTINCT question_id) FROM generations").fetchone()[0]

    if copied_rows != total_source_rows:
        raise ValueError(f"Copied rows mismatch: copied={copied_rows}, expected={total_source_rows}")

    if copied_questions != len(question_ids):
        raise ValueError(f"Copied questions mismatch: copied={copied_questions}, expected={len(question_ids)}")

print(f"Copied {copied_rows} rows from {len(question_ids)} questions to {output_path}")