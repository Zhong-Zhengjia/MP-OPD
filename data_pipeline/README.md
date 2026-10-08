# MP-OPD data pipeline

This package converts source-specific datasets into the normalized MP-OPD
training-row schema. It intentionally does not import `verl`, Ray, FSDP,
vLLM, or model code.

## Current recommendation adapter

The recommendation adapter consumes the split JSONL format under
`datasets/ours/splits/` and maps:

```text
ground_truth_chasing     -> chasing.evidence
ground_truth_long_term   -> long_term.evidence
ground_truth_repurchase  -> repurchase.evidence
ground_truth_generalized -> generalized.evidence
```

The clean prompt is rendered once. Expert instructions and evidence are
stored in `extra_info.expert_contexts`; the runtime MP-OPD prompt builder
combines them with the clean prompt when needed. This avoids storing five
copies of each large user profile.

## Usage

From the repository root:

```bash
python -m mpopd_data.cli recommendation \
  --input datasets/ours/splits/train_1000.jsonl \
  --output MP-OPD/data/train_1000.jsonl \
  --clean-template baseline/student_prompt.txt \
  --expert-prompt-dir MP-OPD/prompts
```

Set `PYTHONPATH=MP-OPD/data_pipeline/src` when running without installing the
package. Parquet conversion is available through the same CLI and requires
`pyarrow` in the data-preparation environment.

For the standard server layout, the complete train_1000 preparation and
validation flow is:

```bash
bash scripts/prepare_train_1000.sh
bash scripts/test_data_pipeline.sh
```

The preparation script accepts `SOURCE_JSONL`, `CLEAN_TEMPLATE`,
`EXPERT_PROMPT_DIR`, `OUTPUT_JSONL`, and `VALIDATION_OUTPUT` environment
overrides. It does not run `git pull`; update the repository explicitly before
running it.

## Normalized row contract

Each row contains `data_source`, `prompt`, `ability`, `reward_model`, and
`extra_info.expert_contexts`. Expert order is always:

```text
chasing, long_term, repurchase, generalized
```

An enabled expert with missing evidence remains active. An observed empty
evidence object is also valid and remains active. Only `enabled: false`
disables an expert.
