# MP-OPD

**Multi-Prompt Expert On-Policy Distillation for recommendation.**

MP-OPD is a development fork of P-OPD that studies how one student response can be distilled from several prompt-conditioned views of a shared frozen teacher.

The design uses five prompt views for each training sample:

1. One clean prompt shared by the student, student base, and teacher base.
2. A chasing-interest prompt.
3. A long-term-interest prompt.
4. A repurchase prompt.
5. A generalized-interest prompt.

The four experts share the same teacher checkpoint. Their behavior differs through direction-specific instructions and optional training-only evidence. The student never receives expert instructions or future evidence.

For the complete design, formulas, data schema, and implementation boundaries, see [docs/MP-OPD.md](docs/MP-OPD.md).

## Repository status

The MP-OPD configuration, data path, expert prompt packing, target construction, worker probability preparation, actor update, and dedicated trainer loop are implemented. CPU coverage for these layers runs on the configured server; full GPU integration remains a separate verification step.

The inherited `verl/` tree is retained as the implementation foundation. Unrelated experiment assets and launchers from the previous domains remain excluded; MP-OPD has its own recommendation launcher.

## Training data and prompt views

Each row provides one clean chat prompt in `prompt`. The data loader reads optional per-sample expert information from `extra_info.expert_contexts`. MP-OPD then builds five views:

1. `clean`, shared byte-for-byte by the student rollout, student base, and teacher base;
2. `chasing`;
3. `long_term`;
4. `repurchase`;
5. `generalized`.

The last four views use one frozen expert checkpoint with different prompt instructions. They do not represent four separately trained models. Their tensor order always follows `algorithm.mp_opd.expert_names`; input dictionary order is ignored.

An expert context can contain `enabled`, `evidence_available`, `evidence`, and an optional `instruction_override`. These states are intentionally distinct:

- `enabled=false`: exclude this expert for the sample;
- missing context, or `enabled=true, evidence_available=false`: keep the expert active using its configured direction, without privileged evidence;
- `enabled=true, evidence_available=true, evidence=[]`: the outcome was observed and was empty; this remains valid evidence and is not treated as missing.

For the complete row schema and an example, see [the data schema](docs/MP-OPD.md#11-数据格式).

## Launching training

The launcher expects local Hugging Face checkpoints and Parquet files. Its defaults are relative to the repository root and can be replaced through environment variables:

```bash
STUDENT_MODEL_PATH=/models/student \
EXPERT_MODEL_PATH=/models/expert \
TRAIN_FILES=/data/recommendation/train.parquet \
VAL_FILES=/data/recommendation/validation.parquet \
bash scripts/mpopd_recommendation.sh \
  trainer.n_gpus_per_node=8
```

Additional arguments are forwarded unchanged as Hydra overrides. The launcher selects FSDP and vLLM with conservative per-GPU micro-batch defaults, fixes the MP-OPD mode, four-expert order, default fusion temperatures, clean/raw-chat retention, console-only logging, and disables reward KL, actor KL loss, and validation by default. Later command-line overrides take precedence.

The student and expert tokenizers must have identical token-to-ID vocabulary maps. MP-OPD deliberately rejects cross-tokenizer bridging because every expert scores the exact top-k token IDs selected by the current student.

Expert instructions and future evidence are training-only privileged information. Validation and deployed inference use the trained student with the clean prompt only; they do not require the expert checkpoint or `expert_contexts`.

## Validation environment

This workstation does not contain the required runtime environment or model weights. Local work is limited to repository and static-content checks. The incremental CPU suites through trainer orchestration have been run on the configured server.

The remaining integration verification is:

- the aggregate MP-OPD CPU regression suite;
- Ray, FSDP, and vLLM integration tests;
- GPU execution with the student and teacher checkpoints;
- the minimal MP-OPD training smoke test.

Commands not actually run on the server must be reported as **not run**, not as passing.

## Upstream

The original project is retained as the Git remote `upstream-p-opd` for history and comparison. MP-OPD development must not modify the sibling `P-OPD/` repository.

## License

See [LICENSE](LICENSE).
