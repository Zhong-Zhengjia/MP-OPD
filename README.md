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

The independent repository and design specification have been initialized. The MP-OPD training path is not implemented yet; subsequent work follows the reviewed implementation plan task by task.

The inherited `verl/` tree is retained as the implementation foundation. Root-level experiment assets and launchers from the previous domains are intentionally excluded from this project.

## Validation environment

This workstation does not contain the required runtime environment or model weights. Local work is limited to repository and static-content checks.

The following verification must run after the repository is deployed to the configured server:

- PyTorch and verl unit tests;
- Ray, FSDP, and vLLM integration tests;
- GPU execution with the student and teacher checkpoints;
- the minimal MP-OPD training smoke test.

Until those commands have run on the server, they must be reported as **not run**, not as passing.

## Upstream

The original project is retained as the Git remote `upstream-p-opd` for history and comparison. MP-OPD development must not modify the sibling `P-OPD/` repository.

## License

See [LICENSE](LICENSE).
