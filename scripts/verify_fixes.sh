#!/usr/bin/env bash
# Verify the MP-OPD design-consistency fixes on this machine.
#
# This script pulls the fix branch, regenerates the (gitignored) training data
# from the updated expert prompt templates, then runs the full MP-OPD test
# suite and prints a pass/fail summary.
#
# One-time bootstrap (run once, from the repo root):
#     git fetch origin
#     git checkout fix/mpopd-design-consistency
#     bash scripts/verify_fixes.sh
#
# Re-run after new fixes are pushed to the same branch:
#     bash scripts/verify_fixes.sh
#
# Override the branch:
#     BRANCH=fix/mpopd-design-consistency bash scripts/verify_fixes.sh

set -euo pipefail

BRANCH="${BRANCH:-fix/mpopd-design-consistency}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Prefer ``python`` (typical in a conda/venv ML env), fall back to ``python3``.
if command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi

echo "############################################################"
echo "# MP-OPD fix verification"
echo "# branch : ${BRANCH}"
echo "# root   : ${REPO_ROOT}"
echo "# python : $(${PY} --version 2>&1)"
echo "############################################################"
echo

# ── 1. Pull latest ──────────────────────────────────────────────
echo "=== [1/3] Pulling ${BRANCH} ==="
git fetch origin
current="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${current}" != "${BRANCH}" ]]; then
  echo "Currently on '${current}'; switching to '${BRANCH}'."
  git checkout "${BRANCH}"
fi
git pull origin "${BRANCH}"
echo "HEAD -> $(git log --oneline -1)"
echo

# ── 2. Regenerate training data ─────────────────────────────────
# data/ is gitignored and the expert prompt templates changed in this fix, so
# any pre-existing data/train_1000.jsonl carries stale instructions. Rebuild it
# from source (../datasets/ours/splits/train_1000.jsonl + ../baseline/...).
echo "=== [2/3] Regenerating data/train_1000.jsonl ==="
bash scripts/prepare_train_1000.sh
echo

# ── 3. Run test suite ───────────────────────────────────────────
echo "=== [3/3] Running test suite ==="
PASS=0
FAIL=0
FAILED_GROUPS=()

run_group() {
  local name="$1"; shift
  echo "── ${name} ──"
  if "$@"; then
    echo ">>> PASS: ${name}"
    PASS=$((PASS + 1))
  else
    echo ">>> FAIL: ${name}"
    FAIL=$((FAIL + 1))
    FAILED_GROUPS+=("${name}")
  fi
  echo
}

# 3.1 Core math + strict mode (#6) — torch required
run_group "core_algos" \
  bash -c "cd verl && ${PY} -m pytest tests/trainer/ppo/test_mpopd_core_algos_on_cpu.py -v"

# 3.2 Expert prompt construction + length bound (#5/#8) — torch required
run_group "expert_prompt_utils" \
  bash -c "cd verl && ${PY} -m pytest tests/trainer/ppo/test_expert_prompt_utils_on_cpu.py -v"

# 3.3 Schema validation (#7) — pure stdlib, no torch
run_group "schema_validation" \
  bash -c "PYTHONPATH=data_pipeline/src ${PY} -m pytest data_pipeline/tests/test_schema_validation.py -v"

# 3.4 Actor metrics + worker helpers (#2/#4/#9) — torch/Ray required
run_group "actor+worker" \
  bash -c "cd verl && ${PY} -m pytest tests/workers/actor/test_mpopd_actor_on_cpu.py tests/workers/test_mpopd_worker_helpers_on_cpu.py -v"

# 3.5 Trainer end-to-end (#8) — torch/Ray required
run_group "trainer" \
  bash -c "cd verl && ${PY} -m pytest tests/trainer/ppo/test_mpopd_trainer_on_cpu.py -v"

# 3.6 Config layer
run_group "config" \
  bash -c "cd verl && ${PY} -m pytest tests/trainer/config/test_mpopd_config_on_cpu.py -v"

# ── Summary ────────────────────────────────────────────────────
echo "════════════════════════════════════════════"
echo "  PASSED: ${PASS}    FAILED: ${FAIL}"
if [[ ${FAIL} -gt 0 ]]; then
  echo "  Failed groups: ${FAILED_GROUPS[*]}"
  echo "════════════════════════════════════════════"
  exit 1
fi
echo "  All test groups passed."
echo "════════════════════════════════════════════"
