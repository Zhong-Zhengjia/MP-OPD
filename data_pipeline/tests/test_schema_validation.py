from mpopd_data.schema import EXPERT_NAMES, SchemaValidationError, validate_mpopd_row


def make_row():
    contexts = {
        name: {
            "enabled": True,
            "instruction": f"focus {name}",
            "evidence_available": False,
        }
        for name in EXPERT_NAMES
    }
    return {
        "data_source": "test",
        "prompt": [{"role": "user", "content": "clean"}],
        "ability": "Recommendation",
        "reward_model": {"ground_truth": {}},
        "extra_info": {"expert_contexts": contexts},
    }


def test_schema_preserves_fixed_order_and_empty_evidence():
    row = make_row()
    row["extra_info"]["expert_contexts"]["repurchase"].update(
        evidence_available=True, evidence={}
    )
    normalized = validate_mpopd_row(row)
    assert list(normalized.expert_contexts) == list(EXPERT_NAMES)
    assert normalized.expert_contexts["repurchase"].evidence == {}


def test_schema_rejects_reordered_experts():
    row = make_row()
    contexts = row["extra_info"]["expert_contexts"]
    row["extra_info"]["expert_contexts"] = dict(reversed(list(contexts.items())))
    try:
        validate_mpopd_row(row)
    except SchemaValidationError:
        pass
    else:
        raise AssertionError("reordered experts should fail validation")


def test_instruction_is_optional_and_override_supported():
    # §11: instructions may be config-level; a row can omit ``instruction`` and only
    # carry ``instruction_override`` when a sample overrides the config default.
    row = make_row()
    for name in EXPERT_NAMES:
        row["extra_info"]["expert_contexts"][name].pop("instruction", None)
        row["extra_info"]["expert_contexts"][name]["instruction_override"] = f"override {name}"
    normalized = validate_mpopd_row(row)
    assert normalized.expert_contexts["chasing"].instruction is None
    assert normalized.expert_contexts["chasing"].instruction_override == "override chasing"
    out = normalized.to_dict()
    assert "instruction" not in out["extra_info"]["expert_contexts"]["chasing"]
    assert out["extra_info"]["expert_contexts"]["chasing"]["instruction_override"] == "override chasing"


def test_instruction_when_present_must_be_nonempty():
    row = make_row()
    row["extra_info"]["expert_contexts"]["chasing"]["instruction"] = "   "
    try:
        validate_mpopd_row(row)
    except SchemaValidationError:
        pass
    else:
        raise AssertionError("blank instruction should fail validation")
