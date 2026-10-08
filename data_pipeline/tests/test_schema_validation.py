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
