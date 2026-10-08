from mpopd_data.adapters.recommendation import adapt_recommendation_row
from mpopd_data.schema import EXPERT_NAMES


def test_recommendation_fields_are_isolated():
    source = {
        "user_id": "u1",
        "profile_llm": {"最近14天历史点击商品": {"面霜": 2}},
        "ubuyereffectivetype": "A",
        "ground_truth": {"面霜": 1},
        "ground_truth_chasing": {},
        "ground_truth_long_term": {"面霜": 1},
        "ground_truth_repurchase": {},
        "ground_truth_generalized": {"护肤": 1},
    }
    templates = {
        name: f"{name} instruction\n{{{{expert_evidence}}}}"
        for name in EXPERT_NAMES
    }
    row = adapt_recommendation_row(
        source,
        clean_template="clean\n<在此处填入输入 JSON>",
        expert_templates=templates,
    )
    assert list(row["extra_info"]["expert_contexts"]) == list(EXPERT_NAMES)
    assert row["extra_info"]["expert_contexts"]["long_term"]["evidence"] == {"面霜": 1}
    assert row["extra_info"]["expert_contexts"]["repurchase"]["evidence"] == {}
    assert "ground_truth" not in row["prompt"][0]["content"]
