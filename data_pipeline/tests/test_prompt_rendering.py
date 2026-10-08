from mpopd_data.prompts import extract_instruction, render_clean_prompt


def test_clean_prompt_contains_no_expert_evidence():
    rendered = render_clean_prompt(
        "before\n<在此处填入输入 JSON>\nafter",
        {"recent": {"商品": 1}},
    )
    assert "商品" in rendered
    assert "expert_evidence" not in rendered


def test_extract_instruction_stops_before_evidence_placeholder():
    assert extract_instruction("focus recent\n{{expert_evidence}}") == "focus recent"
