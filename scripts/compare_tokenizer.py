from transformers import AutoTokenizer

path1 = "models/public_models/Qwen2.5-7B-Instruct"
path2 = "models/public_models/Qwen3-8B"

tok1 = AutoTokenizer.from_pretrained(path1, trust_remote_code=True, local_files_only=True)
tok2 = AutoTokenizer.from_pretrained(path2, trust_remote_code=True, local_files_only=True)


def _token_sets(tokenizer):
    vocab = tokenizer.get_vocab()
    tokens = set(vocab.keys())
    id_by_token = dict(vocab)
    return tokens, id_by_token


def _overlap_stats(name1, name2, tokens1, tokens2, id1, id2):
    inter = tokens1 & tokens2
    only1 = tokens1 - tokens2
    only2 = tokens2 - tokens1
    union = tokens1 | tokens2

    same_id = sum(1 for t in inter if id1[t] == id2[t])
    diff_id = len(inter) - same_id

    def ratio(num, den):
        return num / den if den else 0.0

    print(f"\n=== vocab overlap ({name1} vs {name2}) ===")
    print(f"|{name1}| = {len(tokens1)}")
    print(f"|{name2}| = {len(tokens2)}")
    print(f"|intersection| = {len(inter)}")
    print(f"|union| = {len(union)}")
    print(f"|only_{name1}| = {len(only1)}")
    print(f"|only_{name2}| = {len(only2)}")
    print(f"jaccard (|∩|/|∪|) = {ratio(len(inter), len(union)):.6f}")
    print(f"|∩|/|{name1}| = {ratio(len(inter), len(tokens1)):.6f}")
    print(f"|∩|/|{name2}| = {ratio(len(inter), len(tokens2)):.6f}")
    print(f"same token & same id in ∩: {same_id} ({ratio(same_id, len(inter)):.6f} of ∩)")
    print(f"same token but different id in ∩: {diff_id} ({ratio(diff_id, len(inter)):.6f} of ∩)")

    if only1:
        print(f"sample only_{name1} (up to 10): {sorted(only1)[:10]}")
    if only2:
        print(f"sample only_{name2} (up to 10): {sorted(only2)[:10]}")
    if diff_id:
        diff_samples = [t for t in sorted(inter) if id1[t] != id2[t]][:10]
        print(
            "sample same token, different id:",
            [(t, id1[t], id2[t]) for t in diff_samples],
        )


tokens1, id1 = _token_sets(tok1)
tokens2, id2 = _token_sets(tok2)

print("class:", tok1.__class__.__name__, tok2.__class__.__name__)
print("vocab_size:", tok1.vocab_size, tok2.vocab_size)
print("special_tokens_map same:", tok1.special_tokens_map == tok2.special_tokens_map)
print("vocab same (token->id dict):", id1 == id2)

_overlap_stats("tok1", "tok2", tokens1, tokens2, id1, id2)

text = "你好，Qwen!"
print("\nencode same:", tok1.encode(text) == tok2.encode(text))
print("tok1 ids:", tok1.encode(text))
print("tok2 ids:", tok2.encode(text))
