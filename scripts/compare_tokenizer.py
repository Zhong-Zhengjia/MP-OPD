from transformers import AutoTokenizer

path1 = "/mnt/petrelfs/fudaocheng/checkpoints/huggingface/Qwen3-4B"
path2 = "/mnt/petrelfs/fudaocheng/checkpoints/huggingface/Qwen3-4B-Thinking"

tok1 = AutoTokenizer.from_pretrained(path1, trust_remote_code=True, local_files_only=True)
tok2 = AutoTokenizer.from_pretrained(path2, trust_remote_code=True, local_files_only=True)

print("class:", tok1.__class__.__name__, tok2.__class__.__name__)
print("vocab_size:", tok1.vocab_size, tok2.vocab_size)
print("special_tokens_map same:", tok1.special_tokens_map == tok2.special_tokens_map)
print("vocab same:", tok1.get_vocab() == tok2.get_vocab())

text = "你好，Qwen!"
print("encode same:", tok1.encode(text) == tok2.encode(text))