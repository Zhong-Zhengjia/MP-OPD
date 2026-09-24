# Copyright 2026 MP-OPD contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy

import torch

from verl.trainer.ppo.expert_prompt_utils import EVIDENCE_MARKER, FOCUS_MARKER
from verl.utils.dataset.rl_dataset import RLHFDataset


class _SingleRow:
    def __init__(self, row):
        self.row = row

    def __getitem__(self, item):
        assert item == 0
        return copy.deepcopy(self.row)


class FakeTokenizer:
    pad_token_id = 0
    chat_template = "fake"

    def __init__(self):
        self.vocab = {FOCUS_MARKER: 901, EVIDENCE_MARKER: 902}
        self._next_id = 1000

    def _tokenize(self, text):
        result = []
        for token in text.split():
            if token not in self.vocab:
                self.vocab[token] = self._next_id
                self._next_id += 1
            result.append(self.vocab[token])
        return result

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize=False, **kwargs):
        del kwargs
        text = " ".join(f"{item['role']} {item['content']}" for item in messages)
        if add_generation_prompt:
            text += " assistant"
        return self._tokenize(text) if tokenize else text

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        ids = torch.tensor([self._tokenize(text)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return self._tokenize(text)


def _dataset(row):
    dataset = RLHFDataset.__new__(RLHFDataset)
    dataset.dataframe = _SingleRow(row)
    dataset.prompt_key = "prompt"
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.processor = None
    dataset.tokenizer = FakeTokenizer()
    dataset.apply_chat_template_kwargs = {}
    dataset.max_prompt_length = 16
    dataset.truncation = "error"
    dataset.return_raw_chat = True
    dataset.return_full_prompt = False
    dataset.need_tools_kwargs = False
    return dataset


def test_dataset_propagates_expert_contexts_without_putting_them_in_clean_prompt():
    expert_contexts = {
        "repurchase": {
            "enabled": True,
            "evidence_available": True,
            "evidence": "future click",
        }
    }
    dataset = _dataset(
        {
            "prompt": [{"role": "user", "content": "clean recommendation request"}],
            "data_source": "recommendation",
            "extra_info": {"index": 7, "expert_contexts": expert_contexts},
        }
    )

    item = dataset[0]

    assert item["expert_contexts"] == expert_contexts
    assert item["raw_prompt"] == [{"role": "user", "content": "clean recommendation request"}]
    assert dataset.tokenizer.vocab[FOCUS_MARKER] not in item["input_ids"]
    assert dataset.tokenizer.vocab[EVIDENCE_MARKER] not in item["input_ids"]


def test_dataset_defaults_missing_expert_contexts_to_empty_mapping():
    dataset = _dataset(
        {
            "prompt": [{"role": "user", "content": "clean request"}],
            "data_source": "recommendation",
            "extra_info": None,
        }
    )

    item = dataset[0]

    assert item["extra_info"] == {}
    assert item["expert_contexts"] == {}
