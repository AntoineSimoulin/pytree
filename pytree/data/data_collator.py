import torch
from pytree.parsers.packed_tree import PackedTree

class DataCollatorForTree:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        if isinstance(samples, list):
          samples_dict = {k: [s[k] for s in samples] for k in samples[0]}
          samples = samples_dict
        samples['input_ids_A'] = torch.cat([torch.tensor(self.tokenizer.convert_tokens_to_ids(s.split())) for s in samples['sentence_A']])
        samples['packed_tree_A'] = sum([PackedTree().from_strings(c) for c in samples['dep_A']])
        samples['input_ids_B'] = torch.cat([torch.tensor(self.tokenizer.convert_tokens_to_ids(s.split())) for s in samples['sentence_B']])
        samples['packed_tree_B'] = sum([PackedTree().from_strings(c) for c in samples['dep_B']])
        samples['labels'] = torch.tensor(samples['labels'])
        return samples
