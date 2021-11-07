<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/AntoineSimoulin/pytree/master/imgs/pytree_logo.png" width="400"/>
    <br>
<p>

**PyTree** implements tree-structured neural networks in PyTorch.
The package provides highly generic recursive neural network implementations as well as efficient batching methods.

## Installation

To install please run:

```bash
pip install git+https://github.com/AntoineSimoulin/pytree
```

## Usage

### Prepare constituency tree data

Data must be in the `str` format as detailed in the example below:

```python
from pytree.data import prepare_input_from_constituency_tree

parse_tree_example = '(TOP (S (NP (_ I)) (VP (_ saw) (NP (_ Sarah)) (PP (_ with) (NP (_ a) (_ telescope)))) (_ .)))'
input_test, head_idx_test = prepare_input_from_constituency_tree(parse_tree_example)

print(input_test)
# ['[CLS]', 'I', 'saw', 'Sarah', 'with', 'a', 'telescope', '.', '[S]', '[S]', '[VP]', '[VP]', '[PP]', '[NP]']

print(head_idx_test)
# [0, 8, 10, 10, 11, 12, 12, 7, 0, 7, 8, 9, 9, 11]
```

### Prepare dependency tree data

Data must be in the `conll-X` format as detailed in the example below:

```python
from pytree.data import prepare_input_from_dependency_tree

parse_tree_example = """1	I	_	_	_	_	2	nsubj	_	_
2	saw	_	_	_	_	0	root	_	_
3	Sarah	_	_	_	_	2	dobj	_	_
4	with	_	_	_	_	2	prep	_	_
5	a	_	_	_	_	6	det	_	_
6	telescope	_	_	_	_	4	pobj	_	_
7	.	_	_	_	_	2	punct	_	_
"""
input_test, head_idx_test = prepare_input_from_dependency_tree(parse_tree_example)

print(input_test)
# ['[CLS]', 'I', 'saw', 'Sarah', 'with', 'a', 'telescope', '.']

print(head_idx_test)
# [0, 2, 0, 2, 2, 6, 4, 2]
```

### Prepare data

You may encode the word with GloVe emebddings :

```python
from pytree.data.glove_tokenizer import GloveTokenizer

glove_tokenizer = GloveTokenizer(glove_file_path='./glove.6B.300d.txt', vocab_size=10000)
input_test = glove_tokenizer.convert_tokens_to_ids(input_test)
print(input_test)
# [1, 1, 824, 1, 19, 9, 1, 4, 1, 1, 1, 1, 1, 1]
```

Then prepare the data:

```python
tree_ids_test, tree_ids_test_r, tree_ids_test_l = build_tree_ids_n_ary(head_idx_test)
inputs = {'input_ids': torch.tensor(input_test).unsqueeze(0),
          'packed_tree': torch.tensor(tree_ids_test).unsqueeze(0),
          'packed_tree_r': torch.tensor(tree_ids_test_r).unsqueeze(0),
          'packed_tree_l': torch.tensor(tree_ids_test_l).unsqueeze(0)}
```

And apply the model:

```python
from pytree.models import NaryConfig, NaryTree

config = NaryConfig()
tree_encoder = NaryTree(config)

tree_encoder(inputs)
# tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#          [ 0.0012,  0.0015, -0.0026,  ..., -0.0001,  0.0002, -0.0043],
#          [ 0.0022,  0.0024, -0.0035,  ..., -0.0002,  0.0003, -0.0058],
#          ...,
#          [ 0.0028,  0.0023, -0.0035,  ..., -0.0002,  0.0003, -0.0057],
#          [ 0.0020,  0.0016, -0.0023,  ..., -0.0001,  0.0002, -0.0036],
#          [ 0.0019,  0.0015, -0.0024,  ..., -0.0001,  0.0002, -0.0039]]],
#        grad_fn=<MaskedScatterBackward>)

print(tree_encoder(inputs).shape)
# tree_encoder(inputs).shape
```

