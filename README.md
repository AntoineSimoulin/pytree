<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/AntoineSimoulin/pytree/master/imgs/pytree_logo.png" width="400"/>
    <br>
<p>

**PyTree** implements tree-structured neural networks in PyTorch.
The package provides highly generic tree-structured neural network implementations as well as efficient batching methods.

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
input_test, head_idx_test, head_idx_r_test, head_idx_l_test = prepare_input_from_constituency_tree(parse_tree_example)

print(input_test)
# ['[CLS]', 'I', 'saw', 'Sarah', 'with', 'a', 'telescope', '.', '[S]', '[S]', '[VP]', '[VP]', '[PP]', '[NP]']

print(head_idx_test)
# [0, 9, 11, 11, 12, 13, 13, 8, 0, 8, 9, 10, 10, 12]

print(head_idx_r_test)
# [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0]

print(head_idx_l_test)
# [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1]
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
# [1, 1, 824, 1, 19, 9, 1, 4]
```

Then prepare the data:

```python
from pytree.data.utils import build_tree_ids_n_ary

tree_ids_test, tree_ids_test_r, tree_ids_test_l = build_tree_ids_n_ary(head_idx_test, head_idx_r_test, head_idx_l_test)
inputs = {'input_ids': torch.tensor(input_test).unsqueeze(0),
          'tree_ids': torch.tensor(tree_ids_test).unsqueeze(0),
          'tree_ids_r': torch.tensor(tree_ids_test_r).unsqueeze(0),
          'tree_ids_l': torch.tensor(tree_ids_test_l).unsqueeze(0)}
```

And apply the model:

```python
from pytree.models import NaryConfig, NaryTree

config = NaryConfig()
tree_encoder = NaryTree(config)

(h, c), h_root = tree_encoder(inputs)
print(h)
# tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#          [ 0.0113, -0.0066,  0.0089,  ...,  0.0064,  0.0076, -0.0048],
#          [ 0.0110, -0.0073,  0.0110,  ...,  0.0070,  0.0046, -0.0049],
#          ...,
#          [ 0.0254, -0.0138,  0.0224,  ...,  0.0131,  0.0148, -0.0143],
#          [ 0.0346, -0.0172,  0.0281,  ...,  0.0140,  0.0198, -0.0267],
#          [ 0.0247, -0.0126,  0.0201,  ...,  0.0116,  0.0162, -0.0184]]],
#        grad_fn=<SWhereBackward>)

print(h_root.shape)
# torch.Size([150])
```

We also provide a full demonstration with the SICK dataset and batched processing in the [examples folder](https://github.com/AntoineSimoulin/pytree/tree/main/examples). 