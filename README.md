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
input_test, head_idx_test = prepare_input_from_constituency_tree(parse_tree)

print(input_test)
# ['I', 'saw', 'Sarah', 'with', 'a', 'telescope', '.', '[S]', '[S]', '[VP]', '[VP]', '[PP]', '[NP]']

print(head_idx_test)
# [8, 10, 10, 11, 12, 12, 7, 0, 7, 8, 9, 9, 11]
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
input_test, head_idx_test = prepare_input_from_dependency_tree(parse_tree)

print(input_test)
# ['I', 'saw', 'Sarah', 'with', 'a', 'telescope', '.']

print(head_idx_test)
# [2, 0, 2, 2, 6, 4, 2]
```
