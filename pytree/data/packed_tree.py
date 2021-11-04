from .utils import get_root, get_childrens, get_tree_depth, get_nodes_detph, get_step_length
import numpy as np


def pack_tree(inputs, head_idx):
    node_depth = get_nodes_detph(head_idx)
    depth = get_tree_depth(head_idx)
    step_length = get_step_length(node_depth)
    # packed_inputs = np.zeros((depth, step_length))

    packed_inputs = [[0] for _ in range(depth)]
    packed_head_idx = [[0] for _ in range(depth)]
    packed_node_idx = [[0] for _ in range(v)]

    for head_idx_, input_ in zip(head_idx, inputs):
        depth_ = 0
        node_childrens = [get_root(head_idx_)]
        while len(node_childrens) > 0:
            packed_inputs[depth_] += [input_[n] for n in node_childrens]
            packed_node_idx[depth_] += node_childrens
            node_childrens, node_inverse = get_childrens(node_childrens, head_idx_, return_correpondance=True)  
            packed_head_idx_max = packed_head_idx[depth_][-1] + 1 if len(packed_head_idx[depth_]) else 0
            packed_head_idx[depth_] += [n + packed_head_idx_max for n in node_inverse]
            depth_ += 1
    packed_inputs

    for d in range(depth):
        pad_2_add = step_length - len(packed_inputs[d]) + 1
        packed_inputs[d] += [0] * pad_2_add
        
        pad_2_add = step_length - len(packed_head_idx[d]) + 1
        packed_head_idx[d] += [0] * pad_2_add
        
        pad_2_add = step_length - len(packed_node_idx[d]) + 1
        packed_node_idx[d] += [0] * pad_2_add
    packed_head_idx[-1] = list(range(0, step_length  + 1))
    return packed_inputs


class PackedTree:
  def __init__(self):
    pass



