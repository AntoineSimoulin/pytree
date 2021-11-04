import numpy as np
from collections import Counter


# def get_tree_depth(head_idx):
#   if isinstance(head_idx[0], list):
#     return max([get_tree_depth(h) for h in head_idx])
#   depth = 1
#   root_idx = get_root(head_idx)
#   node_childrens = get_childrens(root_idx, head_idx)
#   while len(node_childrens) > 0:
#     node_childrens = get_childrens(node_childrens, head_idx)
#     depth +=1
#   return depth  

# def build_tree_ids(head_idx):
#   if isinstance(head_idx[0], list):
#     depth = get_tree_depth(head_idx)
#     tree_ids = [build_tree_ids(h) for h in head_idx]
#     return np.array([pad_tree_ids(t[0], depth) for t in tree_ids]), \
#       np.array([pad_tree_ids(t[1], depth) for t in tree_ids]), \
#       np.array([pad_tree_ids(t[2], depth) for t in tree_ids])
#   tree_ids = []
#   node_idx = [get_root(head_idx)]
#   while len(node_idx) > 0:
#     node_idx = get_childrens(node_idx, head_idx)
#     tree_step = [h_idx if idx in node_idx else 0 for idx, h_idx in enumerate(head_idx)]
#     tree_ids.append(tree_step)
#   tree_ids = tree_ids[:-1]
#   tree_ids.append(range(0, len(head_idx)))
#   tree_ids_r = [[t if (i % 2 == 0) else 0 for (i, t) in enumerate(ti)] for ti in tree_ids]
#   tree_ids_d = [[t if (i % 2 == 1) else 0 for (i, t) in enumerate(ti)] for ti in tree_ids]
#   return np.array(tree_ids), np.array(tree_ids_r), np.array(tree_ids_d)

def get_root(head_idx):
  return head_idx[1:].index(0) + 1


def get_childrens(node_idx, head_idx):
  if isinstance(node_idx, int):
    node_idx = [node_idx]
  return [idx for idx, h_idx in enumerate(head_idx) if h_idx in node_idx]

# def get_childrens(node_idx, head_idx, return_correpondance=False):
#   if isinstance(node_idx, int):
#     node_idx = [node_idx]
#   node_correspondance = []
#   node_childrens = []
#   for n_idx, n in enumerate(node_idx):
#     for h_idx, h in enumerate(head_idx):
#       if h == n:
#         node_correspondance.append(n_idx)
#         node_childrens.append(h_idx)
#   if return_correpondance:
#     return node_childrens, node_correspondance
#   return node_correspondance


def get_tree_depth(head_idx):
  if isinstance(head_idx[0], list):
    return max([get_tree_depth(h) for h in head_idx])
  depth = 1
  root_idx = get_root(head_idx)
  node_childrens = get_childrens(root_idx, head_idx)
  while len(node_childrens) > 0:
    node_childrens = get_childrens(node_childrens, head_idx)
    depth +=1
  return depth


def get_nodes_detph(head_idx):
  if isinstance(head_idx[0], list):
    return [get_nodes_detph(h) for h in head_idx]
  depth_ = 0
  depth = [0] * len(head_idx)
  node_childrens = [get_root(head_idx)]
  while len(node_childrens) > 0:
    for n in node_childrens:
      depth[n] = depth_
    node_childrens = get_childrens(node_childrens, head_idx)
    depth_ += 1
  return depth


def pad_tree_ids(tree_ids, depth):
  tree_depth = tree_ids.shape[0]
  padding = np.zeros_like(tree_ids[0:(max(depth - tree_depth, 0))])
  return np.concatenate((tree_ids, padding), axis=0)


def build_tree_ids(head_idx):
  if isinstance(head_idx[0], list):
    depth = get_tree_depth(head_idx)
    return np.array([pad_tree_ids(build_tree_ids(h), depth) for h in head_idx])
  tree_ids = [range(0, len(head_idx))]
  node_idx = [get_root(head_idx)]
  while len(node_idx) > 0:
    node_idx = get_childrens(node_idx, head_idx)
    tree_step = [h_idx if idx in node_idx else 0 for idx, h_idx in enumerate(head_idx)]
    tree_ids.append(tree_step)
  return np.array(tree_ids[:-1])


def build_tree_ids_n_ary(head_idx):
  if isinstance(head_idx[0], list):
    depth = get_tree_depth(head_idx)
    tree_ids = [build_tree_ids_n_ary(h) for h in head_idx]
    return np.array([pad_tree_ids(t[0], depth) for t in tree_ids]), \
      np.array([pad_tree_ids(t[1], depth) for t in tree_ids]), \
      np.array([pad_tree_ids(t[2], depth) for t in tree_ids])
  tree_ids = []
  node_idx = [get_root(head_idx)]
  while len(node_idx) > 0:
    node_idx = get_childrens(node_idx, head_idx)
    tree_step = [h_idx if idx in node_idx else 0 for idx, h_idx in enumerate(head_idx)]
    tree_ids.append(tree_step)
  tree_ids = tree_ids[:-1]
  tree_ids.append(range(0, len(head_idx)))
  tree_ids_r = [[t if (i % 2 == 0) else 0 for (i, t) in enumerate(ti)] for ti in tree_ids]
  tree_ids_d = [[t if (i % 2 == 1) else 0 for (i, t) in enumerate(ti)] for ti in tree_ids]
  return np.array(tree_ids), np.array(tree_ids_r), np.array(tree_ids_d)


def get_step_length(node_depth):
  if isinstance(node_depth[0], list):
    return get_step_length([ll for l in node_depth for ll in l ])
  return Counter([n for n in node_depth if n != 0]).most_common(1)[0][1]