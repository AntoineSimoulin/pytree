from pytree.parsers.utils import Graph
import numpy as np


class DepGraph(Graph):
    """Class to generate Networkx Graph object from conll sentences"""

    def __init__(self, conll):
        super(DepGraph, self).__init__()
        self.build_graph(conll)
        # self.root = self.get_root()
        # self.depth = self.get_depth()

    def build_graph(self, conll):
        conll_hearders = ['idx', 'text', 'lemma_', 'pos_', 'tag_', 'morpho_', 'head_idx', 'dep_']
        for w in conll:
            node_attr = {l[0]: l[1] for l in list(zip(conll_hearders, w.split('\t')))}
            node_attr['node_idx'] = int(node_attr['idx'])
            node_attr['idx'] = int(node_attr['idx']) - 1
            node_attr['head_idx'] = int(node_attr['head_idx'])
            node_attr['tok_id'] = int(w[0])
            # self.seq.append(node_attr['text'])
            self.add_node(node_attr['node_idx'])
            for k, v in node_attr.items():
                self.nodes[node_attr['node_idx']][k] = v
            if node_attr['head_idx'] != 0:
                self.add_edges_from([(int(node_attr['node_idx']), int(node_attr['head_idx']))])

        # no root
        # if len([n for n in self.nodes if self.node[n]['head_idx'] == 0]) == 0:
        #     root = list(self.node)[np.argmax([self.degree[n] for n in self.node()])]
        #     self.node[root]['head_idx'] = 0
        #     self.node[root]['dep_'] = 'root'
        #     for e in [e for e in list(self.edges) if root == e[0]]:
        #         self.remove_edge(*e)
        #
        # # two components
        # if not nx.is_weakly_connected(self):
        #     connected_components = nx.weakly_connected_components(self)
        #     root = [n for n in self.nodes if self.node[n]['head_idx'] == 0][0]
        #     main_component = [c for c in connected_components if root in c][0]
        #     nodes_to_delet = [n for n in self.node if not n in main_component]
        #
        #     for n in nodes_to_delet:
        #         self.remove_node(n)

