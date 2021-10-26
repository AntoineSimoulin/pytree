import itertools
import ast
import torch


class PackedTree:
    """
    :class:`~pytree.PackedTree` sequentialize the operations from a list of graph into a series of steps such as
    every nodes available are computed in a single operation. The computation is organized such that the final step
    operation return concatenation of all graphs root.
    embedding.

    """

    def __init__(self, tree_idx=None, word_idx=None):
        """
        Class attributes:
            - ``tree_idx`` : `optional`, `list` of length the number of nodes in the step. Two nodes with the same parent are assigned a tree_index value.
            - ``word_idx`` : `optional`, `list` of length the number of nodes in the step. Each node is assigned his word index in the vocabularly.
            - ``leafs_idx`` : `optional`, `list` of length the number of nodes in the step. For each step, assign the index his leaf in the word_idx_all vector.
        """

        if word_idx is None:
            word_idx_ = []
        self.word_idx_ = word_idx
        if tree_idx is None:
            tree_idx = []
        self.tree_idx_ = tree_idx

        self.device = torch.device('cpu')

    @property
    def word_idx(self):
        word_idx_ = [torch.tensor(w, device=self.device, dtype=torch.int64) for w in self.word_idx_ \
                     if any([x != -1 for x in w]) ]
        return [torch.masked_select(w, (w != -1)) for w in word_idx_]

    @property
    def tree_idx(self):
        return [torch.tensor(t, device=self.device, dtype=torch.int64) for t in self.tree_idx_ if len(t) > 0]

    def to(self, device):
        self.device = device
        return self

    @property
    def depth(self):
        return len(self.tree_idx)

    @property
    def hidden_idx(self):
        word_idx_ = [torch.tensor(w, device=self.device, dtype=torch.int64) for w in self.word_idx_][:-1]
        return [(w != -1).nonzero().t().squeeze() for w in word_idx_]

    # @property
    # def leaf_idx(self):
    #     return [list(set(range(len(t))) - set(t_1)) \
    #             for (t, t_1) in zip(self.tree_idx, [[]] + self.tree_idx[:-1])]
    #
    # @property
    # def inode_idx(self):
    #     return [list(set(t_1).intersection(set(range(len(w))))) \
    #             for (w, t_1) in zip(self.word_idx, [[]] + self.tree_idx[:-1])]

    def from_graphs(self, graphs, col="idx"):
        r"""Given a list of :class:`~nlp_toolbox.graphs.Graph` or a single instance, gather all the nodes
        that can be computed at the same step for a Tree RNN batching procedure.

        Parameters
        ----------
        ``graphs`` : Union[`Graph`, `list` [`Graph`]]
            `Graph` or list of `Graph`.

        ``N`` : `int`, default 1
            Number of dependant for each leaf nodes. Impact on the leaf_idx_all output.

        Example
        -------

        For the following graph:

        .. plot::

           from nlp_toolbox.graphs import Graph
           from PyTree.src.packed_tree import PackedTree

           import matplotlib.pyplot as plt
           import networkx as nx
           import matplotlib as mpl

           G_2 = Graph()
           G_2.add_edges_from([(2, 1), (3, 1), (4, 2), (5, 2)])
           for n in G_2.node:
               G_2.node[n]['word_idx'] = n

           pos = nx.drawing.nx_agraph.graphviz_layout(G_2.reverse(), prog='dot')

           fig = plt.figure(figsize=(5, 5), dpi=1000)
           plt.axis('off')
           fig.patch.set_alpha(0.)
           nx.draw_networkx(G_2.reverse(), pos=pos, labels={n: G_2.node[n]['word_idx'] for n in G_2.node},
                            node_color="#FCFCFC", node_size=1500)
           plt.show()


        The resulting computational map properties will be as followed. Nodes are computed step by step when their
        children have already been computed.

        +-------+----------+-----------+----------+
        | Step  | word idx | tree idx  | leaf idx |
        +=======+==========+===========+==========+
        | 0     | [4, 5]   | [0, 1]    | []       |
        +-------+----------+-----------+----------+
        | 1     | [2, 3]   | [0, 0, 1] | [0, 1]   |
        +-------+----------+-----------+----------+
        | 2     | [1]      | [0, 0]    | [0, 1]   |
        +-------+----------+-----------+----------+

        """

        if not isinstance(graphs, list):
            graphs = [graphs]

        packed_trees = [self.__graph_2_cm(G, col) for G in graphs]
        self = sum(packed_trees)
        return self

    @staticmethod
    def __graph_2_cm(G, col):
        tree_idx_ = [[] for _ in range(G.depth + 1)]
        word_idx_ = [[] for _ in range(G.depth + 1)]
        word_idx_[0] = [G.root]

        for d in range(G.depth):
            for n in word_idx_[d]:
                childrens = list(G.predecessors(n))
                word_idx_[d + 1].extend(childrens)
                tree_idx_[d + 1].extend([max(tree_idx_[d + 1], default=-1) + 1] * len(childrens))

        word_idx_ = [[G.nodes[ll][col] for ll in l] for l in word_idx_]
        tree_idx_ = tree_idx_[::-1]
        word_idx_ = word_idx_[::-1]

        return PackedTree(tree_idx_, word_idx_)

    def from_strings(self, strings):
        if not isinstance(strings, list):
            strings = [strings]
        packed_trees = [self.__string_2_cm(S) for S in strings]
        self = sum(packed_trees)
        return self

    @staticmethod
    def __string_2_cm(S):
        S = ast.literal_eval(S)
        word_idx_ = S[0]
        tree_idx_ = S[1]
        return PackedTree(tree_idx_, word_idx_)

    @staticmethod
    def extend_word_idx(l1, l2, floor_value=0):
        """Nested concatenation of two nested list. if the floor flag is activated,
        for each nested list, the max value  of the first one is added to the second one.
        """
        l_concat = []
        for ll_1, ll_2 in itertools.zip_longest(l1[::-1], l2[::-1], fillvalue=[]):
            floor_value = floor_value
            l_concat.append([x for x in ll_1] + [x + floor_value if x >= 0 else x for x in ll_2])
        return l_concat[::-1]

    @staticmethod
    def extend_tree_idx(l1, l2):
        """Nested concatenation of two nested list. if the floor flag is activated,
        for each nested list, the max value  of the first one is added to the second one.
        """
        l_concat = []
        for ll_1, ll_2 in itertools.zip_longest(l1[::-1], l2[::-1], fillvalue=[]):
            l_concat.append([x for x in ll_1] + [x + (max(ll_1, default=-1) + 1) for x in ll_2])
        return l_concat[::-1]

    def __add__(self, cm_2):
        r""":class:`~pytree.PackedTree` objects are additive. Given cm1 and cm2

        The parameters from the two objects are concatenated in a single :class:`~pytree.PackedTree`,
        which can be later process.

        Example
        -------

        .. code-block:: python

                >>> from nlp_toolbox.graphs import Graph
                >>> from PyTree.src.packed_tree import PackedTree

                >>> # create two graphs
                >>> G_1 = Graph()
                >>> G_1.add_edges_from([(5, 6), (3, 5), (1, 3), (2, 3), (4, 5), (7, 6), (8, 7), (9, 7), (10, 8), (11, 9)])
                >>> for n in G_1.node:
                >>>     G_1.node[n]['word_idx'] = n

                >>> G_2 = Graph()
                >>> G_2.add_edges_from([(2, 1), (3, 1), (4, 2), (5, 2)])
                >>> for n in G_2.node:
                >>>     G_2.node[n]['word_idx'] = n

                >>> cm = PackedTree().from_graphs(graphs=G_1)
                >>> print(cm.word_idx)
                [[1, 2, 10, 11], [3, 4, 8, 9], [5, 7], [6]]

                >>> cm = PackedTree().from_graphs(graphs=[G_1, G_2])
                >>> print(cm.word_idx)
                [[1, 2, 10, 11], [3, 4, 8, 9, 4, 5], [5, 7, 2, 3], [6, 1]]

        .. plot::

           from nlp_toolbox.graphs import Graph
           from PyTree.src.packed_tree import PackedTree

           import matplotlib.pyplot as plt
           import networkx as nx
           import matplotlib as mpl

           G_1 = Graph()
           G_1.add_edges_from([(5, 6), (3, 5), (1, 3), (2, 3), (4, 5), (7, 6), (8, 7), (9, 7), (10, 8), (11, 9)])
           for n in G_1.node:
               G_1.node[n]['word_idx'] = n

           G_2 = Graph()
           G_2.add_edges_from([(2, 1), (3, 1), (4, 2), (5, 2)])
           for n in G_2.node:
               G_2.node[n]['word_idx'] = n

           G = nx.disjoint_union(G_1, G_2)
           pos = nx.drawing.nx_agraph.graphviz_layout(G.reverse(), prog='dot')

           fig = plt.figure(figsize=(20, 5), dpi=1000)
           plt.axis('off')
           fig.patch.set_alpha(0.)
           nx.draw_networkx(G.reverse(), pos=pos, labels={n: G.node[n]['word_idx'] for n in G.node},
                            node_color="#FCFCFC", node_size=1500)
           plt.show()


        +------+--------------------+-----------------------+--------------------+
        | step | word idx           | tree idx              | leaf idx           |
        +======+====================+=======================+====================+
        | 0    | [1, 2, 10, 11]     | [0, 1, 2, 3]          | []                 |
        +------+--------------------+-----------------------+--------------------+
        | 1    | [3, 4, 8, 9, 4, 5] | [0, 0, 1, 2, 3, 4, 5] | [0, 1, 3, 4]       |
        +------+--------------------+-----------------------+--------------------+
        | 2    | [5, 7, 2, 3]       | [0, 0, 1, 1, 2, 2, 3] | [0, 1, 2, 3, 4, 5] |
        +------+--------------------+-----------------------+--------------------+
        | 3    | [6, 1]             | [0, 0, 1, 1]          | [0, 1, 2, 3]       |
        +------+--------------------+-----------------------+--------------------+

        """
        if cm_2 is None:
            return self
        # if max([max([x for x in ll if x >= 0], default=0) for ll in cm_2.word_idx_]) >= sum([len([x for x in ll if x >= 0]) for ll in cm_2.word_idx_]):
        #     print(max([max([x for x in ll if x >= 0], default=0) for ll in cm_2.word_idx_]), sum([len([x for x in ll if x >= 0]) for ll in cm_2.word_idx_]))
        word_idx_ = self.extend_word_idx(self.word_idx_, cm_2.word_idx_, floor_value=sum([len([x for x in ll if x >= 0]) for ll in self.word_idx_]))

        # if max([max([x for x in ll if x >= 0], default=0) for ll in self.word_idx_]) >= sum([len([x for x in ll if x >= 0]) for ll in self.word_idx_]):
        #     print(max([max([x for x in ll if x >= 0], default=0) for ll in self.word_idx_]), sum([len([x for x in ll if x >= 0]) for ll in self.word_idx_]))
        tree_idx_ = self.extend_tree_idx(self.tree_idx_, cm_2.tree_idx_)
        return PackedTree(tree_idx_, word_idx_)

    def __radd__(self, cm_2):
        if cm_2 == 0:
            return self
        else:
            return self.__add__(cm_2)

    def __str__(self):
        return "{}".format(str((self.word_idx_, self.tree_idx_)))
