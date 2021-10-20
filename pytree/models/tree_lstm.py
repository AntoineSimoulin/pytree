import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TreeLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, p_dropout=0.):
        """
        Class attributes:
            - ``embedding_size``: `int`. Dimension of the embeddings.
            - ``hidden_size``:  `int`. Dimension of the Tree LSTM hidden layer
            - ``vocab_size``: `int`. Dimension of the vocabulary.
        """
        super(TreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.N = 1
        self.p_dropout = p_dropout
        self.dropout = nn.Dropout(p=p_dropout)

    def xavier_init_weights(self):
        # nn.init.xavier_uniform_(self.embeddings.weight.data, gain=1.0)
        for name, param in self.tree_lstm_cell.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data, gain=1.0)
            if 'bias' in name:
                param.data.fill_(0)

    # def load_pretrained_embeddings(self, embeddings_weights, requires_grad=False):
    #     self.embeddings = nn.Embedding.from_pretrained(embeddings_weights, sparse=True)
    #     self.embeddings.weight.requires_grad = requires_grad

    def forward(self, x, packed_tree, hx=None):
        """
        Loop through all steps and output the state and hidden vector for the root nodes in the batch.
        """
        tree_idx, w_idx, hidden_idx, max_depth = packed_tree.tree_idx, packed_tree.word_idx, packed_tree.hidden_idx, packed_tree.depth
        # x = torch.cat((x, torch.zeros((1, x.shape[1]), device=x.device)), 0)
        # idx_minus_two = torch.tensor([max(x.shape[0] - 1, 0)], device=x.device)

        if hx is None:
            h0 = torch.zeros(1, self.hidden_size, dtype=torch.float32, device=w_idx[0].device, requires_grad=True)
            c0 = torch.zeros(1, self.hidden_size, dtype=torch.float32, device=w_idx[0].device, requires_grad=True)
            hx = (h0, c0)

        h, c = None, None
        for d in range(max_depth):
            # w_idx_no_minus_two = torch.where(w_idx[d] != -2, w_idx[d], idx_minus_two.repeat(w_idx[d].shape[0]))
            # x_d = torch.index_select(x, 0, w_idx_no_minus_two)
            if (not w_idx[d].nelement() == 0) and (torch.min(w_idx[d]) == -2):
                idx_greater_than_two = torch.where(w_idx[d] != -2)[0]
                w_idx_no_minus_two = torch.index_select(w_idx[d], 0, idx_greater_than_two)
                zeros = torch.zeros((w_idx[d].shape[0], x.shape[1]), device=w_idx[0].device)
                x_d = zeros.index_add(0, idx_greater_than_two, torch.index_select(x, 0, w_idx_no_minus_two))
            else:
                x_d = torch.index_select(x, 0, w_idx[d])
            # x_d = F.normalize(x_d, p=2, dim=1)
            h, c = self.tree_lstm_cell(x_d, h, c, hx, tree_idx[d], hidden_idx[d])
        if self.p_dropout > 0.:
            h, c = self.dropout(h), self.dropout(c)
        return h, c


class TreeLSTMCell(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(TreeLSTMCell, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

    @staticmethod
    def get_idx(tree_idx):
        tree_idx_out, tree_inverse_idx = \
            torch.unique_consecutive(tree_idx, return_inverse=True)
        return tree_idx_out.shape[0], tree_inverse_idx

    def sum_idx_(self, input, tree_idx):
        if tree_idx.nelement() == 0:
            return input
        len_idx, idx_ = self.get_idx(tree_idx)
        input_idx_add = input.index_add(0, idx_, input)
        input = input_idx_add.add(-1, input)
        input = torch.index_select(input, 0, torch.tensor(range(len_idx)).to(tree_idx.device))
        return input

    @staticmethod
    def repeat_idx_(input, tree_idx):
        input = torch.index_select(input, 0, tree_idx)
        return input

    def add_zeros_idx_(self, input, leafs_idx, tree_idx):
        if leafs_idx.nelement() == 0:
            return input
        zeros = torch.zeros((len(tree_idx), self.hidden_size), dtype=torch.float32,
                            device=tree_idx.device, requires_grad=True)
        input = zeros.index_add(0, leafs_idx, input)
        return input

    @staticmethod
    def replace_idx_(input_a, input_b, leafs_idx):
        if leafs_idx.nelement() == 0:
            return input_a
        input_a = input_a.index_copy(0, leafs_idx, input_b)
        return input_a


class ChildSumTreeLSTMCell(TreeLSTMCell):

    def __init__(self, embedding_size, hidden_size):
        super(ChildSumTreeLSTMCell, self).__init__(embedding_size, hidden_size)
        self.ioux = nn.Linear(self.embedding_size, 3 * self.hidden_size, bias=False)
        self.iouh = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.fx = nn.Linear(self.embedding_size, self.hidden_size, bias=False)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x, h, c, hx, tree_idx, hidden_idx):

        h = self.replace_idx_(hx[0].repeat(tree_idx.shape[0], 1), h, hidden_idx)
        c = self.replace_idx_(hx[1].repeat(tree_idx.shape[0], 1), c, hidden_idx)

        h_sum = self.sum_idx_(h, tree_idx)

        iou = self.ioux(x) + self.iouh(h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = self.fh(h) + self.repeat_idx_(self.fx(x), tree_idx)
        f = torch.sigmoid(f)
        fc = torch.mul(f, c)

        c = torch.mul(i, u) + self.sum_idx_(fc, tree_idx)
        h = torch.mul(o, torch.tanh(c))

        return h, c


class ChildSumTreeLSTM(TreeLSTM):
    r"""
    .. math::
       :nowrap:

       \begin{align}
           \tilde{h}_j &= \sum_{k \in C(j)} h_k, \\
           i_j &=\sigma \left( W^{(i)} x_j + U^{(i)} \tilde{h}_j + b^{(i)} \right), \\
           f_{jk} &= \sigma\left( W^{(f)} x_j + U^{(f)} h_k + b^{(f)} \right),\\
           o_j &= \sigma \left( W^{(o)} x_j + U^{(o)} \tilde{h}_j  + b^{(o)} \right), \\
           u_j &= \tanh\left( W^{(u)} x_j + U^{(u)} \tilde{h}_j  + b^{(u)} \right), \\
           c_j &= i_j \odot u_j + \sum_{k\in C(j)} f_{jk} \odot c_{k}, \\
           h_j &= o_j \odot \tanh(c_j),
       \end{align}

    """

    def __init__(self, embedding_size, hidden_size, vocab_size, xavier_init=False, p_dropout=0.):
        """
        Class attributes:
            - ``embedding_size``: `int`. Dimension of the embeddings.
            - ``hidden_size``:  `int`. Dimension of the Tree LSTM hidden layer
            - ``vocab_size``: `int`. Dimension of the vocabulary.
            - ``xavier_init``: `bool`, default 1. Whether to intiate networks weights using the glorot procedure.
        """
        super(ChildSumTreeLSTM, self).__init__(embedding_size, hidden_size, vocab_size, p_dropout)
        self.tree_lstm_cell = ChildSumTreeLSTMCell(embedding_size, hidden_size)
        if xavier_init:
            self.xavier_init_weights()


class ChildSumTreeAttentiveLSTMCell(ChildSumTreeLSTMCell):

    def __init__(self, embedding_size, hidden_size, keys_size):
        super(ChildSumTreeAttentiveLSTMCell, self).__init__(embedding_size, hidden_size)
        self.attention = SelfAttention(embedding_size, hidden_size, keys_size)

    def forward(self, x, h, c, hx, tree_idx, hidden_idx):

        h = self.replace_idx_(hx[0].repeat(tree_idx.shape[0], 1), h, hidden_idx)
        c = self.replace_idx_(hx[1].repeat(tree_idx.shape[0], 1), c, hidden_idx)

        h_sum, attn = self.attention(h, x, tree_idx)

        iou = self.ioux(x) + self.iouh(h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = self.fh(h) + self.repeat_idx_(self.fx(x), tree_idx)
        f = torch.sigmoid(f)
        fc = torch.mul(f, c)

        c = torch.mul(i, u) + self.sum_idx_(fc, tree_idx)
        h = torch.mul(o, torch.tanh(c))

        return h, c


class ChildSumAttentiveTreeLSTM(TreeLSTM):
    def __init__(self, embedding_size, hidden_size, keys_size, vocab_size, xavier_init=False, p_dropout=0.):
        super(ChildSumAttentiveTreeLSTM, self).__init__(embedding_size, hidden_size, vocab_size, p_dropout)
        self.tree_lstm_cell = ChildSumTreeAttentiveLSTMCell(embedding_size, hidden_size, keys_size)
        if xavier_init:
            self.xavier_init_weights()


class ChildSumTreeGRUCell(TreeLSTMCell):

    def __init__(self, embedding_size, hidden_size):
        super(ChildSumTreeGRUCell, self).__init__(embedding_size, hidden_size)
        # self.zrh = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        # self.zrx = nn.Linear(self.embedding_size, 2 * self.hidden_size, bias=False)
        self.zh = nn.Linear(self.hidden_size, self.hidden_size)
        self.rh = nn.Linear(self.hidden_size, self.hidden_size)
        self.wh = nn.Linear(self.hidden_size, self.hidden_size)
        self.zrwx = nn.Linear(self.embedding_size, 3 * self.hidden_size, bias=False)

    def forward(self, x, h, c, hx, tree_idx, hidden_idx):

        h = self.replace_idx_(hx[0].repeat(tree_idx.shape[0], 1), h, hidden_idx)
        h_sum = self.sum_idx_(h, tree_idx)

        # zr = self.zrh(h_sum) + self.zrx(x)
        # z, r = torch.split(zr, zr.size(1) // 2, dim=1)
        # z, r = torch.sigmoid(z), torch.sigmoid(r)

        zrwx = self.zrwx(x)
        zx, rx, wx = torch.split(zrwx, zrwx.size(1) // 3, dim=1)

        z = torch.sigmoid(self.zh(h_sum) + zx)  # self.zx(x)
        r = torch.sigmoid(self.rh(h) + self.repeat_idx_(rx, tree_idx))  # self.rx(x)
        r = self.sum_idx_(torch.mul(r, h), tree_idx)

        h_hat = self.wh(r) + wx  # self.wx(x)
        h_hat = torch.tanh(h_hat)
        h = torch.mul((1. - z), h_sum) + torch.mul(z, h_hat)

        return h, None


class ChildSumTreeGRU(TreeLSTM):
    r"""
    .. math::
       :nowrap:

       \begin{align}
           \tilde{h}_j &= \sum_{k \in C(j)} h_k, \\
           i_j &=\sigma \left( W^{(i)} x_j + U^{(i)} \tilde{h}_j + b^{(i)} \right), \\
           f_{jk} &= \sigma\left( W^{(f)} x_j + U^{(f)} h_k + b^{(f)} \right),\\
           o_j &= \sigma \left( W^{(o)} x_j + U^{(o)} \tilde{h}_j  + b^{(o)} \right), \\
           u_j &= \tanh\left( W^{(u)} x_j + U^{(u)} \tilde{h}_j  + b^{(u)} \right), \\
           c_j &= i_j \odot u_j + \sum_{k\in C(j)} f_{jk} \odot c_{k}, \\
           h_j &= o_j \odot \tanh(c_j),
       \end{align}

    """

    def __init__(self, embedding_size, hidden_size, vocab_size, xavier_init=False, p_dropout=0.):
        """
        Class attributes:
            - ``embedding_size``: `int`. Dimension of the embeddings.
            - ``hidden_size``:  `int`. Dimension of the Tree LSTM hidden layer
            - ``vocab_size``: `int`. Dimension of the vocabulary.
            - ``xavier_init``: `bool`, default 1. Whether to intiate networks weights using the glorot procedure.
        """
        super(ChildSumTreeGRU, self).__init__(embedding_size, hidden_size, vocab_size, p_dropout)
        self.tree_lstm_cell = ChildSumTreeGRUCell(embedding_size, hidden_size)
        if xavier_init:
            self.xavier_init_weights()


class ChildSumAttentiveTreeGRUCell(ChildSumTreeGRUCell):

    def __init__(self, embedding_size, hidden_size, keys_size):
        super(ChildSumAttentiveTreeGRUCell, self).__init__(embedding_size, hidden_size)
        self.attention = SelfAttention(embedding_size, hidden_size, keys_size)

    def forward(self, x, h, c, hx, tree_idx, hidden_idx):

        h = self.replace_idx_(hx[0].repeat(tree_idx.shape[0], 1), h, hidden_idx)
        # h_sum = self.sum_idx_(h, tree_idx)
        h_sum, attn = self.attention(h, x, tree_idx)

        # zr = self.zrh(h_sum) + self.zrx(x)
        # z, r = torch.split(zr, zr.size(1) // 2, dim=1)
        # z, r = torch.sigmoid(z), torch.sigmoid(r)

        zrwx = self.zrwx(x)
        zx, rx, wx = torch.split(zrwx, zrwx.size(1) // 3, dim=1)

        z = torch.sigmoid(self.zh(h_sum) + zx)  # self.zx(x)
        r = torch.sigmoid(self.rh(h) + self.repeat_idx_(rx, tree_idx))  # self.rx(x)
        r = self.sum_idx_(torch.mul(r, h), tree_idx)

        h_hat = self.wh(r) + wx  # self.wx(x)
        h_hat = torch.tanh(h_hat)
        h = torch.mul((1. - z), h_sum) + torch.mul(z, h_hat)

        return h, None


class ChildSumAttentiveTreeGRU(TreeLSTM):
    def __init__(self, embedding_size, hidden_size, keys_size, vocab_size, xavier_init=False, p_dropout=0.):
        super(ChildSumAttentiveTreeGRU, self).__init__(embedding_size, hidden_size, vocab_size, p_dropout)
        self.tree_lstm_cell = ChildSumAttentiveTreeGRUCell(embedding_size, hidden_size, keys_size)
        if xavier_init:
            self.xavier_init_weights()


class NaryTreeLSTMCell(TreeLSTMCell):

    def __init__(self, embedding_size, hidden_size, N):
        super(NaryTreeLSTMCell, self).__init__(embedding_size, hidden_size)
        self.N = N
        self.ioux = nn.Linear(self.embedding_size, 3 * self.hidden_size, bias=False)
        self.iouh = nn.Linear(N * self.hidden_size, 3 * self.hidden_size)
        self.fx = nn.Linear(self.embedding_size, self.hidden_size, bias=False)
        self.fh = nn.Linear(N * self.hidden_size, N * self.hidden_size)

    def forward(self, x, h, c, hx, tree_idx, hidden_idx):

        h = self.replace_idx_(hx[0].repeat(tree_idx.shape[0], 1), h, hidden_idx).view(-1, self.N * self.hidden_size)
        c = self.replace_idx_(hx[1].repeat(tree_idx.shape[0], 1), c, hidden_idx).view(-1, self.N * self.hidden_size)

        iou = self.ioux(x) + self.iouh(h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = self.fh(h) + self.fx(x).repeat(1, self.N)
        f = torch.sigmoid(f)
        fc = torch.mul(f, c)

        c = torch.mul(i, u) + torch.sum(
            torch.stack(
                torch.split(fc, fc.size(1) // self.N, dim=1),
                dim=0),
            dim=0, keepdim=True).squeeze()
        h = torch.mul(o, torch.tanh(c))
        return h, c


class NaryTreeLSTM(TreeLSTM):
    r"""
    .. math::
       :nowrap:

       \begin{align}
           i_j &=\sigma \left( W^{(i)} x_j + \sum_{\ell=1}^N U^{(i)}_\ell h_{j\ell} + b^{(i)} \right),\\
           f_{jk} &= \sigma\left( W^{(f)} x_j + \sum_{\ell=1}^N U^{(f)}_{k\ell} h_{j\ell} + b^{(f)} \right),\\
           o_j &= \sigma \left( W^{(o)} x_j + \sum_{\ell=1}^N U^{(o)}_\ell h_{j\ell}  + b^{(o)} \right), \\
           u_j &= \tanh\left( W^{(u)} x_j + \sum_{\ell=1}^N U^{(u)}_\ell h_{j\ell}  + b^{(u)} \right), \\
           c_j &= i_j \odot u_j + \sum_{\ell=1}^N f_{j\ell} \odot c_{j\ell}, \\
           h_j &= o_j \odot \tanh(c_j),
       \end{align}

    """

    def __init__(self, embedding_size, hidden_size, vocab_size, N, xavier_init=False, p_dropout=0.):
        super(NaryTreeLSTM, self).__init__(embedding_size, hidden_size, vocab_size, p_dropout)
        self.tree_lstm_cell = NaryTreeLSTMCell(embedding_size, hidden_size, N)
        self.N = N
        if xavier_init:
            self.xavier_init_weights()

