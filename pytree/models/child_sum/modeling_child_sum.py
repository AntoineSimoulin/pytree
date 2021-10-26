import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel
from pytree.parsers.utils import GloveTokenizer


class TreeLSTM(nn.Module):
    def __init__(self, config):
        """
        Class attributes:
            - ``embedding_size``: `int`. Dimension of the embeddings.
            - ``hidden_size``:  `int`. Dimension of the Tree LSTM hidden layer
            - ``vocab_size``: `int`. Dimension of the vocabulary.
        """
        super(TreeLSTM, self).__init__()
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.vocab_size = config.vocab_size
        self.N = 1
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.dropout = nn.Dropout(p=self.hidden_dropout_prob)

    def forward(self, x, packed_tree, hx=None):
        """
        Loop through all steps and output the state and hidden vector for the root nodes in the batch.
        """
        tree_idx, w_idx, hidden_idx, max_depth = \
            packed_tree.tree_idx, packed_tree.word_idx, packed_tree.hidden_idx, packed_tree.depth

        if hx is None:
            h0 = torch.zeros(1, self.hidden_size, dtype=torch.float32, device=w_idx[0].device, requires_grad=True)
            c0 = torch.zeros(1, self.hidden_size, dtype=torch.float32, device=w_idx[0].device, requires_grad=True)
            hx = (h0, c0)

        h, c = None, None
        for d in range(max_depth):
            if (not w_idx[d].nelement() == 0) and (torch.min(w_idx[d]) == -2):
                idx_greater_than_two = torch.where(w_idx[d] != -2)[0]
                w_idx_no_minus_two = torch.index_select(w_idx[d], 0, idx_greater_than_two)
                zeros = torch.zeros((w_idx[d].shape[0], x.shape[1]), device=w_idx[0].device)
                x_d = zeros.index_add(0, idx_greater_than_two, torch.index_select(x, 0, w_idx_no_minus_two))
            else:
                x_d = torch.index_select(x, 0, w_idx[d])
            # x_d = F.normalize(x_d, p=2, dim=1)
            h, c = self.tree_lstm_cell(x_d, h, c, hx, tree_idx[d], hidden_idx[d])
        if self.hidden_dropout_prob > 0.:
            h, c = self.dropout(h), self.dropout(c)
        return h, c


class TreeLSTMCell(nn.Module):
    def __init__(self, config):
        super(TreeLSTMCell, self).__init__()
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size

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

    def xavier_init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data, gain=1.0)
            if 'bias' in name:
                param.data.fill_(0)


class ChildSumTreeLSTMCell(TreeLSTMCell):

    def __init__(self, config):
        super(ChildSumTreeLSTMCell, self).__init__(config)
        self.ioux = nn.Linear(self.embedding_size, 3 * self.hidden_size, bias=False)
        self.iouh = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.fx = nn.Linear(self.embedding_size, self.hidden_size, bias=False)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)
        if config.xavier_init:
                self.xavier_init_weights()

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


class ChildSumTreeLSTMEncoder(TreeLSTM):
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

    def __init__(self, config):
        """
        Class attributes:
            - ``embedding_size``: `int`. Dimension of the embeddings.
            - ``hidden_size``:  `int`. Dimension of the Tree LSTM hidden layer
            - ``vocab_size``: `int`. Dimension of the vocabulary.
            - ``xavier_init``: `bool`, default 1. Whether to intiate networks weights using the glorot procedure.
        """
        super(ChildSumTreeLSTMEncoder, self).__init__(config)
        self.tree_lstm_cell = ChildSumTreeLSTMCell(config)


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


class ChildSumAttentiveTreeLSTMEncoder(TreeLSTM):
    def __init__(self, embedding_size, hidden_size, keys_size, vocab_size, xavier_init=False, p_dropout=0.):
        super(ChildSumAttentiveTreeLSTMEncoder, self).__init__(embedding_size, hidden_size, vocab_size, p_dropout)
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


class ChildSumTreeGRUEncoder(TreeLSTM):
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
        super(ChildSumTreeGRUEncoder, self).__init__(embedding_size, hidden_size, vocab_size, p_dropout)
        self.tree_lstm_cell = ChildSumTreeGRUCell(embedding_size, hidden_size)
        if xavier_init:
            self.xavier_init_weights()


class Attention(nn.Module):
    """
    See :cite:`vaswani_17`

    .. bibliography:: ../references.bib

    .. image:: ../_static/imgs/scaled-dot-product-attention.png
       :height: 200px
       :align: left
    """
    def __init__(self, keys_size):
        super(Attention, self).__init__()
        self.keys_size = keys_size
        # self.w = torch.Parameter(torch.rand(1, keys_size))

    def forward(self, query, keys, values, tree_idx):
        mask = torch.zeros((query.shape[0], tree_idx.shape[0]), device=query.device)\
            .index_add(0, tree_idx, torch.eye(tree_idx.shape[0], device=query.device)).bool()
        # align = torch.mm(query, keys.t()) / np.sqrt(self.keys_size)
        query_n = F.normalize(query, dim=1, p=2)
        keys_n = F.normalize(keys, dim=1, p=2)
        align = torch.mm(query_n, keys_n.t())

        align.masked_fill_(~mask, -float("Inf"))
        alpha = torch.softmax(align, dim=1)
        att = torch.mm(alpha, values)
        return att, alpha.sum(dim=0)


class SelfAttention(nn.Module):

    def __init__(self, embedding_size, hidden_size, keys_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.keys_size = keys_size
        self.W_k = nn.Linear(hidden_size, keys_size)
        self.W_q = nn.Linear(embedding_size, keys_size)
        self.attention = Attention(keys_size)

    def forward(self, child_h, x, tree_idx):
        query = self.W_q(x)
        keys = self.W_k(child_h)
        att, alpha = self.attention(query, keys, child_h, tree_idx)
        return att, alpha


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


class ChildSumAttentiveTreeGRUEncoder(TreeLSTM):
    def __init__(self, embedding_size, hidden_size, keys_size, vocab_size, xavier_init=False, p_dropout=0.):
        super(ChildSumAttentiveTreeGRUEncoder, self).__init__(embedding_size, hidden_size, vocab_size, p_dropout)
        self.tree_lstm_cell = ChildSumAttentiveTreeGRUCell(embedding_size, hidden_size, keys_size)
        if xavier_init:
            self.xavier_init_weights()


class ChildSumTreeEmbeddings(nn.Module):

    def __init__(self, config):
        super(ChildSumTreeEmbeddings, self).__init__()
        self.use_bert = config.use_bert
        self.tune_bert = config.tune_bert
        self.normalize_bert_embeddings = config.normalize_bert_embeddings

        # embeddings
        if self.use_bert:
            self.bert = BertModel.from_pretrained(config.pretrained_model_name_or_path)
            for name, param in self.bert.named_parameters():
                param.requires_grad = self.tune_bert
        else:
            self.embeddings = nn.Embedding(config.vocab_size, config.embedding_size)  # , sparse=True
            nn.init.xavier_uniform_(self.embeddings.weight.data, gain=1.0)
            self.embeddings.weight.requires_grad = True

        if config.xavier_init:
            self.xavier_init_weights()
        
    def load_pretrained_embeddings(self, embeddings_weights, requires_grad=False):
        self.embeddings = nn.Embedding.from_pretrained(embeddings_weights, sparse=True)
        self.embeddings.weight.requires_grad = requires_grad
    
    def forward(self, raw_inputs=None, packed_tree=None, bert_inputs=None):
        if self.use_bert:
            tokens_tensor, tokens_type_ids, attention_mask, sum_idx = bert_inputs
            if self.tune_bert:
                outputs = self.bert(input_ids=tokens_tensor,
                                    token_type_ids=tokens_type_ids,
                                    attention_mask=attention_mask)[0]
            else:
                with torch.no_grad():
                    outputs = self.bert(input_ids=tokens_tensor,
                                        token_type_ids=tokens_type_ids,
                                        attention_mask=attention_mask)[0]
            if self.normalize_bert_embeddings:
                outputs = F.normalize(outputs, p=2, dim=2)
            cat_inputs = torch.reshape(outputs, (-1, outputs.shape[2]))
            embeds = torch.index_select(cat_inputs, 0, sum_idx.long())
            # embeds = torch.sigmoid(self.projection(embeds))
        else:
            cat_inputs = torch.cat(raw_inputs)
            embeds = self.embeddings(cat_inputs)
        return embeds

    def xavier_init_weights(self):
        nn.init.xavier_uniform_(self.embeddings.weight.data, gain=1.0)


class ChildSumTree(nn.Module):
    def __init__(self, config):
        super(ChildSumTree, self).__init__()
        self.config = config
        self.embeddings = ChildSumTreeEmbeddings(config)
        if config.cell_type == 'lstm' and config.use_attention:
            self.encoder = ChildSumAttentiveTreeLSTMEncoder(config)
        elif config.cell_type == 'gru' and config.use_attention:
            self.encoder = ChildSumAttentiveTreeGRUEncoder(config)
        elif config.cell_type == 'lstm' and not config.use_attention:
            self.encoder = ChildSumTreeLSTMEncoder(config)
        elif config.cell_type == 'gru' and not config.use_attention:
            self.encoder = ChildSumTreeGRUEncoder(config)

    def forward(self, inputs):
        embeds = self.embeddings(inputs['input_ids'])
        hidden, _ = self.encoder(embeds, inputs['packed_tree'])
        return hidden