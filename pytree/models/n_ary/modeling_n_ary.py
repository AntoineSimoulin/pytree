import torch.nn as nn
import torch
from typing import List, Tuple, Optional, overload, Union, cast
from torch import Tensor
from transformers import BertModel
from pytree.data.packed_tree import PackedTree


class TreeEmbeddings(nn.Module):

    def __init__(self, config):
        super(TreeEmbeddings, self).__init__()
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
            # cat_inputs = torch.cat(raw_inputs)
            embeds = self.embeddings(raw_inputs)
        return embeds

    def xavier_init_weights(self):
        nn.init.xavier_uniform_(self.embeddings.weight.data, gain=1.0)


class NaryTree(nn.Module):
    def __init__(self, config):
        super(NaryTree, self).__init__()
        self.config = config
        self.embeddings = TreeEmbeddings(config)
        if config.cell_type == 'lstm':
            self.encoder = NaryTreeLSTMEncoder(config)
        elif config.cell_type == 'gru':
            self.encoder = NaryTreeGRUEncoder(config)

    def forward(self, inputs):
        embeds = self.embeddings(inputs['input_ids'])
        hidden, h_root = self.encoder(embeds, inputs['tree_ids'].to(embeds.device), inputs['tree_ids_r'].to(embeds.device), inputs['tree_ids_l'].to(embeds.device))
        return hidden, h_root


class TreeLSTM(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    
    def __init__(self, config):
        super(TreeLSTM, self).__init__()
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.vocab_size = config.vocab_size

    def xavier_init_weights(self):
        # nn.init.xavier_uniform_(self.embeddings.weight.data, gain=1.0)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data, gain=1.0)
            if 'bias' in name:
                param.data.fill_(0)
      
    def forward(self,
                input_ids: Union[Tensor, PackedTree],
                tree_ids: Tensor = None,
                tree_ids_r: Tensor = None,
                tree_ids_l: Tensor = None,
                hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Union[Tensor, PackedTree], Tuple[Tensor, Tensor]]:
        # if isinstance(orig_input, PackedTrees):
        batch_size = input_ids.size(0)  # if self.batch_first else input.size(1)
        n_steps = tree_ids.size(1)
        sequence_length = input_ids.size(1)
        # else:
        #   batch_size = input.size(0) if self.batch_first else input.size(1)
        #   n_steps = tree_ids.size(0)
          
        if hx is None:
            h_zeros = torch.zeros(batch_size, sequence_length, self.hidden_size,
                                  dtype=input_ids.dtype, device=input_ids.device)
            c_zeros = torch.zeros(batch_size, sequence_length, self.hidden_size,
                                  dtype=input_ids.dtype, device=input_ids.device)
            hx = (h_zeros, c_zeros)

        for step in range(n_steps):
            hx = self.tree_lstm_cell(input_ids, hx, tree_ids[:, step, :], tree_ids_r[:, step, :], tree_ids_l[:, step, :])  # .select(0, step)
        roots = tree_ids[:, -1, :].max(axis=1)[0]
        h_root = torch.gather(hx[0], 1, roots.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.hidden_size)).squeeze()
        return hx, h_root


class NaryTreeLSTMCell(nn.Module):

    def __init__(self, config):
        super(NaryTreeLSTMCell, self).__init__()
        self.N = config.N
        self.ioux = nn.Linear(config.embedding_size, 3 * config.hidden_size, bias=True)
        self.iouh = nn.ModuleList([nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False) for i in range(config.N)])
        self.fx = nn.Linear(config.embedding_size, config.hidden_size, bias=True)
        self.fh = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size, bias=False) for i in range(config.N * config.N)])
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size

    def forward(self, x, hx, tree_ids_d, tree_ids_dr, tree_ids_dl):

        # import pdb; pdb.set_trace()
        index = tree_ids_d.unsqueeze(-1).repeat(1, 1, self.hidden_size)
        index_r = tree_ids_dr.unsqueeze(-1).repeat(1, 1, self.hidden_size)
        index_l = tree_ids_dl.unsqueeze(-1).repeat(1, 1, self.hidden_size)
        updated_nodes = torch.zeros_like(index).scatter_add_(1, index, torch.ones_like(index))
        updated_nodes[:, 0, :] = 0
        updated_nodes = updated_nodes.bool()

        # iou_x = self.ioux(x)
        iou = self.ioux(x)
        # print('shape ioux', iou_x.shape)
        iou_hr = self.iouh[0](hx[0])
        # print('iouhr shape', iou_hr.shape)
        iou_hl = self.iouh[1](hx[0])
        # iou = iou_x + \
        #   torch.zeros_like(iou_x).scatter_add_(1, index_r, iou_hr) + \
        #   torch.zeros_like(iou_x).scatter_add_(1, index_l, iou_hl)
        # print('index r shape', index_r.shape)
        # print('index_r', index_r)
        iou = torch.scatter_add(iou, 1, index_r.repeat(1, 1, 3), iou_hr)
        iou = torch.scatter_add(iou, 1, index_l.repeat(1, 1, 3), iou_hl)
        # iou = iou_x.scatter_add_(1, index_r, iou_hr)
        # iou = iou_x.scatter_add_(1, index_l, iou_hl)
        
        i, o, u = torch.split(iou, iou.size(-1) // 3, dim=-1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        
        f = self.fx(x).gather(1, index) + \
            self.fh[0](hx[0]).gather(1, index_r) + \
            self.fh[1](hx[0]).gather(1, index_r) + \
            self.fh[2](hx[0]).gather(1, index_l) + \
            self.fh[3](hx[0]).gather(1, index_l)
        f = torch.sigmoid(f)
        fc = torch.mul(f, hx[1])
        
        # c = torch.mul(i, u) + torch.zeros_like(fc).scatter_add_(1, index, fc)
        c = torch.mul(i, u)
        c = c.scatter_add_(1, index, fc)
        h = torch.mul(o, torch.tanh(c))
        
        # h = hx[0].masked_scatter_(index.bool(), h)
        h = torch.where(updated_nodes, h, hx[0])  # index.bool()
        # c = hx[1].masked_scatter_(index.bool(), c)
        c = torch.where(updated_nodes, c, hx[1])  # index.bool()

        return h, c


class NaryTreeLSTMEncoder(TreeLSTM):
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
        super(NaryTreeLSTMEncoder, self).__init__(config)
        self.tree_lstm_cell = NaryTreeLSTMCell(config)
        if config.xavier_init:
            self.xavier_init_weights()