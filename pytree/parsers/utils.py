import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk
import numpy as np



class Graph(nx.DiGraph):
    def __init__(self):
        super(Graph, self).__init__()
        # self.seq = []

    @property
    def depth(self):
        nodes_depth = []
        for n in self.node:
            d = nx.shortest_path_length(self, n, self.root)
            nodes_depth.append(d)
            self.nodes[n]['depth'] = d
        max_depth = max(nodes_depth)
        return max_depth

    @property
    def root(self):
        return list(nx.topological_sort(self))[-1]

    def parents_as_children(self):
        for n in list(self.node):
            if len(list(self.predecessors(n))) > 0:
                self.add_node(len(self.node) + 1)
                for k, v in self.nodes[n].items():
                    self.nodes[len(self.node)][k] = v
                self.add_edges_from([(len(self.node), n)])
        return self

    def add_gost_childrens(self, N):
        for n in list(self.node):
            if len(list(self.predecessors(n))) == 0:
                for _ in range(N):
                    self.add_node(max(self.node) + 1)
                    # self.nodes[len(self.node)]['text'] = "__gost__"
                    self.nodes[max(self.node)]['idx'] = -1
                    self.add_edges_from([(max(self.node), n)])
        return self

    def draw(self, col=['idx']):
        pos = nx.drawing.nx_agraph.graphviz_layout(self.reverse(), prog='dot')

        fig = plt.figure(figsize=(10, 5), dpi=300)
        plt.axis('off')
        fig.patch.set_alpha(0.)
        nx.draw_networkx(self.reverse(), pos=pos, labels={n: ' '.join([self.node[n][c] for c in col]) for n in self.node},
                         node_color="#FCFCFC", node_size=1500)
        plt.show()


class GloveTokenizer:

    def __init__(self, glove_file_path, vocab_size=None):
        self.glove_file_path = glove_file_path
        vocab, self.embeddings_arr = self._read_embedding_file(glove_file_path, vocab_size)
        self.unk_token_id = 1
        self.unk_token = "UNK"
        self.pad_token_id = 0
        self.pad_token = "PAD"
        self.w2idx = {w: i for (i, w) in enumerate(vocab, 2)}
        self.idx2w = {i: w for (i, w) in enumerate(vocab, 2)}

    @property
    def vocab_size(self):
        return len(self.w2idx)

    @property
    def vocab(self):
        return sorted(self.w2idx.keys(), reverse=False)  # sorted(self.w2idx, key=self.w2idx.get, reverse=True)

    @staticmethod
    def _read_embedding_file(embeddings_f_path, vocab_size):
        # num_lines = sum(1 for _ in open(embeddings_f_path))
        num_lines = min(vocab_size, 2000000)
        with open(embeddings_f_path, 'rb') as f:
            for i in tqdm(range(0, num_lines - 2), total=num_lines, desc="load embedding file"):
                line = next(f)
                values = line.decode('utf-8').split()
                if i == 0:
                    embeddings_size = len(values[1:])
                    w_emb = np.zeros((num_lines, embeddings_size), dtype='float32')
                    w = []
                word_len = len(values) - embeddings_size
                w.append(' '.join(values[:word_len]))
                w_emb[i + 2] = values[word_len:]

        w_emb[0] = np.zeros_like(w_emb[1])  # np.mean(w_emb[2:, :], axis=0)
        w_emb[1] = np.zeros_like(w_emb[1])
        return w, w_emb

    @staticmethod
    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        return tokens

    def convert_ids_to_tokens(self, token_ids):
        tokens = [self.idx2w.get(ids, self.unk_token) for ids in token_ids]
        return tokens

    def convert_tokens_to_ids(self, tokens):
        ids = [self.w2idx.get(t, self.unk_token_id) for t in tokens]
        return ids

    @staticmethod
    def convert_tokens_to_string(tokens):
        return " ".join(tokens)

    def convert_ids_to_embeddings(self, token_ids):
        embeddings = [self.embeddings_arr[ids] for ids in token_ids]
        embeddings = np.vstack(embeddings)
        return embeddings

    def pad_token_ids(self, token_ids, max_len):
        pad_len = max(max_len - len(token_ids), 0)
        token_ids.extend([self.pad_token_id] * pad_len)
        return token_ids

    def decode(self, token_ids):
        tokens = [self.idx2w.get(ti, self.unk_token) for ti in token_ids]
        return tokens

    def encode(self, text_or_tokens, max_length=-1, pad_to_max_length=False, return_embedding=False):
        if not isinstance(text_or_tokens, list):
            tokens = self.tokenize(text_or_tokens)
        else:
            tokens = text_or_tokens
        if len(tokens) == 0:
            tokens = [self.unk_token]
        if max_length > 0:
            tokens = tokens[:max_length]
        if pad_to_max_length:
            tokens = self.pad_tokens(tokens, max_length)
        token_ids = self.convert_tokens_to_ids(tokens)
        if return_embedding:
            embeddings = self.convert_ids_to_embeddings(token_ids)
            return embeddings
        return token_ids


class Parser:

    def __init__(self, device):
        self.logger = logging.getLogger(__name__)
        self.dep_parser = BiaffineParser()
        self.dep_parser.to(device)
        self.logger.info('Loaded Biaffine Dependency Parser')
        self.const_parser = benepar.Parser("benepar_en2", batch_size=16)  # 16
        self.logger.info('Loaded Berkeley Neural Parser Constituency Parser')
        self.cons_tree = ConsTree([])

    def tokenize(self, sentences):
        if not isinstance(sentences, list):
            sentences = [sentences]
        return [self.tokenize_sentence(s) for s in sentences]  # [s.split() for s in sentences]

    # def tokenize_sentence(self, sentence):
    #     bert_tokens = tokenizer.tokenize(sentence)
    #     tokens = [''] * (len(bert_tokens) + 1)
    #     j = 0
    #     for i in range(len(bert_tokens)):
    #         if str(bert_tokens[i]).startswith('##'):
    #             tokens[j] = tokens[j] + str(bert_tokens[i][2:])
    #         else:
    #             j += 1
    #             tokens[j] = str(bert_tokens[i])
    #     tokens = tokens[1:j+1]
    #     return tokens

    def tokenize_sentence(self, sentence, tokens_max_letter=26):
        tokens = nltk.tokenize.word_tokenize(sentence)  # sentence.split(" ")  #
        tokens = [t[:tokens_max_letter] for t in tokens]
        return tokens

    def parse(self, sentences):
        sentences = self.tokenize(sentences)
        # try:
        dep = self.parse_dep(sentences)
        const = self.parse_const(sentences)
        # except:
        #     print(sentences)
        return [' '.join(s) for s in sentences], dep, const

    def tokens_2_conll(self, tokens):
        return '\n'.join([str(i) + '\t' + str(t) + '\t' + '\t'.join(['_'] * 8) for (i, t) in enumerate(tokens, 1)])

    def sentence_to_conll(self, sentences):
        tokenized_sentences = self.tokenize(sentences)
        if not isinstance(tokenized_sentences, list):
            tokenized_sentences = [tokenized_sentences]
        if not isinstance(tokenized_sentences[0], list):
            tokenized_sentences = self.tokenize(tokenized_sentences)
        conll = [self.tokens_2_conll(t) for t in tokenized_sentences]
        conll = self.dep_parser.parse(conll)
        conll = [[x for x in str(c).split('\n') if x != ''] for c in conll]
        return conll

    def parse_dep(self, tokenized_sentences):
        if not isinstance(tokenized_sentences, list):
            tokenized_sentences = [tokenized_sentences]
        if not isinstance(tokenized_sentences[0], list):
            tokenized_sentences = self.tokenize(tokenized_sentences)
        conll = [self.tokens_2_conll(t) for t in tokenized_sentences]
        conll = self.dep_parser.parse(conll)
        conll = [[x for x in str(c).split('\n') if x != ''] for c in conll]
        dep = [DepGraph(c) for c in conll]
        return dep

    def parse_const(self, tokenized_sentences):
        if not isinstance(tokenized_sentences, list):
            tokenized_sentences = [tokenized_sentences]
        if not isinstance(tokenized_sentences[0], list):
            tokenized_sentences = self.tokenize(tokenized_sentences)
        tree = list(self.const_parser.parse_sents(tokenized_sentences))
        const = [self.post_process_const(t) for t in tree]
        return const


    def post_process_const(self, tree):
        cons_tree = ConsTree([])
        tree = cons_tree.read_tree(str(tree))
        tree.close_unaries()
        tree.left_markovize()
        const = cons_tree.linearize_parse_tree(str(tree))
        const = ConstGraph().from_string(const)
        return const


# class Parser:
#
#     def __init__(self, stanford_parser_path):
#         self.logger = logging.getLogger(__name__)
#         self.dep_parser = Spacy2ConllParser(disable_sbd=True, model='en_core_web_lg')
#         self.logger.info('Loaded Spacy Depencency Parser')
#         self.const_parser = StanfordCoreNLP(stanford_parser_path, memory='20g')
#         self.logger.info('Loaded Stanford Core NLP Constituency Parser')
#         self.cons_tree = ConsTree([])
#         self.const_props = {'annotators': 'tokenize,parse', 'pipelineLanguage': 'en', 'tokenize.whitespace': 'true',
#                             'outputFormat': 'json'}  # , 'threads': '20', pos
#
#     def close(self):
#         self.const_parser.close()
#         return
#
#     def parse_dep(self, sent):
#         conll = next(self.dep_parser.parse(input_str=sent, is_tokenized=True))
#         conll = [x for x in conll.split('\n') if x != '']
#         dep = DepGraph(conll)
#         return dep
#
#     def parse_const(self, sent):
#         sent = sent.replace('(', ' ').replace(')', ' ')
#         r_dict = ast.literal_eval(self.const_parser.annotate(sent, properties=self.const_props))
#         tree = [s['parse'] for s in r_dict['sentences']][0]
#         # tree = self.const_parser.parse(sent)
#         tree = self.cons_tree.read_tree(str(tree))
#         tree.close_unaries()
#         tree.left_markovize()
#         const = self.cons_tree.linearize_parse_tree(str(tree))
#         const = ConstGraph().from_string(const)
#         return const