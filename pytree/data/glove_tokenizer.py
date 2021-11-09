import nltk
from tqdm.auto import tqdm
import numpy as np


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
            for i in tqdm(range(0, num_lines - 2), total=(num_lines - 2), desc="load embedding file"):
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