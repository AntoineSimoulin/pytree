

class NaryConfig:

    def __init__(
        self,
        embedding_size=300,
        hidden_size=150,
        vocab_size=10000,
        hidden_dropout_prob=0.,
        cell_type='lstm',
        use_attention=False,
        use_bert=False,
        tune_bert=False,
        normalize_bert_embeddings=False,
        xavier_init=True,
        N=2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.cell_type = cell_type
        self.use_attention = use_attention
        self.use_bert = use_bert
        self.tune_bert = tune_bert
        self.normalize_bert_embeddings = normalize_bert_embeddings
        self.xavier_init = xavier_init
        self.N = N
