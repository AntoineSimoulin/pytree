

class ChildSumConfig:

    def __init__(
        self,
        embedding_size=300,
        hidden_size=100,
        vocab_size=10000,
        hidden_dropout_prob=0.,
        cell_type='lstm',
        use_attention=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.cell_type = cell_type
        self.use_attention = use_attention
