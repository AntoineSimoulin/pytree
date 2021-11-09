

class SimilarityConfig:

    def __init__(
        self,
        num_classes=5,
        hidden_similarity_size=50,        
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.hidden_similarity_size = hidden_similarity_size

