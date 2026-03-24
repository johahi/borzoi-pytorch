from transformers import PretrainedConfig

class BorzoiConfig(PretrainedConfig):
    model_type = "borzoi"

    def __init__(
        self,
        dim = 1536,
        depth = 8,
        heads = 8,
        output_heads = dict(human = 7611, mouse = 2608),
        return_center_bins_only = True,
        attn_dim_key = 64,
        attn_dim_value = 192,
        dropout_rate = 0.2,
        attn_dropout = 0.05,
        pos_dropout = 0.01,
        bins_to_return = 6144,
        **kwargs,
    ):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.output_heads = output_heads
        self.attn_dim_key = attn_dim_key
        self.attn_dim_value = attn_dim_value
        self.dropout_rate = dropout_rate
        self.attn_dropout = attn_dropout
        self.pos_dropout = pos_dropout
        self.return_center_bins_only = return_center_bins_only
        self.bins_to_return = bins_to_return
        super().__init__(**kwargs)