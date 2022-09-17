from torch import nn

from configs.model_config import cfg as model_cfg


class FastText(nn.Module):
    def __init__(self, cfg):
        """
        :param cfg: model config
        """
        super(FastText, self).__init__()
        self.cfg = cfg

        self.A = nn.EmbeddingBag(self.cfg.A_rows_num, self.cfg.embedding_dim)  # , sparse=True
        nn.init.uniform_(self.A.weight, -1 / self.cfg.embedding_dim, 1 / self.cfg.embedding_dim)
        self.B = nn.Linear(self.cfg.embedding_dim, self.cfg.out_features_size, bias=False)
        nn.init.zeros_(self.B.weight)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, offsets):
        """
        Forward pass.
        :param input: input vector
        :param offsets: offsets for EmbeddingBag
        :return: model output
        """
        emb = self.A(input, offsets=offsets)
        out = self.B(emb)
        return self.log_softmax(out)


def get_model():
    """
    Gets MLP model.
    :return: MLP model
    """
    model = FastText(model_cfg)
    return model  # .cuda()
