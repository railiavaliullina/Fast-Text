import torch
from torch import nn

from configs.model_config import cfg as model_cfg


class FastText(nn.Module):
    def __init__(self, cfg):
        """
        :param cfg: model config
        """
        super(FastText, self).__init__()
        self.cfg = cfg

        self.A = nn.EmbeddingBag(self.cfg.A_rows_num, self.cfg.embedding_dim)
        nn.init.uniform_(self.A.weight, -1 / self.cfg.embedding_dim, 1 / self.cfg.embedding_dim)
        self.B = nn.Linear(self.cfg.embedding_dim, self.cfg.out_features_size, bias=False)
        nn.init.zeros_(self.B.weight)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, lens_):
        """
        Forward pass.
        :param x: input vector
        :return: model output
        """
        # input = torch.tensor([2, 2, 2, 2, 4, 3, 2, 9], dtype=torch.long)
        # offsets = torch.tensor([0], dtype=torch.long)
        emb = self.A(input, lens_)
        out = self.B(emb)
        return self.log_softmax(out)


def get_model():
    """
    Gets MLP model.
    :return: MLP model
    """
    model = FastText(model_cfg)
    return model  # .cuda()
