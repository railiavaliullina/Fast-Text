from easydict import EasyDict

from configs.dataset_config import cfg as dataset_cfg
from configs.train_config import cfg as train_cfg

cfg = EasyDict()

cfg.A_rows_num = dataset_cfg.fin_vocab_size + dataset_cfg.fin_bucket_size_with_bigrams \
                 if train_cfg.use_bigrams else dataset_cfg.fin_vocab_size + dataset_cfg.fin_bucket_size

cfg.embedding_dim = 200
cfg.out_features_size = 4
