import torch
import numpy as np

from datasets.AGNewsDataset import AGNewsDataset
from configs.dataset_config import cfg as dataset_cfg
from configs.train_config import cfg as train_cfg


def get_dataloaders():
    """
    Initializes train, test datasets and gets their dataloaders.
    :return: train and test dataloaders
    """
    train_dataset = AGNewsDataset(dataset_cfg=dataset_cfg, train_cfg=train_cfg, dataset_type='train')
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=train_cfg.batch_size, drop_last=True,
                                           shuffle=True, collate_fn=collate_fn)

    test_dataset = AGNewsDataset(dataset_cfg=dataset_cfg, train_cfg=train_cfg, dataset_type='test')
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=train_cfg.batch_size, collate_fn=collate_fn)
    return train_dl, test_dl


def collate_fn(data):
    bs = len(data)
    indices, labels = zip(*data)
    lens_ = [len(ind) for ind in indices]
    offsets = [0] + list(np.cumsum(lens_))[:-1]

    # max_len = np.max([len(i) for i in indices])
    # indices_ = np.zeros((bs, max_len))
    #
    # for i in range(bs):
    #     len_ = len(indices[i])
    #     indices_[i] = np.concatenate([indices[i], np.zeros((max_len - len_))], -1)

    indices_ = np.concatenate(indices)
    return torch.tensor(indices_, dtype=torch.int), torch.tensor(labels, dtype=torch.long), \
           torch.tensor(offsets, dtype=torch.int)
