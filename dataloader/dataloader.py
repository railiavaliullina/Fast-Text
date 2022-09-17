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
    train_dataset = AGNewsDataset(cfg=dataset_cfg, dataset_type='train')
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=train_cfg.batch_size, drop_last=True,
                                           pin_memory=True, collate_fn=collate_fn)
    test_dataset = AGNewsDataset(cfg=dataset_cfg, dataset_type='test')
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=train_cfg.batch_size, pin_memory=True,
                                          collate_fn=collate_fn)
    return train_dl, test_dl


def collate_fn(data):
    indices, labels = zip(*data)
    lens_ = [0] + [len(data[i][0]) for i in range(len(data) - 1)]
    indices_ = np.concatenate(indices)
    return torch.tensor(indices_, dtype=torch.int), torch.tensor(labels, dtype=torch.long), \
           torch.tensor(lens_, dtype=torch.int)
