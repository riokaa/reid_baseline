# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    (
        imgs,
        pids,
        _,
        _,
    ) = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids


def train_collate_fn_with_index(batch):
    imgs, pids, _, _, index = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    index = torch.tensor(index, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, index
