# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import numpy as np
import torch.nn.functional as F
from data.datasets.eval_reid import evaluate
from fastai.torch_core import to_np


def inference(cfg, model, data_bunch, tst_loader, num_query):
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing")

    model.eval()
    feats = []
    pids = []
    camids = []
    for imgs, pid, camid in data_bunch.test_dl:
        with torch.no_grad():
            feat = model(imgs.cuda())
        feats.append(feat)
        pids.append(pid)
        camids.append(camid)

    feats = torch.cat(feats, dim=0)
    qf = feats[:num_query]
    gf = feats[num_query:]
    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])

    m, n = qf.shape[0], gf.shape[0]
    # Cosine distance
    distmat = torch.mm(F.normalize(qf), F.normalize(gf).t())

    # Euclid distance
    # distmat = torch.pow(qf,2).sum(dim=1,keepdim=True).expand(m,n) + \
    # torch.pow(gf,2).sum(dim=1,keepdim=True).expand(n,m).t()
    # distmat.addmm_(1, -2, qf, gf.t())

    distmat = to_np(distmat)

    # Compute CMC and mAP.
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Compute CMC Curve")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
