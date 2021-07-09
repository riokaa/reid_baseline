# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
import test_tube  # noqa: F401

from data.datasets.eval_reid import evaluate
from modeling import build_model, ReidLoss
from solver.build import make_optimizer, make_lr_scheduler


class ReidSystem(pl.LightningModule):
    def __init__(self, cfg, train_loader, val_loader, num_classes, num_query):
        super().__init__()
        # Define networks
        (
            self.cfg,
            self.train_loader,
            self.val_loader,
            self.num_classes,
            self.num_query,
        ) = (cfg, train_loader, val_loader, num_classes, num_query)
        self.model = build_model(cfg, num_classes)
        self.loss_fns = ReidLoss(cfg.SOLVER.LOSSTYPE, cfg.SOLVER.MARGIN, num_classes)

    def training_step(self, batch, batch_nb):
        inputs, labels = batch
        outs = self.model(inputs, labels)
        loss = self.loss_fns(outs, labels)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        inputs, pids, camids = batch
        feats = self.model(inputs)
        return {
            "feats": feats,
            "pids": pids.cpu().numpy(),
            "camids": camids.cpu().numpy(),
        }

    def validation_end(self, outputs):
        feats, pids, camids = [], [], []
        for o in outputs:
            feats.append(o["feats"])
            pids.extend(o["pids"])
            camids.extend(o["camids"])
        feats = torch.cat(feats, dim=0)
        if self.cfg.TEST.NORM:
            feats = F.normalize(feats, p=2, dim=1)
        # query
        qf = feats[: self.num_query]
        q_pids = np.asarray(pids[: self.num_query])
        q_camids = np.asarray(camids[: self.num_query])
        # gallery
        gf = feats[self.num_query :]
        g_pids = np.asarray(pids[self.num_query :])
        g_camids = np.asarray(camids[self.num_query :])

        m, n = qf.shape[0], gf.shape[0]
        distmat = (
            torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        )
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
        self.log(f"Test Results - Epoch: {self.current_epoch + 1}")
        self.log(f"mAP: {mAP:.1%}")
        for r in [1, 5, 10]:
            self.log(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")
        tqdm_dic = {"rank1": cmc[0], "mAP": mAP}
        return tqdm_dic

    def configure_optimizers(self):
        opt_fns = make_optimizer(self.cfg, self.model)
        lr_sched = make_lr_scheduler(self.cfg, opt_fns)
        return [opt_fns], [lr_sched]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def do_train(
    cfg,
    local_rank,
    train_loader,
    val_loader,
    num_classes,
    num_query,
):
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    epochs = cfg.SOLVER.MAX_EPOCHS
    gpus = cfg.MODEL.GPUS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start Training")

    dirpath = os.path.join(output_dir, "ckpts")
    filename = f"ckpts-v{cfg.MODEL.VERSION}" + "-epoch{epoch:02d}-rank1{rank1:02d}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        monitor="rank1",
        save_top_k=3,
        verbose=True,
        mode="max",
    )

    model = ReidSystem(cfg, train_loader, val_loader, num_classes, num_query)
    tt_logger = TestTubeLogger(
        save_dir=output_dir, name=cfg.DATASETS.TEST_NAMES, version=cfg.MODEL.VERSION
    )

    trainer = pl.Trainer(
        logger=tt_logger,
        max_epochs=epochs,
        checkpoint_callback=checkpoint_callback,
        check_val_every_n_epoch=eval_period,
        gpus=gpus,
        auto_select_gpus=True,
        num_sanity_val_steps=0,
        weights_summary="top",
        log_every_n_steps=len(train_loader) // 2,
    )

    trainer.fit(model)

    # continue training
    # if cfg.MODEL.CHECKPOINT is not '':
    #     state = torch.load(cfg.MODEL.CHECKPOINT)
    #     if set(state.keys()) == {'model', 'opt'}:
    #         model_state = state['model']
    #         learn.model.load_state_dict(model_state)
    #         learn.create_opt(0, 0)
    #         learn.opt.load_state_dict(state['opt'])
    #     else:
    #         learn.model.load_state_dict(state['model'])
    #     logger.info(f'continue training from checkpoint {cfg.MODEL.CHECKPOINT}')
