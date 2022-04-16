#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import timeit
from tqdm import tqdm
from torch.utils import data as torch_data
from hype.graph import eval_reconstruction

from hype.Euclidean import EuclideanManifold
from hype.Poincare import PoincareManifold
from hype.Lorentz import LorentzManifold
# from hype.Halfspace import HalfspaceManifold
from hype.NLorentz import NLorentzManifold
from hype.LTiling_rsgd import LTilingRSGDManifold
from hype.NLTiling_rsgd import NLTilingRSGDManifold
from hype.LTiling_sgd import LTilingSGDManifold
from hype.HTiling_rsgd import HTilingRSGDManifold
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt


_lr_multiplier = 0.01


def data_loader_lr(opt, data, epoch, progress=False):
    data.burnin = False
    lr = opt.lr
    if epoch < opt.burnin:
        data.burnin = True
        lr = opt.lr * _lr_multiplier
    loader_iter = tqdm(data) if progress else data
    return loader_iter, lr


def train(
        device,
        model,
        data,
        optimizer,
        opt,
        log,
        progress=False
):
    epoch_loss = torch.Tensor(len(data))
    LOSS = np.zeros(opt.epochs)

    for epoch in range(opt.epoch_start, opt.epochs):
        largest_weight_emb = round(
            torch.abs(model.lt.weight.data).max().item(), ndigits=6)
        print(largest_weight_emb, "is the largest absolute weight in the embedding")
        epoch_loss.fill_(0)
        t_start = timeit.default_timer()
        loader_iter, lr = data_loader_lr(opt, data, epoch, progress=progress)

        for i_batch, (inputs, targets) in enumerate(loader_iter):

            elapsed = timeit.default_timer() - t_start
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = model.loss(preds, targets, size_average=True)
            loss.backward()
            optimizer.step(lr=lr)
            epoch_loss[i_batch] = loss.cpu().item()

        LOSS[epoch] = torch.mean(epoch_loss).item()
        log.info('json_stats: {' f'"epoch": {epoch}, '
                 f'"elapsed": {elapsed}, ' f'"loss": {LOSS[epoch]}, ' '}')
