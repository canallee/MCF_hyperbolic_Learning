#!/usr/bin/env python3
import time
import torch
import numpy as np
import logging
import argparse
from hype.sn import Embedding, initialize
from hype.adjacency_matrix_dataset import AdjacencyDataset
from hype import train
from hype.graph import load_adjacency_matrix, load_edge_list, eval_reconstruction
from hype.rsgd import RiemannianSGD
from hype.Euclidean import EuclideanManifold
from hype.Poincare import PoincareManifold
from hype.Lorentz import LorentzManifold
# from hype.Halfspace import HalfspaceManifold
from hype.NLorentz import NLorentzManifold
from hype.LTiling_rsgd import LTilingRSGDManifold
from hype.NLTiling_rsgd import NLTilingRSGDManifold
from hype.LTiling_sgd import LTilingSGDManifold
from hype.HTiling_rsgd import HTilingRSGDManifold
import sys
import json
import torch.multiprocessing as mp


torch.manual_seed(42)
np.random.seed(42)


MANIFOLDS = {
    'Euclidean': EuclideanManifold,
    'Poincare': PoincareManifold,
    'Lorentz': LorentzManifold,
    # 'Halfspace': HalfspaceManifold,
    'NLorentz': NLorentzManifold,
    'LTiling_rsgd': LTilingRSGDManifold,
    'NLTiling_rsgd': NLTilingRSGDManifold,
    'LTiling_sgd': LTilingSGDManifold,
    'HTiling_rsgd': HTilingRSGDManifold
}


# Adapated from:
# https://thisdataguy.com/2017/07/03/no-options-with-argparse-and-python/
class Unsettable(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(Unsettable, self).__init__(
            option_strings, dest, nargs='?', **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        val = None if option_string.startswith('-no') else values
        setattr(namespace, self.dest, val)


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Train Hyperbolic Embeddings')
    parser.add_argument('-dset', type=str, required=True,
                        help='Dataset identifier')
    parser.add_argument('-dim', type=int, default=20,
                        help='Embedding dimension')
    parser.add_argument('-com_n', type=int, default=2,
                        help='Embedding components number')
    parser.add_argument('-manifold', type=str, default='lorentz',
                        choices=MANIFOLDS.keys(), help='Embedding manifold')
    parser.add_argument('-lr', type=float, default=1000,
                        help='Learning rate')
    parser.add_argument('-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('-batchsize', type=int, default=12800,
                        help='Batchsize')
    parser.add_argument('-negs', type=int, default=50,
                        help='Number of negatives')
    parser.add_argument('-burnin', type=int, default=20,
                        help='Epochs of burn in')
    parser.add_argument('-dampening', type=float, default=0.75,
                        help='Sample dampening during burnin')
    parser.add_argument('-ndproc', type=int, default=8,
                        help='Number of data loading processes')
    parser.add_argument('-eval_each', type=int, default=1,
                        help='Run evaluation every n-th epoch')
    parser.add_argument('-debug', action='store_true', default=False,
                        help='Print debuggin output')
    parser.add_argument('-gpu', default=-1, type=int,
                        help='Which GPU to run on (-1 for no gpu)')
    parser.add_argument('-sym', action='store_true', default=False,
                        help='Symmetrize dataset')
    parser.add_argument('-maxnorm', '-no-maxnorm', default='500000',
                        action=Unsettable, type=int)
    parser.add_argument('-sparse', default=False, action='store_true',
                        help='Use sparse gradients for embedding table')
    parser.add_argument('-burnin_multiplier', default=0.01, type=float)
    parser.add_argument('-neg_multiplier', default=1.0, type=float)
    parser.add_argument('-quiet', action='store_true', default=True)
    parser.add_argument(
        '-lr_type', choices=['scale', 'constant'], default='constant')
    parser.add_argument('-train_threads', type=int, default=1,
                        help='Number of threads to use in training')
    parser.add_argument('-eval_embedding', default=False,
                        help='path for the embedding to be evaluated')
    opt = parser.parse_args()

    opt.nor = 'none'

    # setup debugging and logigng
    log_level = logging.DEBUG if opt.debug else logging.INFO
    log = logging.getLogger('Poincare')
    logging.basicConfig(
        level=log_level, format='%(message)s', stream=sys.stdout)

    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')

    manifold = PoincareManifold(
        debug=opt.debug, max_norm=opt.maxnorm, com_n=opt.com_n)

    opt.dim = manifold.dim(opt.dim)

    log.info('Using edge list dataloader')
    idx, objects, weights = load_edge_list(opt.dset, opt.sym)
    model, data, model_name, conf = initialize(
        manifold, opt, idx, objects, weights, sparse=opt.sparse)

    # set burnin parameters
    data.neg_multiplier = opt.neg_multiplier
    train._lr_multiplier = opt.burnin_multiplier
    # Build config string for log
    log.info(f'json_conf: {json.dumps(vars(opt))}')
    # if scale learning rate is used, scale lr with batchsize

    # setup optimizer
    optimizer = RiemannianSGD(model.optim_params(manifold), lr=opt.lr)
    opt.epoch_start = 0

    adj = {}
    for inputs, _ in data:
        for row in inputs:
            x = row[0].item()
            y = row[1].item()
            if x in adj:
                adj[x].add(y)
            else:
                adj[x] = {y}

    print("opt parameter value:", opt.quiet)
    # return

    # print("########## device is here:  ########## \n", device)
    # print("########## model is here :  ########## \n", model)
    # print("########## data is here:  ########## \n", data)
    # print("########## optimizer is here:  ########## \n", optimizer)
    # print("########## log is here:  ########## \n", log)
    train.train(device, model, data, optimizer,
                opt, log, progress=False)

    print("Training time is:", time.time() - start_time)

    meanrank, maprank = eval_reconstruction(
        adj, model.lt.weight.data.clone(), manifold.distance, workers=opt.ndproc)
    sqnorms = manifold.pnorm(model.lt.weight.data.clone())

    log.info(
        'json_stats final test: {'
        f'"sqnorm_min": {sqnorms.min().item()}, '
        f'"sqnorm_avg": {sqnorms.mean().item()}, '
        f'"sqnorm_max": {sqnorms.max().item()}, '
        f'"mean_rank": {meanrank}, '
        f'"map": {maprank}, '
        '}'
    )
    print(model.lt.weight.data[0])


if __name__ == '__main__':
    main()
