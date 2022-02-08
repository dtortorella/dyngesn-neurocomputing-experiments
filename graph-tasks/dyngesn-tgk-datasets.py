"""
dyngesn-tgk-datasets.py
Copyright (C) 2022, Domenico Tortorella
Copyright (C) 2022, University of Pisa

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import argparse
import math
import random
import statistics

import torch
import numpy as np
from scipy.io import savemat
from torch.nn.functional import one_hot, pad
from torch_geometric.nn import global_add_pool
from tqdm import tqdm

from graphesn import DynamicGraphReservoir, initializer, Readout
from graphesn.data import DynamicBatch
from graphesn.dataset import TGKDataset
from graphesn.util import compute_dynamic_graph_alpha


def holdout_split(dataset, train_ratio, valid_ratio):
    samples0 = [i for i in range(len(dataset)) if dataset[i].y == 0]
    samples1 = [i for i in range(len(dataset)) if dataset[i].y == 1]
    n0 = len(samples0)
    n1 = len(samples0)
    random.shuffle(samples0)
    random.shuffle(samples1)
    train_split = samples0[:int(n0 * train_ratio)] + samples1[:int(n1 * train_ratio)]
    valid_split = samples0[int(n0 * train_ratio):int(n0 * (train_ratio + valid_ratio))] + samples1[int(n1 * train_ratio):int(n1 * (train_ratio + valid_ratio))]
    test_split = samples0[int(n0 * (train_ratio + valid_ratio)):] + samples1[int(n1 * (train_ratio + valid_ratio)):]
    return train_split, valid_split, test_split


def prepare_batch(dataset, device):
    data_batch = DynamicBatch(dataset).to(device)
    x = [data.x for data in data_batch]
    edge_index = [data.edge_index for data in data_batch]
    mask = [data.mask for data in data_batch]
    batch = data_batch[-1].batch
    y = data_batch[-1].y
    return edge_index, x, batch, mask, y


def accuracy(y_pred, y_true):
    return (y_pred.argmax(dim=-1) == y_true).float().mean()


def trim_and_pad_timesteps(dataset):
    max_T = max(sample.mask.all(dim=1).int().sum().item() for sample in dataset)
    for i in range(len(dataset)):
        mask = dataset[i].mask.all(dim=1)
        T = mask.int().sum().item()
        dataset[i]._storage['edge_index'] = [adj for t, adj in enumerate(dataset[i].edge_index) if mask[t]] + [torch.empty((2, 0), dtype=torch.int64)] * (max_T - T)
        dataset[i]._storage['x'] = pad(dataset[i].x[mask], (0, 0, 0, 0, 0, max_T - T), 'constant', 0.0)
        dataset[i]._storage['mask'] = pad(dataset[i].mask[mask], (0, 0, 0, max_T - T), 'constant', False)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset name')
parser.add_argument('--root', help='root directory for dataset', default='/tmp')
parser.add_argument('--device', help='device for torch computations', default='cpu')
parser.add_argument('--layers', help='reservoir layers', type=int, nargs='+', default=[4])
parser.add_argument('--units', help='reservoir units per layer', type=int, nargs='+', default=[16])
parser.add_argument('--split', help='(training, validation) split ratios', nargs=2, default=[.8, .1])
parser.add_argument('--sigma', help='sigma for recurrent matrix initialization', type=float, nargs='+', default=[0.9])
parser.add_argument('--leakage', help='leakage constant', type=float, nargs='+', default=[0.9])
parser.add_argument('--ld', help='readout lambda', type=float, nargs='+', default=[1e-3])
parser.add_argument('--trials', help='number of trials', type=int, default=100)
args = parser.parse_args()

dataset = TGKDataset(name=args.dataset, root=args.root)
trim_and_pad_timesteps(dataset)
device = torch.device(args.device)
alpha_mean = statistics.mean(compute_dynamic_graph_alpha(dataset[i]) for i in range(len(dataset)))
print(f'alpha = {alpha_mean:.2f}')

edge_index, x, batch, mask, y = prepare_batch(dataset, device)
Y = one_hot(y).float()

train_acc = torch.empty(len(args.layers), len(args.units), len(args.sigma), len(args.leakage), len(args.ld), args.trials).fill_(math.nan)
val_acc = torch.empty_like(train_acc).fill_(math.nan)
test_acc = torch.empty_like(train_acc).fill_(math.nan)

with tqdm(total=train_acc.numel()) as progress:
    for trial_index in range(args.trials):
        train_set, valid_set, test_set = holdout_split(dataset, args.split[0], args.split[1])
        select_set = train_set + valid_set
        for unit_index, unit in enumerate(args.units):
            reservoir = DynamicGraphReservoir(num_layers=max(args.layers), in_features=1, hidden_features=unit,
                                              pooling=global_add_pool, fully=True)
            for sigma_index, sigma in enumerate(args.sigma):
                for leakage_index, leakage in enumerate(args.leakage):
                    reservoir.initialize_parameters(recurrent=initializer('uniform', sigma=sigma / alpha_mean),
                                                    input=initializer('uniform', scale=1),
                                                    leakage=leakage)
                    reservoir.to(device)
                    X = reservoir(edge_index=edge_index, input=x, mask=mask, batch=batch)
                    for layer_index, layer in enumerate(args.layers):
                        readout = Readout(num_features=layer * unit, num_targets=2)
                        Xl = X[:, :layer * unit]
                        for ld_index, ld in enumerate(args.ld):
                            try:
                                readout.fit(data=(Xl[train_set], Y[train_set]), regularization=ld)
                                yp = readout(Xl)
                                val_acc[layer_index, unit_index, sigma_index, leakage_index, ld_index, trial_index] = accuracy(yp[valid_set], y[valid_set])
                                readout.fit(data=(Xl[select_set], Y[select_set]), regularization=ld)
                                yp = readout(Xl)
                                train_acc[layer_index, unit_index, sigma_index, leakage_index, ld_index, trial_index] = accuracy(yp[select_set], y[select_set])
                                test_acc[layer_index, unit_index, sigma_index, leakage_index, ld_index, trial_index] = accuracy(yp[test_set], y[test_set])
                            except:
                                pass
                            progress.update(1)

savemat(f'{args.dataset}.mat', mdict={
    'train_acc': train_acc.cpu().numpy(), 'val_acc': val_acc.cpu().numpy(), 'test_acc': test_acc.cpu().numpy(),
    'layers': np.array(args.layers), 'units': np.array(args.units), 'sigma': np.array(args.sigma),
    'leakage': np.array(args.leakage), 'ld': np.array(args.ld), 'alpha': np.array(alpha_mean)
})

print()
