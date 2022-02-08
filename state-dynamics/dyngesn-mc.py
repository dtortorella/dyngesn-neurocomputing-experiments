"""
dyngesn-mc.py
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

import networkx
import numpy as np
import torch
from scipy.io import savemat
from torch_geometric.utils import erdos_renyi_graph, from_networkx
from tqdm import tqdm

from graphesn import DynamicGraphReservoir, initializer, Readout
from graphesn.util import compute_dynamic_graph_alpha


def k_regular_graph(degree, nodes):
    G = networkx.random_graphs.random_regular_graph(degree, nodes)
    return from_networkx(G).edge_index


parser = argparse.ArgumentParser()
parser.add_argument('--graph', help='random graph type (er, reg)', default='er')
parser.add_argument('--mc', help='memory capacity approximation', type=int, default=100)
parser.add_argument('--times', help='sequence length', type=int, default=1000)
parser.add_argument('--nodes', help='number of nodes', type=int, default=50)
parser.add_argument('--prob', help='edge connection probability for E-R graphs', type=float, default=.2)
parser.add_argument('--deg', help='node degree for k-regular graphs', type=int, default=3)
parser.add_argument('--device', help='device for torch computations', default='cpu')
parser.add_argument('--layers', help='reservoir layers', type=int, nargs='+', default=[4])
parser.add_argument('--units', help='reservoir units per layer', type=int, nargs='+', default=[16])
parser.add_argument('--sigma', help='sigma for recurrent matrix initialization', type=float, nargs='+', default=[0.7])
parser.add_argument('--leakage', help='leakage constant', type=float, nargs='+', default=[0.9])
parser.add_argument('--ld', help='readout lambda', type=float, nargs='+', default=[1e-8])
parser.add_argument('--trials', help='number of trials', type=int, default=100)
args = parser.parse_args()

if args.graph.lower() == 'er':
    edge_index = [erdos_renyi_graph(args.nodes, args.prob) for _ in range(args.times)]
else:
    edge_index = [k_regular_graph(args.deg, args.nodes) for _ in range(args.times)]
input = (torch.rand(args.times, args.nodes, 1) - 0.5) * 0.04

alpha = compute_dynamic_graph_alpha(edge_index)
print(f'T = {args.times}, N = {args.nodes}, alpha = {alpha:.3f}')

device = torch.device(args.device)
edge_index = [adj.to(device) for adj in edge_index]
input = input.to(device)

mc = torch.zeros(len(args.layers), len(args.units), len(args.sigma), len(args.leakage), len(args.ld), args.trials).to(device)

with tqdm(total=mc.numel()) as progress:
    for trial_index in range(args.trials):
        for unit_index, unit in enumerate(args.units):
            reservoir = DynamicGraphReservoir(num_layers=max(args.layers), in_features=1, hidden_features=unit,
                                              pooling=None, fully=True, return_sequences=True)
            for sigma_index, sigma in enumerate(args.sigma):
                for leakage_index, leakage in enumerate(args.leakage):
                    reservoir.initialize_parameters(recurrent=initializer('uniform', sigma=sigma / alpha),
                                                    input=initializer('uniform', scale=1),
                                                    leakage=leakage)
                    reservoir.to(device)
                    X = reservoir(edge_index=edge_index, input=input)
                    for layer_index, layer in enumerate(args.layers):
                        readout = Readout(num_features=layer * unit, num_targets=1)
                        Xl = X[:, :, :layer * unit]
                        for ld_index, ld in enumerate(args.ld):
                            try:
                                for k in range(1, args.mc + 1):
                                    readout.fit(data=(Xl[k:].view(-1, Xl.shape[-1]), input[:-k].view(-1, 1)), regularization=ld)
                                    recon = readout(Xl[k:].view(-1, Xl.shape[-1]))
                                    corr = torch.corrcoef(torch.stack([recon.squeeze(), input[:-k].view(-1)]))
                                    mc[layer_index, unit_index, sigma_index, leakage_index, ld_index, trial_index] += corr[0, 1] ** 2
                            except:
                                pass
                            progress.update(1)

savemat(f'mc-{args.graph}_{alpha:.2f}.mat', mdict={
    'mc': mc.cpu().numpy(), 'T': np.array(args.times), 'N': np.array(args.nodes), 'k': np.array(args.mc),
    'layers': np.array(args.layers), 'units': np.array(args.units), 'sigma': np.array(args.sigma),
    'leakage': np.array(args.leakage), 'ld': np.array(args.ld), 'alpha': np.array(alpha)
})


print()
