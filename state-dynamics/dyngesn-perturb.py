"""
dyngesn-perturb.py
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

import numpy as np
import torch
from scipy.io import savemat
from torch.nn.functional import one_hot
from torch_geometric.utils import erdos_renyi_graph

from graphesn import DynamicGraphReservoir, initializer
from graphesn.util import compute_dynamic_graph_alpha

parser = argparse.ArgumentParser()
parser.add_argument('--layers', help='reservoir layers', type=int, default=8)
parser.add_argument('--units', help='reservoir units per layer', type=int, default=32)
parser.add_argument('--times', help='sequence length', type=int, default=1000)
parser.add_argument('--nodes', help='number of nodes', type=int, default=50)
parser.add_argument('--prob', help='edge connection probability', type=int, default=.2)
parser.add_argument('--symbols', help='number input symbols', type=int, default=8)
parser.add_argument('--sigma', help='sigma for recurrent matrix initialization', type=float, default=.6)
parser.add_argument('--leakage', help='leakage constant in recurrent layers', type=float, default=.9)
args = parser.parse_args()

edge_index = [erdos_renyi_graph(args.nodes, args.prob) for _ in range(args.times)]
input_symbols = torch.randint(args.symbols, (args.times, args.nodes))
input = one_hot(input_symbols).float()

alpha = compute_dynamic_graph_alpha(edge_index)

reservoir = DynamicGraphReservoir(num_layers=args.layers, in_features=args.symbols, hidden_features=args.units,
                                  fully=True, return_sequences=True)
reservoir.initialize_parameters(recurrent=initializer('uniform', sigma=args.sigma / alpha),
                                input=initializer('uniform', scale=1),
                                leakage=args.leakage)

X = reservoir(edge_index, input)

x0 = torch.rand(args.nodes, args.units) * .1
x0 = [x0] + [torch.zeros_like(x0)] * (args.layers - 1)
Xp = reservoir(edge_index, input, x0)

input_symbols[args.times // 10] = (input_symbols[args.times // 10] + 1) % args.symbols
input = one_hot(input_symbols).float()
Xi = reservoir(edge_index, input)

savemat(f'perturb_{args.sigma:.2f}_{args.leakage:.2f}.mat',
        {'X': X.numpy(), 'Xp': Xp.numpy(), 'Xi': Xi.numpy(), 'alpha': np.array(alpha),
         'layers': np.array(args.layers), 'units': np.array(args.units), 'T': np.array(args.times),
         'sigma': np.array(args.sigma), 'leakage': np.array(args.leakage)})
