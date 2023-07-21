import os

import dgl
from dgl.data import DGLDataset
from dgl.data.utils import download, extract_archive, makedirs, save_info, load_info
import dgl.function as fn
from dgl import save_graphs, load_graphs
from dgl.data.utils import split_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import Bio.PDB
from Bio.PDB.SASA import ShrakeRupley

from tqdm.notebook import tqdm

from CATH_S40_Dataset import CATH_S40_Dataset

class LGN(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size):
        super(LGN, self).__init__()

        self.egc_stack = dgl.nn.pytorch.Sequential()
        self.egc_stack.append(EGNNConv(in_size, hidden_size, out_size, edge_feat_size))
        for _ in range(5):
            self.egc_stack.append(EGNNConv(in_size, hidden_size, out_size))

        self.fully_connected = nn.Linear(in_size, 2)

    def forward(self, g, h, x, e):
        h, _ = self.egc_stack(g, h.float(), x, e)
        h = self.fully_connected(h)
        return h

class EGNNConv(nn.Module):
    r"""Equivariant Graph Convolutional Layer from `E(n) Equivariant Graph
    Neural Networks <https://arxiv.org/abs/2102.09844>`__

    .. math::

        m_{ij}=\phi_e(h_i^l, h_j^l, ||x_i^l-x_j^l||^2, a_{ij})

        x_i^{l+1} = x_i^l + C\sum_{j\in\mathcal{N}(i)}(x_i^l-x_j^l)\phi_x(m_{ij})

        m_i = \sum_{j\in\mathcal{N}(i)} m_{ij}

        h_i^{l+1} = \phi_h(h_i^l, m_i)

    where :math:`h_i`, :math:`x_i`, :math:`a_{ij}` are node features, coordinate
    features, and edge features respectively. :math:`\phi_e`, :math:`\phi_h`, and
    :math:`\phi_x` are two-layer MLPs. :math:`C` is a constant for normalization,
    computed as :math:`1/|\mathcal{N}(i)|`.

    Parameters
    ----------
    in_size : int
        Input feature size; i.e. the size of :math:`h_i^l`.
    hidden_size : int
        Hidden feature size; i.e. the size of hidden layer in the two-layer MLPs in
        :math:`\phi_e, \phi_x, \phi_h`.
    out_size : int
        Output feature size; i.e. the size of :math:`h_i^{l+1}`.
    edge_feat_size : int, optional
        Edge feature size; i.e. the size of :math:`a_{ij}`. Default: 0.

    Example
    -------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import EGNNConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> node_feat, coord_feat, edge_feat = th.ones(6, 10), th.ones(6, 3), th.ones(6, 2)
    >>> conv = EGNNConv(10, 10, 10, 2)
    >>> h, x = conv(g, node_feat, coord_feat, edge_feat)
    """

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0):
        super(EGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the radial feature: ||x_i - x_j||^2
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size),
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
        )

    def message(self, edges):
        """message function for EGNN"""
        # concat features for edge mlp
        if self.edge_feat_size > 0:
            f = torch.cat(
                [
                    edges.src["h"],
                    edges.dst["h"],
                    edges.data["radial"],
                    edges.data["a"],
                ],
                dim=-1,
            )
        else:
            f = torch.cat(
                [edges.src["h"], edges.dst["h"], edges.data["radial"]], dim=-1
            )

        msg_h = self.edge_mlp(f)
        msg_x = self.coord_mlp(msg_h) * edges.data["x_diff"]

        return {"msg_x": msg_x, "msg_h": msg_h}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        r"""
        Description
        -----------
        Compute EGNN layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        node_feat : torch.Tensor
            The input feature of shape :math:`(N, h_n)`. :math:`N` is the number of
            nodes, and :math:`h_n` must be the same as in_size.
        coord_feat : torch.Tensor
            The coordinate feature of shape :math:`(N, h_x)`. :math:`N` is the
            number of nodes, and :math:`h_x` can be any positive integer.
        edge_feat : torch.Tensor, optional
            The edge feature of shape :math:`(M, h_e)`. :math:`M` is the number of
            edges, and :math:`h_e` must be the same as edge_feat_size.

        Returns
        -------
        node_feat_out : torch.Tensor
            The output node feature of shape :math:`(N, h_n')` where :math:`h_n'`
            is the same as out_size.
        coord_feat_out: torch.Tensor
            The output coordinate feature of shape :math:`(N, h_x)` where :math:`h_x`
            is the same as the input coordinate feature dimension.
        """
        with graph.local_scope():
            # node feature
            graph.ndata["h"] = node_feat
            # coordinate feature
            graph.ndata["x"] = coord_feat
            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat
            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
            graph.edata["radial"] = (
                graph.edata["x_diff"].square().sum(dim=1).unsqueeze(-1)
            )
            # normalize coordinate difference
            graph.edata["x_diff"] = F.normalize(graph.edata["x_diff"], p=2, dim=1)
            # graph.edata["x_diff"] = graph.edata["x_diff"] / (
            #     graph.edata["radial"].sqrt() + 1e-30
            # )
            
            graph.apply_edges(self.message)
            graph.update_all(fn.copy_e("msg_x", "m"), fn.mean("m", "x_neigh"))
            graph.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))

            h_neigh, x_neigh = graph.ndata["h_neigh"], graph.ndata["x_neigh"]

            h = self.node_mlp(torch.cat([node_feat, h_neigh], dim=-1))
            x = coord_feat + x_neigh

            return h, x

if __name__ == '__main__':
    
    print("Loading dataset")
    dataset = CATH_S40_Dataset()
    print("Dataset loaded!")
    
    train_set, val_set, test_set = split_dataset(dataset, (0.8, 0.1, 0.1))
    batched_val = dgl.batch(val_set)

    dataloader = dgl.dataloading.GraphDataLoader(
        train_set,
        batch_size=1024,
        drop_last=False,
        shuffle=True,
        num_workers=8
    )

    g = dataset.graphs[0]
    model = LGN(g.ndata['h'].shape[1], 10, g.ndata['h'].shape[1], g.edata['e'].shape[1])
    
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    model.float()
    
    print("Here we go...")
    for epoch in range(300):
        total_loss = 0
        for i, batched_g in enumerate(dataloader):
            print("Batch", i, "of", len(dataloader), "batches", end='\r')
            labels  = batched_g.ndata['h'][:, 20:22]
            # Zero out the bfactor and sasa features
            h = batched_g.ndata['h'].clone()
            # TODO: add noise to h[:, :20] (one hot AA encoding)
            h[:, 20] = 0
            h[:, 21] = 0
            x = batched_g.ndata['X']
            e = batched_g.edata['e']

            pred = model(batched_g, h, x, e)
            loss = F.mse_loss(pred.to(torch.float64), labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
            opt.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            h = batched_val.ndata['h'].clone()
            # Zero out the bfactor and sasa features
            h[:, 20] = 0
            h[:, 21] = 0
            x = batched_val.ndata['X']
            e = batched_val.edata['e']
            labels   = batched_val.ndata['h'][:, 20:22]
            pred     = model(batched_val, h, x, e)
            val_loss = F.mse_loss(pred.to(torch.float64), labels)
            print(f'Epoch {epoch:3d}\ttrain loss: {total_loss:.2f}\tval loss: {val_loss:.2f}')

# Training on bfactor and sasa only
# Epoch 0 | train loss: 90416.77 val loss: 3269.58
# Epoch 5 | train loss: 23730.76 val loss: 886.62