import torch
import torch.nn.functional as F

from torch import nn as nn
from spectral_beta import Spectral_Beta

def fraud_model(args, in_feats, n_classes, w_tanhscore,sg_graph):
    sg_graph = sg_graph.to('cuda')
    model = Spectral_Beta(in_feats, args.n_hidden, n_classes, sg_graph, w_tanhscore)
    return model

class Hetero_Evaluator(nn.Module):
    def __init__(self, g, hidden_dim=16, etype=('r','homo','r')):
        super().__init__()
        self.etype = etype
        self.input_dim = g.nodes['r'].data['feature'].shape[1]
        self.Wx = nn.Linear(self.input_dim, hidden_dim, bias=True)
        self.sigma = nn.ReLU()
        self.Wsd = nn.Linear(1, hidden_dim, bias=True)
        self.Wh = nn.Linear(4*hidden_dim, 1, bias=True)
        nn.init.xavier_uniform_(self.Wx.weight); nn.init.zeros_(self.Wx.bias)
        nn.init.xavier_uniform_(self.Wsd.weight); nn.init.zeros_(self.Wsd.bias)
        nn.init.xavier_uniform_(self.Wh.weight); nn.init.zeros_(self.Wh.bias)
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.score_shift = nn.Parameter(torch.tensor(0.0))
    def _score_edges(self, edges):
        x_i = edges.src['feature']
        x_j = edges.dst['feature']
        h_i = self.sigma(self.Wx(x_i))
        h_j = self.sigma(self.Wx(x_j))
        FD_ij = torch.cat([h_i, h_j, (h_i - h_j)], dim=-1)

        deg_i = edges.src['deg_log']
        deg_j = edges.dst['deg_log']
        deg_diff = torch.abs(deg_i - deg_j)
        SD_ij = self.sigma(self.Wsd(deg_diff))
        fused = torch.cat([FD_ij, SD_ij], dim=-1)
        phi_raw = torch.tanh(self.Wh(fused)).squeeze(-1)
        phi = 0.5 * (phi_raw + 1.0)
        return {
            'phi_raw': phi_raw,
            'phi': phi,
            'score': phi_raw
        }

    def forward(self, g):
        if 'deg_log' not in g.ndata:
            deg = g.in_degrees(etype=self.etype).float().unsqueeze(-1)
            g.ndata['deg_log'] = torch.log1p(deg)

        g.apply_edges(self._score_edges, etype=self.etype)

        return g.edges[self.etype].data['phi_raw']

    def loss(self, g):
        with g.local_scope():
            self.forward(g)
            msk = g.edges[self.etype].data['train_mask'].bool()
            y_raw = g.edges[self.etype].data['label'][msk]
            s_raw = g.edges[self.etype].data['phi_raw'][msk].float()
            y_pm1 = torch.where(y_raw > 0, torch.ones_like(y_raw), -torch.ones_like(y_raw)).float()
            s_eff = (s_raw - self.score_shift) * self.score_scale
            pos_idx = (y_pm1 ==  1).nonzero(as_tuple=True)[0]
            neg_idx = (y_pm1 == -1).nonzero(as_tuple=True)[0]
            n_pos, n_neg = pos_idx.numel(), neg_idx.numel()

            if n_pos == 0 or n_neg == 0:
                margin = 1.0
                loss = F.relu(margin - y_pm1 * s_eff).mean()
                return loss
            k = min(n_pos, n_neg)
            dev = s_eff.device
            pos_take = torch.randperm(n_pos, device=dev)[:k]
            neg_take = torch.randperm(n_neg, device=dev)[:k]
            sel = torch.cat([pos_idx[pos_take], neg_idx[neg_take]], dim=0)
            sel = sel[torch.randperm(sel.numel(), device=dev)]
            margin = 1.0
            loss = F.relu(margin - y_pm1[sel] * s_eff[sel]).mean()
            return loss

class EarlyStop:
    def __init__(self, patience=20, if_more=True, min_delta=1e-4):
        self.if_more = if_more
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best = -float('inf') if if_more else float('inf')
        self.num_bad = 0

    def step(self, current, epoch=None):
        if self.if_more:
            improved = current > self.best + self.min_delta
        else:
            improved = current < self.best - self.min_delta

        do_stop = False

        if improved:
            self.best = current
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                do_stop = True

        return  do_stop
