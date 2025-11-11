import torch
import torch.nn.functional as F
import dgl.function as fn
import sympy
import scipy
from torch import nn
from torch.nn import init
from torch_geometric.nn import MessagePassing

def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas

class WeightedEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_edges):
        super(WeightedEdgeConv, self).__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels)
        self.edge_weight = nn.Parameter(torch.randn(num_edges))

    def forward(self, x, edge_index, edge_weight):
        x = self.linear(x)
        return self.propagate(edge_index=edge_index, x=x, edge_weight=self.edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

class PolyConv_hetero(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv_hetero, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h

class WeightedGraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, edge_weight, alpha):
        super(WeightedGraphConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.edge_weight = edge_weight
        self.alpha = alpha
    def forward(self, graph):
        with graph.local_scope():
            graph.edges['homo'].data['w'] = (1 - self.edge_weight).float()
            graph.update_all(
                fn.u_mul_e('feature', 'w', 'm'),
                fn.sum('m', 'h_new'),
                etype='homo'
            )
            feat = graph.ndata['h_new']
            out = self.linear(feat)
            return self.alpha * out

class Spectral_Beta(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, w_tanhscore, alpha, d=2):
        super(Spectral_Beta, self).__init__()
        self.g = graph
        self.g.ndata['feature'] = self.g.ndata['feature'].float()
        self.thetas = calculate_theta2(d=d)
        self.alpha = alpha
        self.h_feats = h_feats
        self.conv = [PolyConv_hetero(h_feats, h_feats, theta, lin=False) for theta in self.thetas]
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats*len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.LeakyReLU()
        for param in self.parameters():
            print(type(param), param.size())
        edge_ids = graph.edata['_ID'][('r', 'homo', 'r')].long().to('cpu')
        self.num_edges = graph.num_edges(etype='homo')
        subgraph_w_tanhscore = w_tanhscore[edge_ids].unsqueeze(1)
        self.weconv = WeightedGraphConv(in_feats, h_feats, subgraph_w_tanhscore, alpha=self.alpha)
        del subgraph_w_tanhscore

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_all = []
        h_final = torch.zeros([len(in_feat), 0])
        for conv in self.conv:
            h0 = conv(self.g, h)
            h_final = torch.cat([h_final, h0], -1)
        h = self.linear3(h_final)
        h_all.append(h)
        h_we = self.weconv(self.g)
        h_all.append(h_we)
        h_all = torch.stack(h_all).sum(0)
        h_all = self.act(h_all)
        h_all = self.linear4(h_all)
        return h_all
