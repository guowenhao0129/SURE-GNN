import numpy as np
import torch
import dgl
import os
from part_utils import part_graph_fast

def map_idx_fraud(idx, orig_id):
    idx_np = idx.numpy() if isinstance(idx, torch.Tensor) else idx
    inter_idx = np.intersect1d(idx_np, orig_id)
    if len(inter_idx) == 0:
        print("No intersection found between idx and orig_id.")
        return np.array([])

    newid = []
    for i in inter_idx:
        tmp = np.where(orig_id == i)[0]
        assert len(tmp) == 1, f"Expected exactly one index for {i}, but found {len(tmp)}."
        newid.append(tmp[0])
    return np.array(newid)


def gen_mask(n_nodes, idx):
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    mask[idx] = True
    return mask

def map_mask(mask):
    tmp = torch.where(mask == True)
    return tmp[0]

class FullFraudGraph:
    def __init__(self, args, graph):
        self.gid = -1
        self.dataset = args.dataset
        self.graph = graph
        self.labels = graph.ndata['label']
        self.train_mask = graph.ndata['train_mask'].long()
        self.val_mask = graph.ndata['valid_mask'].long()
        self.test_mask = graph.ndata['test_mask'].long()

        self.train_idx = torch.nonzero(self.train_mask, as_tuple=True)[0]
        self.val_idx = torch.nonzero(self.val_mask, as_tuple=True)[0]
        self.test_idx = torch.nonzero(self.test_mask, as_tuple=True)[0]
        self.n_nodes = graph.ndata['label'].shape[0]

class FullAggred:
    def __init__(self, fg):

        self.labels = fg.labels
        self.n_nodes = fg.n_nodes
        self.train_idx = fg.train_idx
        self.val_idx = fg.val_idx
        self.test_idx = fg.test_idx

class SubFraudGraph:
    def __init__(self, g, args, gid, num_parts, part_mode, nodes_id, device, first_time):

        self.gid = gid   # subgraph id
        self.device = device
        if first_time:
            self.g = g
        else:
            self.g = g
        self.graph, self.labels, self.train_idx, self.val_idx, self.test_idx \
                = self.load_subgraph(args, num_parts, part_mode, nodes_id, first_time)

        self.in_feats = self.graph.ndata['feature'].shape[1]
        self.num_nodes = self.graph.ndata['feature'].shape[0]
        if args.dataset == 'yelp' or 'amazon':
            self.n_classes = 2
        else:
            self.n_classes = 40

    def to_device(self, device):
        self.graph = self.graph.to(device)
        self.labels = self.labels.to(device)
        self.train_idx = self.train_idx.to(device)
        self.val_idx = self.val_idx.to(device)
        self.test_idx = self.test_idx.to(device)

    def move_graph(self, device):
        self.labels = self.labels.to(device)

    def get_orig_id(self, idx):
        return self.graph.ndata['orig_id'][idx]

    def load_subgraph(self, args, num_parts, part_mode, nodes_id, first_time):
        fp = os.path.join('./partition', args.dataset, str(num_parts))
        os.makedirs(fp, exist_ok=True)
        fn = os.path.join(fp, 'mode' + str(part_mode) + '-sg' + str(self.gid) + '.bin')
        if first_time:
            fg = FullFraudGraph(args, self.g)
            g0 = dgl.node_subgraph(fg.graph, nodes_id)
            edge_scores = g0.edata['edge_score'][('r', 'homo', 'r')]
            weight_np = edge_scores.cpu().detach().numpy()
            g0_orig_id = g0.ndata['_ID']
            g0_orig_idlong = g0_orig_id.clone().detach().long()
            g0.ndata['orig_id'] = g0_orig_id
            g0.ndata['feature'] = fg.graph.ndata['feature'][g0_orig_idlong]
            g0.ndata['label'] = fg.labels[g0_orig_idlong]

            g0 = g0.to('cpu')
            g0_orig_id = g0.ndata['orig_id'].cpu().numpy()
            g0_labels = fg.labels[g0_orig_id]
            g0.ndata['label'] = g0_labels.to('cpu')
            fg_train_idx = fg.train_idx.cpu()
            fg_val_idx = fg.val_idx.cpu()
            fg_test_idx = fg.test_idx.cpu()
            g0_train_idx = map_idx_fraud(fg_train_idx, g0_orig_id)
            g0_val_idx = map_idx_fraud(fg_val_idx, g0_orig_id)
            g0_test_idx = map_idx_fraud(fg_test_idx, g0_orig_id)
            g0.ndata['train_mask'] = gen_mask(g0.number_of_nodes(), g0_train_idx)
            g0.ndata['valid_mask'] = gen_mask(g0.number_of_nodes(), g0_val_idx)
            g0.ndata['test_mask'] = gen_mask(g0.number_of_nodes(), g0_test_idx)
            print(f'saving file to {fn}')
            dgl.save_graphs(fn, g0)

            g0_train_idx, g0_val_idx, g0_test_idx = map(lambda x: torch.from_numpy(x).long(),
                                                        (g0_train_idx, g0_val_idx, g0_test_idx))
        else:
            print(f'loading file from {fn}')
            data = dgl.load_graphs(fn)
            g0 = data[0][0]
            g0_labels = g0.ndata['label']
            g0_train_idx = map_mask(g0.ndata['train_mask'])
            g0_val_idx = map_mask(g0.ndata['valid_mask'])
            g0_test_idx = map_mask(g0.ndata['test_mask'])

        g0, g0_labels, g0_train_idx, g0_val_idx, g0_test_idx = map(
            lambda x: x.to(self.device), (g0, g0_labels, g0_train_idx, g0_val_idx, g0_test_idx)
        )

        print(f"Subgraph: node number {g0.number_of_nodes()}, edge number {g0.number_of_edges()}")
        print(f"Finish loading subgraph {self.gid}")
        print('-' * 70)

        return g0, g0_labels, g0_train_idx, g0_val_idx, g0_test_idx

def load_sg(g, args, w_hetero, HG_Score):
    nodes_list = part_graph_fast(args.dataset, args.num_parts, args.part_mode,
                                args.first_time, w_hetero, HG_Score, args.ada_tau)
    sg_list = []
    for idx, nodes_id in enumerate(nodes_list):
        sg = SubFraudGraph(g, args, idx, args.num_parts, args.part_mode,
                           nodes_id, 'cpu', args.first_time)
        sg_list.append(sg)
    return sg_list