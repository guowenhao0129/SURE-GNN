import argparse
import os
import sys
import time
import dgl
import torch
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split

def build_and_save_structure(g, dataset, etype='homo'):
    os.makedirs(os.path.join('./dataset', dataset), exist_ok=True)
    out = os.path.join('./dataset', dataset, 'struct_{}.npz'.format(etype))
    g = g.to('cpu')
    N = g.num_nodes()
    src, dst = g.edges(etype=etype)
    src = src.numpy().astype(np.int64, copy=False)
    dst = dst.numpy().astype(np.int64, copy=False)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    meta = {
        'num_nodes': np.array([N], dtype=np.int64),
        'directed': np.array([1], dtype=np.int8),
        'etype': np.array([0], dtype=np.int8),
        'created_ts': np.array([int(time.time())], dtype=np.int64)
    }
    np.savez_compressed(out, src=src, dst=dst, **meta)
    print(f'[OK] save structure: {out}, nodes={N}, edges={len(src)}')

def generate_edges_labels(edges, labels, train_idx, valid_idx):
    print('generate edge labels')
    row, col = edges
    edge_labels = []
    edge_train_mask = []
    edge_valid_mask = []
    
    for i, j in zip(row, col):
        i = i.item()
        j = j.item()
        if labels[i] == labels[j]:
            edge_labels.append(0)
        else:
            edge_labels.append(1)
        if i in train_idx and j in train_idx:
            edge_train_mask.append(1)
        else:
            edge_train_mask.append(0)
        if i in valid_idx and j in valid_idx:
            edge_valid_mask.append(1)
        else:
            edge_valid_mask.append(0)
    edge_labels = torch.Tensor(edge_labels).long()
    edge_train_mask = torch.Tensor(edge_train_mask).bool()
    edge_valid_mask = torch.Tensor(edge_valid_mask).bool()
    return edge_labels, edge_train_mask, edge_valid_mask

if __name__ == '__main__':
    dataset_path = './dataset/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amazon')
    args = parser.parse_args()
    print('**********************************')
    print(f'Generate {args.dataset}')
    print('**********************************')
    if args.dataset == 'yelp':
        if os.path.exists(dataset_path+'yelp.dgl'):
            print('Dataset yelp has been created')
            sys.exit()
        print('Convert to DGL Graph.')
        yelp_path = dataset_path+'YelpChi.mat'
        yelp = scio.loadmat(yelp_path)
        feats = yelp['features'].todense()
        features = torch.from_numpy(feats)
        lbs = yelp['label'][0]
        labels = torch.from_numpy(lbs)
        homo = yelp['homo']
        homo = homo+homo.transpose()
        homo = homo.tocoo()
        rur = yelp['net_rur']
        rur = rur+rur.transpose()
        rur = rur.tocoo()
        rtr = yelp['net_rtr']
        rtr = rtr+rtr.transpose()
        rtr = rtr.tocoo()
        rsr = yelp['net_rsr']
        rsr = rsr+rsr.transpose()
        rsr = rsr.tocoo()
        yelp_graph_structure = {
            ('r','homo','r'):(torch.tensor(homo.row), torch.tensor(homo.col)),
            ('r','u','r'):(torch.tensor(rur.row), torch.tensor(rur.col)),
            ('r','t','r'):(torch.tensor(rtr.row), torch.tensor(rtr.col)),
            ('r','s','r'):(torch.tensor(rsr.row), torch.tensor(rsr.col))
        }
        yelp_graph = dgl.heterograph(yelp_graph_structure)
        for t in yelp_graph.etypes:
            yelp_graph.remove_self_loop(t)
            yelp_graph.add_self_loop(etype=t)
        yelp_graph.nodes['r'].data['feature'] = features
        yelp_graph.nodes['r'].data['label'] = labels
        print('Generate dataset partition.')
        train_ratio = 0.4
        test_ratio = 0.67
        index = list(range(len(lbs)))
        dataset_l = len(lbs)
        train_idx, rest_idx, train_lbs, rest_lbs = train_test_split(index, lbs, stratify=lbs, train_size=train_ratio, random_state=2, shuffle=True)
        valid_idx, test_idx, _,_ = train_test_split(rest_idx, rest_lbs, stratify=rest_lbs, test_size=test_ratio, random_state=2, shuffle=True)
        train_mask = torch.zeros(dataset_l, dtype=torch.bool)
        train_mask[np.array(train_idx)] = True
        valid_mask = torch.zeros(dataset_l, dtype=torch.bool)
        valid_mask[np.array(valid_idx)] = True
        test_mask = torch.zeros(dataset_l, dtype=torch.bool)
        test_mask[np.array(test_idx)] = True
        yelp_graph.nodes['r'].data['train_mask'] = train_mask
        yelp_graph.nodes['r'].data['valid_mask'] = valid_mask
        yelp_graph.nodes['r'].data['test_mask'] = test_mask
        homo_edges = yelp_graph.edges(etype='homo')
        homo_labels, homo_train_mask, homo_valid_mask = generate_edges_labels(homo_edges, lbs, train_idx, valid_idx)
        yelp_graph.edges['homo'].data['label'] = homo_labels
        yelp_graph.edges['homo'].data['train_mask'] = homo_train_mask
        yelp_graph.edges['homo'].data['valid_mask'] = homo_valid_mask
        dgl.save_graphs(dataset_path+'yelp.dgl', yelp_graph)
        build_and_save_structure(yelp_graph, dataset='yelp', etype='homo')

    elif args.dataset == 'amazon':
        if os.path.exists(dataset_path+'amazon.dgl'):
            print('dataset amazon has been created')
            sys.exit()
        amazon_path = dataset_path+'Amazon.mat'
        amazon = scio.loadmat(amazon_path)
        feats = amazon['features'].todense()
        features = torch.from_numpy(feats).float()
        lbs = amazon['label'][0]
        labels = torch.from_numpy(lbs).long()
        homo = amazon['homo']
        homo = homo+homo.transpose()
        homo = homo.tocoo()
        upu = amazon['net_upu']
        upu = upu+upu.transpose()
        upu = upu.tocoo()
        usu = amazon['net_usu']
        usu = usu+usu.transpose()
        usu = usu.tocoo()
        uvu = amazon['net_uvu']
        uvu = uvu+uvu.transpose()
        uvu = uvu.tocoo()
        
        amazon_graph_structure = {
            ('r','homo','r'):(torch.tensor(homo.row), torch.tensor(homo.col)),
            ('r','p','r'):(torch.tensor(upu.row), torch.tensor(upu.col)),
            ('r','s','r'):(torch.tensor(usu.row), torch.tensor(usu.col)),
            ('r','v','r'):(torch.tensor(uvu.row), torch.tensor(uvu.col))
        }
        amazon_graph = dgl.heterograph(amazon_graph_structure)
        for t in amazon_graph.etypes:
            amazon_graph.remove_self_loop(t)
            amazon_graph.add_self_loop(etype=t)
        amazon_graph.nodes['r'].data['feature'] = features
        amazon_graph.nodes['r'].data['label'] = labels
        print('Generate dataset partition.')
        train_ratio = 0.4
        test_ratio = 0.67
        index = list(range(3305, len(labels)))
        dataset_l = len(lbs)
        train_idx, rest_idx, train_lbs, rest_lbs = train_test_split(index, lbs[3305:], stratify=lbs[3305:], train_size=train_ratio, random_state=2, shuffle=True)
        valid_idx, test_idx, _,_ = train_test_split(rest_idx, rest_lbs, stratify=rest_lbs, test_size=test_ratio, random_state=2, shuffle=True)
        train_mask = torch.zeros(dataset_l, dtype=torch.bool)
        train_mask[np.array(train_idx)] = True
        valid_mask = torch.zeros(dataset_l, dtype=torch.bool)
        valid_mask[np.array(valid_idx)] = True
        test_mask = torch.zeros(dataset_l, dtype=torch.bool)
        test_mask[np.array(test_idx)] = True
        amazon_graph.nodes['r'].data['train_mask'] = train_mask
        amazon_graph.nodes['r'].data['valid_mask'] = valid_mask
        amazon_graph.nodes['r'].data['test_mask'] = test_mask
        print('Generate edge labels.')
        homo_edges = amazon_graph.edges(etype='homo')
        homo_labels, homo_train_mask, homo_valid_mask = generate_edges_labels(homo_edges, lbs, train_idx, valid_idx)
        amazon_graph.edges['homo'].data['label'] = homo_labels
        amazon_graph.edges['homo'].data['train_mask'] = homo_train_mask
        amazon_graph.edges['homo'].data['valid_mask'] = homo_valid_mask
        
        dgl.save_graphs(dataset_path+'amazon.dgl', amazon_graph)
        build_and_save_structure(amazon_graph, dataset='amazon', etype='homo')
    print('***************end****************')


