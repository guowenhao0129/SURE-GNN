import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, confusion_matrix, \
    average_precision_score
import os, time, numpy as np
from scipy.sparse import coo_matrix
import pymetis

def acc_fraud(pred_numpy,labels_numpy):
    pred_labels = (pred_numpy > 0.5).astype(int)
    pred_labels = pred_labels.squeeze()
    if pred_labels.shape != labels_numpy.shape:
        raise ValueError(
            f"Shape mismatch: pred_labels shape {pred_labels.shape}, labels shape {labels_numpy.shape}")
    accuracy = accuracy_score(labels_numpy, pred_labels)
    return accuracy

def ap_fraud(pred_numpy,labels_numpy):
    pred_labels = (pred_numpy > 0.5).astype(int)
    pred_labels = pred_labels.squeeze()
    if pred_labels.shape != labels_numpy.shape:
        raise ValueError(
            f"Shape mismatch: pred_labels shape {pred_labels.shape}, labels shape {labels_numpy.shape}")
    ap_score = average_precision_score(labels_numpy, pred_numpy)
    return ap_score

def auc_fraud(pred_numpy,labels_numpy):

    if pred_numpy.shape[1] == 2:
        pred_positive_class = pred_numpy[:, 1]
    else:
        pred_positive_class = pred_numpy
    auc = roc_auc_score(labels_numpy, pred_positive_class)
    return auc
def f1_fraud(pred_numpy,labels_numpy):
    pred_labels = (pred_numpy > 0.5).astype(int)
    pred_labels = pred_labels.squeeze()
    if pred_labels.shape != labels_numpy.shape:
        raise ValueError(
            f"Shape mismatch: pred_labels shape {pred_labels.shape}, labels shape {labels_numpy.shape}")
    f1_macro = f1_score(labels_numpy, pred_labels, average='macro')
    return f1_macro
def conf_gmean(conf):
	tn, fp, fn, tp = conf.ravel()
	return (tp*tn/((tp+fn)*(tn+fp)))**0.5
def gmean_fraud(pred_numpy,labels_numpy):

    pred_labels = (pred_numpy > 0.5).astype(int)
    pred_labels = pred_labels.squeeze()
    if pred_labels.shape != labels_numpy.shape:
        raise ValueError(
            f"Shape mismatch: pred_labels shape {pred_labels.shape}, labels shape {labels_numpy.shape}")
    conf = confusion_matrix(labels_numpy, pred_labels)
    gmean = conf_gmean(conf)
    return gmean

def recall_fraud(pred_numpy,labels_numpy):

    pred_labels = (pred_numpy > 0.5).astype(int)
    pred_labels = pred_labels.squeeze()
    if pred_labels.shape != labels_numpy.shape:
        raise ValueError(
            f"Shape mismatch: pred_labels shape {pred_labels.shape}, labels shape {labels_numpy.shape}")
    recall = recall_score(labels_numpy, pred_labels, average='macro')
    return recall
def agg_train_logit(fullgraph, sg_list, logit_list, datatype):
    labels = fullgraph.labels
    full_logit = torch.zeros((fullgraph.n_nodes, logit_list[0].shape[1]), dtype=logit_list[0].dtype).to(logit_list[0].device)
    predcnt = torch.zeros(fullgraph.n_nodes).to(logit_list[0].device)
    for i in range(len(sg_list)):
        sg = sg_list[i]
        logit = logit_list[i]
        if datatype == 'train':
            idx = sg.train_idx
        elif datatype == 'val':
            idx = sg.val_idx
        else:
            idx = sg.test_idx
        sg_orid_id_idx = sg.get_orig_id(idx).long()
        predcnt[sg_orid_id_idx] = predcnt[sg_orid_id_idx] + 1
        full_logit[sg_orid_id_idx] = full_logit[sg_orid_id_idx] + logit[idx]
    valid_nodes = predcnt > 0
    full_logit[valid_nodes] = (full_logit[valid_nodes].T / predcnt[valid_nodes]).T
    if datatype == 'train':
        fullidx = fullgraph.train_idx
    elif datatype == 'val':
        fullidx = fullgraph.val_idx
    else:
        fullidx = fullgraph.test_idx
    return full_logit
def agg_pred_fraud(fullgraph, sg_list, pred_list, datatype, detail=False):
    labels = fullgraph.labels
    fullpred = torch.zeros((fullgraph.n_nodes, pred_list[0].shape[1]), dtype=pred_list[0].dtype).to(pred_list[0].device)
    predcnt = torch.zeros(fullgraph.n_nodes).to(pred_list[0].device)
    sg_acc = []
    sg_ap = []
    sg_roauc = []
    sg_f1 = []
    sg_gmean = []
    sg_recall = []

    for i in range(len(sg_list)):
        sg = sg_list[i]
        pred = pred_list[i]
        if datatype == 'train':
            idx = sg.train_idx
        elif datatype == 'val':
            idx = sg.val_idx
        else:
            idx = sg.test_idx
        sg_orid_id_idx = sg.get_orig_id(idx).long()
        predcnt[sg_orid_id_idx] = predcnt[sg_orid_id_idx] + 1
        fullpred[sg_orid_id_idx] = fullpred[sg_orid_id_idx] + pred[idx]

    valid_nodes = predcnt > 0
    fullpred[valid_nodes] = (fullpred[valid_nodes].T / predcnt[valid_nodes]).T
    #torch.save(fullpred,"fullpred.pt")
    if datatype == 'train':
        fullidx = fullgraph.train_idx
    elif datatype == 'val':
        fullidx = fullgraph.val_idx
    else:
        fullidx = fullgraph.test_idx

    pred_numpy = fullpred[fullidx].cpu().detach().numpy()
    labels_numpy = labels[fullidx].cpu().detach().numpy()
    labels = labels.unsqueeze(1)
    acc = acc_fraud(pred_numpy, labels_numpy)
    ap = ap_fraud(pred_numpy, labels_numpy)
    roauc = auc_fraud(pred_numpy, labels_numpy)
    f1 = f1_fraud(pred_numpy, labels_numpy)
    gmean = gmean_fraud(pred_numpy, labels_numpy)
    recall = recall_fraud(pred_numpy, labels_numpy)
    sg_acc.append(acc)
    sg_ap.append(ap)
    sg_roauc.append(roauc)
    sg_f1.append(f1)
    sg_gmean.append(gmean)
    sg_recall.append(recall)
    return sg_acc,sg_ap, sg_roauc, sg_f1, sg_gmean, sg_recall

def part_graph_fast(dataset, part_num, scalor_factor, first_time,
                    w_hetero, HG_Score, ada_tau, etype='homo',
                    prune_strategy: str = 'remove',
                    down_weight: int = 1
                    ):
    base_dir = os.path.join('./dataset', dataset)
    os.makedirs(base_dir, exist_ok=True)
    file_name1 = os.path.join(base_dir, f'mode{scalor_factor}.npy')
    file_name2 = os.path.join(base_dir, 'nodelist.npy')
    struct_path = os.path.join('./dataset', dataset, f'struct_{etype}.npz')
    if not os.path.exists(struct_path):
        raise FileNotFoundError(
            f': Offline Caching Not Exist for {struct_path}\n'
        )
    z = np.load(struct_path, allow_pickle=False, mmap_mode='r')
    src, dst = z['src'], z['dst']
    N = int(z['num_nodes'][0])
    E = len(src)

    if isinstance(w_hetero, torch.Tensor):
        w = w_hetero.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        w = np.asarray(w_hetero, dtype=np.float32)
    if w.shape[0] != E:
        raise ValueError(f'lenth w_hetero={w.shape[0]} Error')
    scale_factor = int(scalor_factor)
    weights = np.rint((1.0 - w) * scale_factor).astype(np.int32)
    np.maximum(weights, 1, out=weights)

    rows0 = np.concatenate([src, dst])
    cols0 = np.concatenate([dst, src])
    data0 = np.concatenate([weights, weights]).astype(np.int32, copy=False)
    keep0 = rows0 != cols0
    A0 = coo_matrix((data0[keep0], (rows0[keep0], cols0[keep0])), shape=(N, N), dtype=np.int32).tocsr()

    if first_time or (not os.path.exists(file_name1) or not os.path.exists(file_name2)):
        cuts1, parts = pymetis.part_graph(
            nparts=part_num,
            xadj=A0.indptr.astype(np.int32),
            adjncy=A0.indices.astype(np.int32),
            eweights=A0.data.astype(np.int32),
        )
        node_list = np.arange(N, dtype=np.int64)
        np.save(file_name1, np.asarray(parts, dtype=np.int32))
        np.save(file_name2, node_list)
    else:
        parts = np.load(file_name1)
        node_list = np.load(file_name2)

    parts_np = np.asarray(parts, dtype=np.int32)
    tau = float(ada_tau)
    sign = np.where(w > tau, 1, -1).astype(np.int32)
    part_src = parts_np[src]
    part_dst = parts_np[dst]
    internal = (part_src == part_dst)

    S_global = int(sign.sum())
    S_global_avg = S_global / float(part_num)

    if prune_strategy == 'remove':
        keep_mask = np.ones(E, dtype=bool)
    elif prune_strategy == 'downweight':
        weights_after = np.array(weights, copy=True)
        down_weight = max(int(down_weight), 1)
    else:
        raise ValueError("prune_strategy must 'remove' or 'downweight'")

    total_pruned = 0
    for k in range(part_num):
        mask_k = internal & (part_src == k)
        if not np.any(mask_k):
            continue

        H_k = float(w[mask_k].mean())
        if H_k <= HG_Score:
            continue

        S_k = int(sign[mask_k].sum())
        n_cut_k = int(max(0, np.round(S_k - S_global_avg)))
        if n_cut_k <= 0:
            continue

        cand = mask_k & (sign == 1)
        if not np.any(cand):
            continue

        idx = np.flatnonzero(cand)
        if n_cut_k >= idx.size:
            sel = idx
        else:
            rel = np.argpartition(w[idx], -n_cut_k)[-n_cut_k:]
            sel = idx[rel]

        if prune_strategy == 'remove':
            keep_mask[sel] = False
        else:  # downweight
            weights_after[sel] = down_weight

        total_pruned += sel.size

    if prune_strategy == 'remove':
        out_path = os.path.join(base_dir, f'edge_keep_mask_{etype}.npy')
        np.save(out_path, keep_mask)
    else:
        out_path = os.path.join(base_dir, f'edge_weights_after_{etype}.npy')
        np.save(out_path, weights_after.astype(np.int32))

    part_id_list = []
    for j in range(part_num):
        part_id_list.append(node_list[np.where(parts_np == j)[0]])
    return part_id_list
