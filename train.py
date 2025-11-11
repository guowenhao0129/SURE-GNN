import time
import argparse
import random
import dgl
import numpy as np
import torch
import torch.optim as optim
import scipy.sparse as sp
import torch.nn.functional as F
import dgl.dataloading
import dgl.dataloading

from train_subgraph import train_subgraph
from subgraph import load_sg, FullFraudGraph, FullAggred
from model_he import  Hetero_Evaluator, EarlyStop, fraud_model
from part_utils import agg_pred_fraud, agg_train_logit
from sklearn.metrics import f1_score


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def normalize(mx):
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def best_f1_macro(y01, s, n_tau=81, use_quantile=True):
    if use_quantile:
        lo = torch.quantile(s, 0.02)
        hi = torch.quantile(s, 0.98)
        taus = torch.linspace(lo, hi, steps=n_tau, device=s.device)
    else:
        taus = torch.linspace(-1.0, 1.0, steps=n_tau, device=s.device)

    best = torch.tensor(0.0, device=s.device)
    for t in taus:
        pred = (s > t)
        tp = (pred & (y01 == 1)).sum()
        fp = (pred & (y01 == 0)).sum()
        fn = ((~pred) & (y01 == 1)).sum()
        tn = ((~pred) & (y01 == 0)).sum()
        f1_pos = (2*tp).float() / (2*tp + fp + fn + 1e-12)
        f1_neg = (2*tn).float() / (2*tn + fn + fp + 1e-12)
        f1_macro = 0.5 * (f1_pos + f1_neg)
        best = torch.maximum(best, f1_macro)
    return best

def evaluate_f1(labels, logits, threshold=0.5):
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = np.where(labels == 0, 0, 1)
    preds = np.where(probs > threshold, 1, 0)
    f1_macro = f1_score(labels, preds, average='macro')
    return f1_macro

def run_uni_subfraud(args, sg, w_tanhscore):
    model = fraud_model(args, sg.in_feats, sg.n_classes, w_tanhscore, sg.graph)
    model.to(device)
    best_f1, best_test_roauc, final_pred, final_logit = train_subgraph(model, sg.graph, args)
    return best_f1, best_test_roauc, final_pred, final_logit

def run(args,fullaggred, w_tanhscore):
    pred_list = []
    logit_list = []
    for i, sg in enumerate(sg_list):
        sg.to_device('cpu')
        _, _, pred, logit = run_uni_subfraud(args, sg, w_tanhscore)
        pred_list.append(pred)
        logit_list.append(logit)
        sg.to_device('cpu')
    # train_logit = agg_train_logit(fullaggred, sg_list, logit_list, 'train')
    # torch.save(train_logit, args.dataset + "full"+"train_logit.pt")
    train_acc, train_ap, train_roauc, train_f1,train_gmean, train_recall = agg_pred_fraud(fullaggred, sg_list, pred_list, 'train')
    val_acc, val_ap, val_roauc, val_f1, val_gmean, val_recall = agg_pred_fraud(fullaggred, sg_list, pred_list, 'val')
    test_acc, test_ap, test_roauc, test_f1, test_gmean, test_recall = agg_pred_fraud(fullaggred, sg_list, pred_list, 'test')
    # print('Subgraph 0 / Subgraph 1 / ... / Full graph ')
    # print(f'train acc score:', [round(i, 4) for i in train_acc])
    # print(f'val acc score:', [round(i, 4) for i in val_acc])
    # print(f'test acc score:', [round(i, 4) for i in test_acc])
    # print(f'train ap score:', [round(i, 4) for i in train_ap])
    # print(f'val ap score:', [round(i, 4) for i in val_ap])
    # print(f'test ap score:', [round(i, 4) for i in test_ap])
    # print(f'train roauc score:', [round(i, 4) for i in train_roauc])
    # print(f'val roauc score:', [round(i, 4) for i in val_roauc])
    # print(f'test roauc score:', [round(i, 4) for i in test_roauc])
    # print(f'train f1 score:', [round(i, 4) for i in train_f1])
    # print(f'val f1 score:', [round(i, 4) for i in val_f1])
    # print(f'test f1 score:', [round(i, 4) for i in test_f1])

    return val_acc[-1], test_acc[-1], val_ap[-1], test_ap[-1], val_roauc[-1], test_roauc[-1], val_f1[-1], test_f1[-1],\
        val_gmean[-1], test_gmean[-1], val_recall[-1], test_recall[-1]

if __name__ == '__main__':
    global device, sg_list
    startall = time.time()
    argparser = argparse.ArgumentParser("SURE-GNN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--dataset", type=str, default="amazon", help="dataset name")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--seed", type=int, default=42, help="seed")
    argparser.add_argument("--n-runs", type=int, default=1, help="running times")
    argparser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    argparser.add_argument('--save-model', action="store_true", help="whether to save model")
    argparser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    argparser.add_argument("--n-hidden", type=int, default=64, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--first-time", action='store_true', help="run evaluation and partition for first time")
    argparser.add_argument("--num-parts", type=int, default=2, help="num of partitions")
    argparser.add_argument("--part-mode", type=int, default=100, help="how to partition the graph")
    argparser.add_argument("--edgelr", type=float, default=0.001, help="edge learning rate")
    argparser.add_argument("--edgewd", type=float, default=0.00001, help="weight decay")
    argparser.add_argument('--dataset_path', type=str, default='./dataset/')
    argparser.add_argument('--result_path', type=str, default='')
    argparser.add_argument('--early_stop', type=int, default=30)
    argparser.add_argument('--evaluation_epoch', type=int, default=1000)
    argparser.add_argument('--caching_load', type=int, default=1)
    argparser.add_argument('--adjust', type=int, default=1,help="adaptive adjustment")
    argparser.add_argument('--ada_tau', type=float, default=0.5, help="adaptive adjustment threshold")
    argparser.add_argument('--alpha', type=float, default=0.5)
    args = argparser.parse_args()
    device = torch.device("cpu") if args.gpu < 0 else torch.device("cuda:" + str(args.gpu))
    args.device = device
    dataset_path = args.dataset_path + args.dataset + '.dgl'
    dataset = dgl.load_graphs(dataset_path)[0][0]
    best_f1, best_auc, best_accuracy = 0., 0., 0.
    best_loss = float('inf')
    HG_Score =0.0
    if args.caching_load == 1:
        w_tanhscore = torch.load(f'w_tanhscore_{args.dataset}.pth', map_location='cpu')
    else:
        dataset = dataset.to(device)
        feat = dataset.ndata['feature'].to(device=device, dtype=torch.float32)
        feat = F.normalize(feat, p=2, dim=1, eps=1e-12)
        dataset.ndata['feature'] = feat
        model = Hetero_Evaluator(dataset).to(device)
        optimizer = optim.Adam(params=model.parameters(), lr=args.edgelr, weight_decay=args.edgewd)
        early_stop = EarlyStop(args.early_stop)
        edge_train_mask = dataset.edges['homo'].data['train_mask'].bool()
        train_labels = dataset.edges['homo'].data['label'][edge_train_mask].cpu()
        edge_valid_mask = dataset.edges['homo'].data['valid_mask'].bool()
        valid_labels = dataset.edges['homo'].data['label'][edge_valid_mask].cpu()
        valid_y01 = (valid_labels > 0).long().to(device)
        start_time = time.time()
        best_f1 = -1.0
        best_loss = float('inf')
        for e in range(args.evaluation_epoch):
            model.train()
            loss = model.loss(dataset)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                model.eval()
                logits = model(dataset)
                ed = dataset.edges['homo'].data
                if 'phi_raw' in ed:
                    s_all = ed['phi_raw'].float()
                else:
                    s_all = ed['score'].float()
                if hasattr(model, 'score_scale') and hasattr(model, 'score_shift'):
                    s_eff = (s_all - model.score_shift) * model.score_scale
                else:
                    s_eff = s_all
                s_val = s_eff[edge_valid_mask]
                f1 = best_f1_macro(valid_y01, s_val)
                if f1 > best_f1:
                    best_f1 = f1
                    best_loss = loss.item()
                    w_tanhscore = logits
                if e % 10 == 0:
                    print(f'Loss:{loss.item():.6f} ,bestloss:{best_loss:.6f} ,best F1-Macro:{best_f1:.6f}')
                do_store, do_stop = early_stop.step(best_accuracy, e)
                if do_stop:
                    print('Early Stop and Break')
                    break
        he_score = 0.5 * (w_tanhscore + 1.0)
        he_score = torch.clamp(he_score, 0.0, 1.0)
        HG_Score = he_score.mean().item()

    dataset.edges['homo'].data['edge_score'] =w_tanhscore.float()
    fg = FullFraudGraph(args, dataset)
    sg_list = load_sg(fg.graph, args, w_tanhscore, HG_Score)
    fullaggred =FullAggred(fg)
    del fg

    val_accs, test_accs = [], []
    val_aps, test_aps = [], []
    val_roaucs, test_roaucs = [], []
    val_f1s, test_f1s = [], []
    val_gmeans, test_gmeans = [], []
    val_recalls, test_recalls = [], []
    for i in range(args.n_runs):
        val_acc, test_acc, val_ap, test_ap, val_roauc, test_roauc, val_f1, test_f1, \
            val_gmean, test_gmean, val_recall, test_recall = run(args, fullaggred, w_tanhscore)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        val_aps.append(val_ap)
        test_aps.append(test_ap)
        val_roaucs.append(val_roauc)
        test_roaucs.append(test_roauc)
        val_f1s.append(val_f1)
        test_f1s.append(test_f1)
        val_gmeans.append(val_gmean)
        test_gmeans.append(test_gmean)
        val_recalls.append(val_recall)
        test_recalls.append(test_recall)

    print(f"Runned {args.n_runs} times")
    print(f"Final val accuracy: {np.mean(val_accs, axis=0)} ± {np.std(val_accs, axis=0)}")
    print(f"Final test accuracy: {np.mean(test_accs, axis=0)} ± {np.std(test_accs, axis=0)}")
    print(f"Final val ap: {np.mean(val_aps, axis=0)} ± {np.std(val_aps, axis=0)}")
    print(f"Final test ap: {np.mean(test_aps, axis=0)} ± {np.std(test_aps, axis=0)}")
    print(f"Final val roauc: {np.mean(val_roaucs, axis=0)} ± {np.std(val_roaucs, axis=0)}")
    print(f"Final test roauc: {np.mean(test_roaucs, axis=0)} ± {np.std(test_roaucs, axis=0)}")
    print(f"Final val f1-macro: {np.mean(val_f1s, axis=0)} ± {np.std(val_f1s, axis=0)}")
    print(f"Final test f1-macro: {np.mean(test_f1s, axis=0)} ± {np.std(test_f1s, axis=0)}")
    print(f"Final val gmean: {np.mean(val_gmeans, axis=0)} ± {np.std(val_gmeans, axis=0)}")
    print(f"Final test gmean: {np.mean(test_gmeans, axis=0)} ± {np.std(test_gmeans, axis=0)}")
    print(f"Final val recall: {np.mean(val_recalls, axis=0)} ± {np.std(val_recalls, axis=0)}")
    print(f"Final test recall: {np.mean(test_recalls, axis=0)} ± {np.std(test_recalls, axis=0)}")
    print(args)
    print('Finish')