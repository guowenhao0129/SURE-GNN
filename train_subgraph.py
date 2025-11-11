import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, roc_auc_score, average_precision_score
from model_he import EarlyStop

in_feats, n_classes = None, None
epsilon = 1 - math.log(2)
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro',zero_division=0)
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def train_subgraph(model, sg_graph, args):
    print('train_bw sg_graph',sg_graph)
    features = sg_graph.ndata['feature'].to('cuda')
    if features.dtype == torch.float64 or torch.int64:
        features = features.float()
    labels = sg_graph.ndata['label'].to('cuda')

    train_mask = sg_graph.ndata['train_mask'].clone().detach().to('cuda')
    val_mask = sg_graph.ndata['valid_mask'].clone().detach().to('cuda')
    test_mask = sg_graph.ndata['test_mask'].clone().detach().to('cuda')

    train_mask = train_mask.to(torch.bool)
    val_mask = val_mask.to(torch.bool)
    test_mask = test_mask.to(torch.bool)
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
    best_loss = 100
    best_auc = 0
    best_pre = 0
    best_score= 0

    weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    early_stop = EarlyStop(patience=100)

    for e in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight], device='cuda'))
        loss.backward()
        optimizer.step()
        model.eval()
        probs = logits.softmax(1)
        f1, thres = get_best_f1(labels[val_mask].to('cpu'), probs[val_mask].to('cpu'))
        preds = torch.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        trec = recall_score(labels[test_mask].to('cpu'), preds[test_mask].to('cpu'))
        tpre = average_precision_score(labels[test_mask].to('cpu'), preds[test_mask].to('cpu'))
        tmf1 = f1_score(labels[test_mask].to('cpu'), preds[test_mask].to('cpu'), average='macro')
        tauc = roc_auc_score(labels[test_mask].to('cpu'), probs[test_mask].to('cpu')[:, 1].detach().numpy())
        score = tmf1 + tauc
        do_store, do_stop = early_stop.step(score)
        if do_store:
            best_score = score
            best_loss = loss
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
            pred_y = probs
            final_logit = logits
        if do_stop:
            print(f"Early stopping at epoch {e} with best AUC: {final_tauc}, best F1: {final_tmf1}")
            break
        if e % 10 == 0:
            print('Epoch {}, loss: {:.4f}, final_tmf1 {:.4f},final_tauc: {:.4f}'
                  .format(e, best_loss,  final_tmf1, final_tauc, best_pre))

    final_tauc = best_auc
    final_pred = pred_y[:, 1].unsqueeze(1)
    return final_tmf1, final_tauc, final_pred, final_logit
