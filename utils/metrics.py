import torch
import numpy as np
from sklearn.metrics import average_precision_score
from scipy.special import softmax

def cal_metrics(p_labels, g_labels):
    tp, fp, fn, tn = 0, 0, 0, 0
    target = g_labels

    pred = p_labels

    tp += ((pred + target)==2).sum(axis=0)
    fp += ((pred - target)==1).sum(axis=0)
    fn += ((pred - target)==-1).sum(axis=0)
    tn += ((pred + target)==0).sum(axis=0)
    p_c = [float(tp[i] / (tp[i] + fp[i])) if tp[i] > 0 else 0.0 for i in range(len(tp))]
    r_c = [float(tp[i] / (tp[i] + fn[i])) if tp[i] > 0 else 0.0 for i in range(len(tp))]

    mean_p_c = sum(p_c) / len(p_c)
    mean_r_c = sum(r_c) / len(r_c)
    if mean_p_c==0 and mean_r_c ==0:
        mean_f_c = 0.
    else:
        mean_f_c = 2 * mean_p_c * mean_r_c / (mean_p_c + mean_r_c)

    p_o = tp.sum() / (tp + fp).sum()
    r_o = tp.sum() / (tp + fn).sum()
    if p_o==0 and r_o ==0:
        f_o = 0.
    else:
        f_o = 2 * p_o * r_o / (p_o + r_o)

    Result = {}
    Result['CP'] = mean_p_c
    Result['CR'] = mean_r_c
    Result['CF1'] = mean_f_c
    Result['OP'] = p_o
    Result['OR'] = r_o
    Result['OF1'] = f_o

    return Result

def compute_map(y_scores: np.ndarray, y_true: np.ndarray) -> float:

    assert y_scores.shape == y_true.shape, "Shape of prediction scores and ground truth labels must be the same."
    
    n_classes = y_scores.shape[1]
    APs = []
    
    for i in range(n_classes):
        if np.sum(y_true[:, i]) == 0:
            continue
        ap = average_precision_score(y_true[:, i], y_scores[:, i])
        APs.append(ap)
    
    if len(APs) == 0:
        return 0.0
    return np.mean(APs), APs


def multilabel_evaluation(scores, targets, k=1):
    scores = torch.tensor(scores)
    targets[targets == -1] = 0
    n, c = scores.size()
    pred = np.zeros((n, c))
    index = scores.topk(k, 1, True, True)[1].numpy()
    for i in range(n):
        for ind in index[i]:
            pred[i, ind] = 1
    return cal_metrics(pred, targets)

