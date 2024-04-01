import sys
import numpy as np
import torch
from utils_doc.tools import get_preds_mask, get_preds_prob_map
from sklearn.metrics import roc_auc_score, average_precision_score


def get_tp(preds_mask, mask):
    assert len(preds_mask.shape) == 3
    assert len(mask.shape) == 3
    return np.logical_and(np.equal(preds_mask, 1), np.equal(mask, 1)).sum(axis=(1, 2))


def get_tn(preds_mask, mask):
    assert len(preds_mask.shape) == 3
    assert len(mask.shape) == 3
    return np.logical_and(np.equal(preds_mask, 0), np.equal(mask, 0)).sum(axis=(1, 2))


def get_fp(preds_mask, mask):
    assert len(preds_mask.shape) == 3
    assert len(mask.shape) == 3
    return np.logical_and(np.equal(preds_mask, 1), np.equal(mask, 0)).sum(axis=(1, 2))


def get_fn(preds_mask, mask):
    assert len(preds_mask.shape) == 3
    assert len(mask.shape) == 3
    return np.logical_and(np.equal(preds_mask, 0), np.equal(mask, 1)).sum(axis=(1, 2))


def get_tp_tn_fp_fn(preds_mask, mask):
    assert len(preds_mask.shape) == 3
    assert len(mask.shape) == 3
    return get_tp(preds_mask, mask), get_tn(preds_mask, mask), get_fp(preds_mask, mask), get_fn(preds_mask, mask)


def AUC(pred_prob_map, mask):

    batch_size = pred_prob_map.shape[0]
    auc_arr = np.array([0.] * batch_size)
    pred_prob_map = pred_prob_map.reshape(batch_size, -1)
    mask = mask.reshape(batch_size, -1)
    for i in range(batch_size):
        try:
            auc_arr[i] = roc_auc_score(mask[i], pred_prob_map[i])
        except ValueError:
            auc_arr[i] = 1.0
    return auc_arr


def AP(pred_prob_map, mask):

    batch_size = pred_prob_map.shape[0]
    ap_arr = np.array([0.] * batch_size)
    pred_prob_map = pred_prob_map.reshape(batch_size, -1)
    mask = mask.reshape(batch_size, -1)
    for i in range(batch_size):
        try:
            ap_arr[i] = average_precision_score(mask[i], pred_prob_map[i])
        except ValueError:
            ap_arr[i] = 1.0
    return ap_arr


def matthews_correlation_coefficient(confusion_matrix):
    tp, tn, fp, fn = confusion_matrix
    for i in range(tp.shape[0]):
        if tp[i] == 0 and fn[i] == 0:
            tp[i] = tn[i]
            fn[i] = fp[i]
            fp[i] = 0
            tn[i] = 0
    N = tn + tp + fn + fp
    S = (tp + fn) / N
    P = (tp + fp) / N
    mcc_denominator = np.sqrt(P * S * (1 - S) * (1 - P))
    mcc_denominator = np.nan_to_num(mcc_denominator, nan=float('inf'), posinf=float('inf'), neginf=float('-inf'))
    mcc_denominator[mcc_denominator == 0.0] = float('inf')
    mcc = (tp / N - S * P) / mcc_denominator
    return np.nan_to_num(mcc)


def f1_score(confusion_matrix):
    tp, tn, fp, fn = confusion_matrix
    for i in range(tp.shape[0]):
        if tp[i] == 0 and fn[i] == 0:
            tp[i] = tn[i]
            fn[i] = fp[i]
            fp[i] = 0
            tn[i] = 0

    f1 = 2 * tp / (2 * tp + fp + fn)
    return np.nan_to_num(f1)


def iou_measure(confusion_matrix):
    tp, tn, fp, fn = confusion_matrix
    for i in range(len(tp)):
        if tp[i] == 0 and fn[i] == 0:
            tp[i] = tn[i]
            fn[i] = fp[i]
            fp[i] = 0
            tn[i] = 0

    iou = tp / (tp + fp + fn)
    return np.nan_to_num(iou)


def get_Precision(confusion_matrix):
    tp, _, fp, _ = confusion_matrix
    precision = tp / (tp + fp)

    return np.nan_to_num(precision)


def get_Recall(confusion_matrix):
    tp, tn, fp, fn = confusion_matrix
    recall = tp / (tp + fn)

    return np.nan_to_num(recall)

def get_accuracy(confusion_matrix):
    tp, tn, fp, fn = confusion_matrix
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return np.nan_to_num(accuracy)

def get_metrics(logits, mask, threshold='best', best_metric='f1'):

    if len(mask.shape) == 4:
        if mask.shape[1] == 1:
            mask = mask.reshape(mask.shape[0], mask.shape[2], mask.shape[3])
        else:
            print(f'mask shape {mask.shape} is illegal!')
            sys.exit(1)
    if len(logits.shape) == 4:
        if logits.shape[1] == 1:
            logits = logits.reshape(logits.shape[0], logits.shape[2], logits.shape[3])
        else:
            print(f'logits shape {logits.shape} is illegal!')
            sys.exit(1)

    auc = AUC(get_preds_prob_map(logits), mask)
    ap = AP(get_preds_prob_map(logits), mask)

    if type(threshold) is not str:
        pred_mask = get_preds_mask(logits, threshold=threshold)
        confusion_matrix = get_tp_tn_fp_fn(pred_mask.cpu(), mask.cpu())
        f1 = f1_score(confusion_matrix)
        mcc = matthews_correlation_coefficient(confusion_matrix)
        iou = iou_measure(confusion_matrix)
        acc = get_accuracy(confusion_matrix)
        p = get_Precision(confusion_matrix)
        r = get_Recall(confusion_matrix)
        return list(auc), list(f1), list(mcc), list(iou), list(acc), list(p), list(r), list(ap), threshold
        # return list(auc), list(f1), list(mcc), list(iou), list(acc), threshold
    elif threshold == 'best':
        threshold_list = np.arange(0.1, 1.0, 0.1)
        f1_list, mcc_list, iou_list, acc_list, p_list, r_list = [], [], [], [], [], []
        for threshold in threshold_list:
            pred_mask = get_preds_mask(logits, threshold=threshold)
            confusion_matrix = get_tp_tn_fp_fn(pred_mask.cpu(), mask.cpu())
            f1 = f1_score(confusion_matrix)
            mcc = matthews_correlation_coefficient(confusion_matrix)
            iou = iou_measure(confusion_matrix)
            acc = get_accuracy(confusion_matrix)
            p = get_Precision(confusion_matrix)
            r = get_Recall(confusion_matrix)
            f1_list.append(list(f1))
            mcc_list.append(list(mcc))
            iou_list.append(list(iou))
            acc_list.append(list(acc))
            p_list.append(list(p))
            r_list.append(list(r))
        f1 = np.max(f1_list, axis=0)
        mcc = np.max(mcc_list, axis=0)
        iou = np.max(iou_list, axis=0)
        acc = np.max(acc_list, axis=0)
        p = np.max(p_list, axis=0)
        r = np.max(r_list, axis=0)

        if best_metric == 'f1':
            best_thresholds = (np.argmax(f1_list, axis=0) + 1) * 0.1
        elif best_metric == 'mcc':
            best_thresholds = (np.argmax(mcc_list, axis=0) + 1) * 0.1
        elif best_metric == 'iou':
            best_thresholds = (np.argmax(iou_list, axis=0) + 1) * 0.1
        elif best_metric == 'acc':
            best_thresholds = (np.argmax(acc_list, axis=0) + 1) * 0.1
        elif best_metric == 'p':
            best_thresholds = (np.argmax(p_list, axis=0) + 1) * 0.1
        elif best_metric == 'r':
            best_thresholds = (np.argmax(r_list, axis=0) + 1) * 0.1
        else:
            print(f'Parameter best_metric = {best_metric} is illegal!\nbest_metric must be one of "f1" or "mcc" or "iou" or "acc" ')
            sys.exit(1)
        return list(auc), list(f1), list(mcc), list(iou), list(acc), list(p), list(r), list(ap), best_thresholds[0]
    else:
        print('Parameter threshold is illegal!')
        sys.exit(1)


if __name__ == '__main__':
    batch_size = 2
    logits = torch.randn(size=(batch_size, 256, 256))
    mask = torch.randint(low=0, high=2, size=(batch_size, 256, 256))

    import time
    t1 = time.time()

    metrics = get_metrics(logits, mask, threshold=0.5)
    print(metrics)
    metrics = get_metrics(logits, mask, threshold='best', best_metric='f1')
    print(metrics)

    t2 = time.time()
    print(t2 - t1, 's')
