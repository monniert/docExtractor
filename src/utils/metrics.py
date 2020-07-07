from collections import OrderedDict
import numpy as np


class RunningMetrics:
    """
    Compute the following metrics:
        - overall accuracy (weighted mean accuracy)
        - average accuracy
        - weighted mean IoU
        - average IoU
        - IoU per class
    """
    def __init__(self, model_labels, metric_labels=None):
        self.labels = [0] + model_labels
        self.metric_labels = self.labels if metric_labels is None else [l for l in self.labels if l in metric_labels]
        if 0 not in self.metric_labels:
            self.metric_labels = [0] + self.metric_labels
        self.metric_labels_idx = [self.labels.index(l) for l in self.metric_labels]
        self.n_classes = len(self.metric_labels)
        self.names = ["overall_acc", "avg_acc", "weighted_mean_iou", "avg_iou"] + \
            ["iou_class_{}".format(l) for l in self.metric_labels]
        self.score_name = "avg_iou"
        self.reset()

    def get(self):
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        avg_acc = np.diag(hist) / hist.sum(axis=1)
        avg_acc = np.mean(np.nan_to_num(avg_acc))
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        avg_iu = np.nanmean(np.nan_to_num(iu))
        freq = hist.sum(axis=1) / hist.sum()
        w_mean_iu = (freq[freq > 0] * iu[freq > 0]).sum()

        metrics = [acc, avg_acc, w_mean_iu, avg_iu] + list(iu)
        return OrderedDict(zip(self.names, metrics))

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            lt_flat, lp_flat = lt.flatten(), lp.flatten()

            if len(self.metric_labels) < len(self.labels):
                for idx in range(1, len(self.labels)):
                    new_idx = self.metric_labels_idx.index(idx) if idx in self.metric_labels_idx else 0
                    lt_flat[lt_flat == idx] = new_idx
                    lp_flat[lp_flat == idx] = new_idx

            self.confusion_matrix += self._fast_hist(lt_flat, lp_flat)

    def _fast_hist(self, label_true, label_pred):
        hist = np.bincount(self.n_classes * label_true + label_pred,
                           minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes)
        return hist


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
