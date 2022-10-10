import numpy as np
from collections import defaultdict
from itertools import zip_longest, product


def symmetric_matrix(mat):
    return np.maximum(mat, mat.T)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class StatisticsCenter():
    def __init__(self,
                 pred_list,
                 step_size=0.01,
                 symmetric_matrices=True,
                 full_eval=False,
                 triangle_loss=False,
                 ):
        self.symmetric_matrices = symmetric_matrices
        self.full_eval = full_eval
        self.triangle_loss = triangle_loss

        self.types2evaluate = ['wc', 'wobble', 'nc', 'canonical', 'all_pairs']
        self.wc_pairs = ['GC', 'CG', 'AU', 'UA']
        self.wobble_pairs = ['GU', 'UG']

        self.pred_list = pred_list

        self.metrics = defaultdict(list)
        self.predictions = defaultdict(list)

        self.thresholds2evaluate = np.arange(0.0, 1.0, step_size)
        self.evaluated_thresholds = []

    def eval_prediction(self, true_mat, pred_mat, mask_dict, threshold):
        metrics = {}

        if self.triangle_loss:
            idx1, idx2 = np.triu_indices(pred_mat.shape[1], 1)

            pred_pair_sec1 = pred_mat[idx1, idx2]
            pred_pair_sec2 = pred_mat.transpose(1, 0)[idx1, idx2]
            pred_pair_sec = (pred_pair_sec1 + pred_pair_sec2) / 2
            pred_pair_sec = sigmoid(pred_pair_sec)
            pred_pair_sec = pred_pair_sec > threshold

            pred = np.zeros(pred_mat.shape)
            pred[idx1, idx2] = pred_pair_sec
            if self.symmetric_matrices:
                pred[idx2, idx1] = pred_pair_sec
        else:

            pred_mat = sigmoid(pred_mat)
            pred = pred_mat > threshold
            if self.symmetric_matrices:
                pred = symmetric_matrix(pred)

        pred = pred.flatten().astype(int)
        true = true_mat.flatten()

        for key, mask_mat in mask_dict.items():

            if key != 'all':
                mask = mask_mat.flatten()
                del_pos = np.where(mask == 0)[0]

                pred_del = np.delete(pred, del_pos)
                true_del = np.delete(true, del_pos)

                k_metrics = self.eval_array(pred_del, true_del)
            else:
                k_metrics = self.eval_array(pred, true)
            k_metrics = {f"{key}_{k}": v for k, v in k_metrics.items()}
            metrics.update(k_metrics)
        return metrics

    def eval_array(self, pred, true):

        solved = np.all(np.equal(true, pred)).astype(int)
        if solved == 1:
            f1_score = 1
            non_correct = 0
            precision = 1
            recall = 1
            specificity = 1
        else:
            tp = np.logical_and(pred, true).sum()
            non_correct = (tp == 0).astype(int)
            tn = np.logical_and(np.logical_not(pred), np.logical_not(true)).sum()
            fp = pred.sum() - tp
            fn = true.sum() - tp

            recall = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            f1_score = 2 * tp / (2 * tp + fp + fn)

        metrics = {'f1_score': f1_score, 'solved': solved}
        if self.full_eval:
            metrics['non_correct'] = non_correct
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['specificity'] = specificity

        return metrics

    def get_pair_type_masks(self, sequence):
        all_mask = np.ones((len(sequence), len(sequence)))
        wc = np.zeros((len(sequence), len(sequence)))
        wobble = np.zeros((len(sequence), len(sequence)))

        a = [i for i, sym in enumerate(sequence) if sym.upper() == 'A']
        c = [i for i, sym in enumerate(sequence) if sym.upper() == 'C']
        g = [i for i, sym in enumerate(sequence) if sym.upper() == 'G']
        u = [i for i, sym in enumerate(sequence) if sym.upper() == 'U']

        for wc1, wc2, wob in zip_longest(product(g, c), product(a, u), product(g, u), fillvalue=None):
            if wc1:
                wc[wc1[0], wc1[1]] = 1
                wc[wc1[1], wc1[0]] = 1
            if wc2:
                wc[wc2[0], wc2[1]] = 1
                wc[wc2[1], wc2[0]] = 1
            if wob:
                wobble[wob[0], wob[1]] = 1
                wobble[wob[1], wob[0]] = 1

        canonical = wc + wobble
        nc = all_mask - canonical

        return {'all': all_mask, 'wc': wc, 'canonical': canonical, 'wobble': wobble, 'nc': nc}

    def eval_pred(self, pred_sample, threshold):

        pred_mat = pred_sample['pred']
        true_mat = pred_sample['true'].astype(int)

        mask_dict = self.get_pair_type_masks(pred_sample['sequence'])
        metrics = self.eval_prediction(true_mat, pred_mat, mask_dict, threshold)

        return metrics

    def eval_threshold(self, threshold):

        assert 0.0 <= threshold <= 1.0

        metrics_list = list(map(lambda x: self.eval_pred(x, threshold), self.pred_list))
        metrics = {k: np.mean([dic[k] for dic in metrics_list]) for k in metrics_list[0]}

        return metrics

    def find_best_threshold(self):

        best_threshold = 0
        best_all_f1_score = 0
        best_metric = {}

        for threshold in self.thresholds2evaluate:
            metrics = self.eval_threshold(threshold)

            if metrics['all_f1_score'] > best_all_f1_score:
                best_all_f1_score = metrics['all_f1_score']
                best_metric = metrics
                best_threshold = threshold

        return best_metric, best_threshold
