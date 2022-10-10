from typing import List, Dict, Tuple
import numpy as np
import torch


class SummaryDict():
    """
    Similar to TensorFlow summary but can deal with lists, stores everything in numpy arrays. Please see main for usage.
    """

    def __init__(self, summary=None):
        self.summary = {}
        if summary is not None:
            for key, value in summary.items():
                self.summary[key] = value

    @property
    def keys(self):
        keys = list(self.summary.keys())
        if "step" in keys:
            keys.remove("step")
        return keys

    def __call__(self, summary):

        if isinstance(summary, SummaryDict):
            for key, value_lists in summary.summary.items():
                if key in self.summary.keys():
                    if key == "step":
                        if min(value_lists) != 1 + max(self.summary['step']):
                            value_lists = np.asarray(value_lists) + max(self.summary['step']) + 1 - min(value_lists)
                    self.summary[key] = np.concatenate([self.summary[key], value_lists], axis=0)
                else:
                    self.summary[key] = value_lists

        elif isinstance(summary, Dict):
            for name, value in summary.items():
                self.__setitem__(name, value)
        elif isinstance(summary, (Tuple, List)):
            for l in summary:
                self.__call__(l)
        else:
            raise UserWarning(f"SummaryDict: call not implementet for type: {type(summary)}")

    def __setitem__(self, key, item):

        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        if isinstance(item, List):
            if isinstance(item[0], torch.Tensor):
                item = [v.cpu().numpy() for v in item]
        if isinstance(item, np.ndarray):
            item = np.squeeze(item)

        item = np.expand_dims(np.asarray(item), axis=0)
        if item.shape.__len__() < 2:
            item = np.expand_dims(item, axis=0)

        if key not in self.summary.keys():
            self.summary[key] = item
        else:
            self.summary[key] = np.concatenate([self.summary[key], item], axis=0)

    def __getitem__(self, key):
        return self.summary[key]

    def save(self, file):
        np.save(file, self.summary)

    def load(self, file):
        return np.load(file).tolist()
