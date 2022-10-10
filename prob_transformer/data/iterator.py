from typing import List, Dict, Tuple
from types import SimpleNamespace
import numpy as np
import torch


class MyIterator():
    def __init__(self,
                 data_handler,
                 batch_size,
                 pool_size=20,
                 pre_sort_samples=False,
                 device='cpu',
                 repeat=False,
                 shuffle=True,
                 batching=True,
                 seed=1,
                 ignore_index=-1,
                 pad_index=0,
                 ):

        self.repeat = repeat
        self.shuffle = shuffle
        self.batching = batching

        assert callable(getattr(data_handler, "batch_sort_key", None)), "data handler has no 'batch_sort_key' method"
        assert callable(getattr(data_handler, "pool_sort_key", None)), "data handler has no 'pool_sort_key' method"
        assert callable(
            getattr(data_handler, "get_sample_by_index", None)), "data handler has no 'get_sample_by_index' method"
        assert hasattr(data_handler, "set_size"), "data handler has no 'set_size' attribute"
        self.data_handler = data_handler

        self.batch_size = batch_size  # batchsize in cumulated sequence length in batch
        self.pool_size = pool_size
        self.pre_sort_samples = pre_sort_samples
        self.device = device

        self.ignore_index = ignore_index
        self.pad_index = pad_index

        self.rng = np.random.default_rng(seed=seed)

        self.set_size = self.data_handler.set_size

    def get_index_list(self):
        index_list = [i for i in range(self.set_size)]
        if self.shuffle:
            self.rng.shuffle(index_list)
        return index_list

    def get_index_iter(self):
        while True:
            index_list = self.get_index_list()
            for i in index_list:
                yield i

    def cluster_index_iter(self, cluster_index_list):
        while True:
            for i in cluster_index_list:
                yield i

    def pool_and_sort(self, sample_iter):

        pool = []

        for sample in sample_iter:
            if not self.pre_sort_samples:
                yield sample
            else:
                pool.append(sample)
                if len(pool) >= self.pool_size:
                    pool.sort(key=self.data_handler.pool_sort_key)

                    while len(pool) > 0:
                        yield pool.pop()
        if len(pool) > 0:
            pool.sort(key=self.data_handler.pool_sort_key)
            while len(pool) > 0:
                yield pool.pop()

    def __iter__(self):

        minibatch, max_size_in_batch = [], 0

        while True:

            if self.repeat:
                index_iter = self.get_index_iter()
            else:
                index_iter = self.get_index_list()

            for sample in self.pool_and_sort(self.data_handler.get_sample_by_index(index_iter)):

                if self.batching:
                    minibatch.append(sample)
                    max_size_in_batch = max(max_size_in_batch, self.data_handler.batch_sort_key(sample))
                    size_so_far = len(minibatch) * max(max_size_in_batch, self.data_handler.batch_sort_key(sample))
                    if size_so_far == self.batch_size:
                        yield self.batch_samples(minibatch)
                        minibatch, max_size_in_batch = [], 0
                    if size_so_far > self.batch_size:
                        yield self.batch_samples(minibatch[:-1])
                        minibatch = minibatch[-1:]
                        max_size_in_batch = self.data_handler.batch_sort_key(minibatch[0])
                else:
                    yield self.batch_samples([sample])

            if not self.repeat:
                if self.batching and len(minibatch) > 0:
                    yield self.batch_samples(minibatch)
                return

    def batch_samples(self, sample_dict_minibatch: List[Dict]):

        with torch.no_grad():
            batch_dict = {k: [dic[k] for dic in sample_dict_minibatch] for k in sample_dict_minibatch[0]}

            for key, tensor_list in batch_dict.items():

                max_shape = [list(i.shape) for i in tensor_list]
                if len(tensor_list[0].shape) == 0:
                    max_shape = [len(tensor_list)]
                else:
                    max_shape = [len(tensor_list)] + [max([s[l] for s in max_shape]) for l in range(len(max_shape[0]))]

                if tensor_list[0].dtype == torch.float64 or tensor_list[0].dtype == torch.float32 or tensor_list[
                    0].dtype == torch.float16:
                    max_tensor = torch.zeros(size=max_shape, dtype=tensor_list[0].dtype, device=self.device)

                elif tensor_list[0].dtype == torch.int64 or tensor_list[0].dtype == torch.int32 or tensor_list[
                    0].dtype == torch.int16:
                    if "trg_seq" == key or "trg_msa" == key:
                        max_tensor = torch.ones(size=max_shape, dtype=tensor_list[0].dtype,
                                                device=self.device) * self.ignore_index
                    else:
                        max_tensor = torch.ones(size=max_shape, dtype=tensor_list[0].dtype,
                                                device=self.device) * self.pad_index
                else:
                    raise UserWarning(f"key {key} has an unsupported dtype: {tensor_list[0].dtype}")

                for b, tensor in enumerate(tensor_list):
                    ts = tensor.shape
                    if len(tensor.shape) == 0:
                        max_tensor[b] = tensor.to(self.device)
                    elif len(tensor.shape) == 1:
                        max_tensor[b, :ts[0]] = tensor.to(self.device)
                    elif len(tensor.shape) == 2:
                        max_tensor[b, :ts[0], :ts[1]] = tensor.to(self.device)
                    elif len(tensor.shape) == 3:
                        max_tensor[b, :ts[0], :ts[1], :ts[2]] = tensor.to(self.device)
                    elif len(tensor.shape) == 4:
                        max_tensor[b, :ts[0], :ts[1], :ts[2], :ts[3]] = tensor.to(self.device)
                    else:
                        raise UserWarning(f"key {key} has an unsupported dimension: {tensor_list[0].shape}")

                batch_dict[key] = max_tensor

        batch = SimpleNamespace(**batch_dict)

        return batch
