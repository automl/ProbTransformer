import torch


class DummyHandler():
    def __init__(self,
                 data_samples,
                 max_length,
                 max_hamming=None,
                 device='cpu'):

        data_samples = [sample for sample in data_samples if sample["length"] < max_length]
        if max_hamming != None:
            data_samples = [sample for sample in data_samples if sample["hamming"] < max_hamming]

        self.samples = data_samples
        self.max_length = max_length
        self.max_hamming = max_hamming

        self.set_size = len(data_samples)

        self.device = device

        self.ignore_index = -1

    def get_sample_by_index(self, index_iter):
        for index in index_iter:
            sample = self.samples[index]

            sample = self.prepare_sample(sample, self.max_length)

            yield sample

    def batch_sort_key(self, sample):
        return sample['length']

    def pool_sort_key(self, sample):
        return sample['length']

    def prepare_sample(self, input_sample, max_length=None):

        if 'sequence' in input_sample:
            del input_sample['sequence']
        if 'hamming' in input_sample:
            del input_sample['hamming']
        input_sample['length'] = torch.tensor([input_sample['length']])

        return input_sample
