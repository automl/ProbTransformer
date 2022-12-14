import string, os
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import torch


class SSDHandler():

    def __init__(self, max_len, min_len, sample_amount, trg_vocab_size, src_vocab_size, sentence_len, n_sentence,
                 sentence_variations,
                 seed=123, device='cpu', pre_src_vocab=None, pre_trg_vocab=None, token_dict=None):

        super().__init__()
        self.rng = np.random.RandomState(seed=seed)

        self.set_size = sample_amount

        self.blank_word = '<blank>'

        self.sentence_len = sentence_len
        self.sentence_variations = sentence_variations
        self.min_len = min_len
        self.max_len = max_len

        self.device = device

        if pre_src_vocab == None:
            self.pre_src_vocab = self._make_vocab(src_vocab_size)
            self.pre_trg_vocab = self._make_vocab(trg_vocab_size, numbers=True)
            self.token_dict = self._make_token_dict(sentence_len, n_sentence, sentence_variations)
        else:
            self.pre_src_vocab = pre_src_vocab
            self.pre_trg_vocab = pre_trg_vocab
            self.token_dict = token_dict

        self.source_stoi = dict(zip(self.pre_src_vocab, range(len(self.pre_src_vocab))))
        self.source_itos = dict((y, x) for x, y in self.source_stoi.items())

        self.target_itos = dict(zip(range(len(self.pre_trg_vocab)), self.pre_trg_vocab))
        self.target_stoi = dict((y, x) for x, y in self.target_itos.items())

        self.source_vocab_size = len(self.source_stoi)
        self.target_vocab_size = len(self.target_stoi)

        os.makedirs("cache", exist_ok=True)
        file_name = f"cache/{min_len}_{max_len}_{sentence_variations}_{sentence_len}_{sample_amount}_{seed}.tlist"

        if not os.path.isfile(file_name):
            self.sample_list = self._generate_data(size=sample_amount)
            torch.save(self.sample_list, file_name)
        else:
            self.sample_list = torch.load(file_name)

        self.trg_vocab = SimpleNamespace(**{"id_to_token": lambda i: self.target_itos[i],
                                            "token_to_id": lambda i: self.target_stoi[i],
                                            "size": self.target_vocab_size})

    def get_sample_by_index(self, index_iter):
        for index in index_iter:
            sample = self.sample_list[index]

            yield sample

    def batch_sort_key(self, sample):
        return sample['src_len'].detach().tolist()

    def pool_sort_key(self, sample):
        return sample['src_len'].item()

    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _make_vocab(self, vocab_size, numbers=False):
        vocab = []
        word_size = 5

        while len(vocab) < vocab_size:
            if numbers:
                word = "".join([str(s) for s in self.rng.randint(0, 10, word_size)])
            else:
                word = "".join(self.rng.choice([s for s in string.ascii_letters], word_size, replace=True))
            if word not in vocab:
                vocab.append(word)

        if numbers:
            pass
        else:
            vocab.append(self.blank_word)

        return vocab

    def _make_token_dict(self, sentence_len, n_sentence, sentence_variations):

        token_dict = OrderedDict()
        for _ in range(int(min([n_sentence, len(self.pre_src_vocab) ** sentence_len,
                                len(self.pre_trg_vocab) ** sentence_len]))):

            # create new src token
            src_list = self.rng.choice(self.pre_src_vocab, sentence_len, replace=True).tolist()
            src_token = "-".join(src_list)
            while src_token in token_dict.keys():
                src_list = self.rng.choice(self.pre_src_vocab, sentence_len, replace=True).tolist()
                src_token = "-".join(src_list)

            # create trg
            trg_options_list = []
            trg_dist_list = []

            sample_sentence_len = sentence_len
            trg_choice = self.pre_trg_vocab.__len__()

            for sub in range(sample_sentence_len):
                rand_options = self.rng.randint(1, 1 + sentence_variations)
                trg_options = self.rng.choice(trg_choice, rand_options, replace=False).tolist()
                trg_dist = self._softmax(self.rng.uniform(0, 2, rand_options))
                trg_options_list.append(trg_options)
                trg_dist_list.append(trg_dist)

            token_dict[src_token] = {"src_list": src_list, "trg_options_list": trg_options_list,
                                     "trg_dist_list": trg_dist_list}
        return token_dict

    def _generate_data(self, size):

        data_set = []
        for idx in range(int(size)):
            if self.min_len == self.max_len:
                length = self.min_len
            else:
                length = self.rng.randint(self.min_len, self.max_len + 1)
                length = (length // self.sentence_len) * self.sentence_len
            source, target = self.make_sample(length)

            if self.device == 'cpu':
                torch_length = torch.LongTensor([length])[0]
            else:
                torch_length = torch.cuda.LongTensor([length], device=self.device)[0]

            torch_sample = {}
            src_seq = self.sequence2index_matrix(source, self.source_stoi)
            torch_sample['src_seq'] = src_seq
            torch_sample['src_len'] = torch_length

            trg_seq = self.sequence2index_matrix(target, self.target_stoi)

            torch_sample['trg_seq'] = trg_seq
            torch_sample['trg_len'] = torch_length
            torch_sample['post_seq'] = trg_seq

            data_set.append(torch_sample)

        return data_set

    def sequence2index_matrix(self, sequence, mapping):

        int_sequence = list(map(mapping.get, sequence))

        if self.device == 'cpu':
            tensor = torch.LongTensor(int_sequence)
        else:
            tensor = torch.cuda.LongTensor(int_sequence, device=self.device)
        return tensor

    def make_sample(self, n_steps):

        n_sub_token = int(n_steps / self.sentence_len)
        source, target = [], []

        for _ in range(n_sub_token):
            src_sub_token = self.rng.choice(list(self.token_dict.keys()))

            sub_token = self.token_dict[src_sub_token]
            source.extend(sub_token['src_list'])

            for trg_options, trg_dist in zip(sub_token['trg_options_list'], sub_token['trg_dist_list']):
                trg_idx = self.rng.choice(trg_options, 1, p=trg_dist)[0]
                trg_symbol = self.pre_trg_vocab[trg_idx]
                target.append(trg_symbol)

        return source, target

    def get_valid_data(self):

        source = []
        for sub_token in self.token_dict.values():
            source.extend(sub_token['src_list'])

        target_dist = self.get_sample_dist(source)

        return source, target_dist

    def get_sample_dist(self, src):

        if isinstance(src[0], int) or isinstance(src[0], np.int32) or isinstance(src[0], np.int64):
            src = [self.source_itos[s] for s in src]

        elif isinstance(src[0], torch.Tensor):
            src = src.detach().cpu().tolist()
            src = [self.source_itos[s] for s in src]
        else:
            raise UserWarning(f"unknown source type: {type(src[0])}")

        target_dist = []
        for idx in range(len(src) // self.sentence_len):
            sub = src[idx * self.sentence_len: idx * self.sentence_len + self.sentence_len]
            sub_name = '-'.join(sub)
            if sub_name != '-'.join([self.blank_word] * self.sentence_len):
                sub_token = self.token_dict[sub_name]
                for tidx, (trg_options, trg_dist) in enumerate(
                        zip(sub_token['trg_options_list'], sub_token['trg_dist_list'])):
                    t_dist = np.zeros([self.target_vocab_size])
                    for trg_pos, t_prob in zip(trg_options, trg_dist):
                        trg_symb = self.pre_trg_vocab[trg_pos]
                        trg_pos = self.target_stoi[trg_symb]
                        t_dist[trg_pos] = t_prob
                    target_dist.append(t_dist)
        target_dist = np.stack(target_dist, axis=0)

        return target_dist

    def get_batch_dist(self, src_batch):
        trg_dist_list = []
        for b in range(src_batch.size()[0]):
            src = src_batch[b, :]
            trg_dist = self.get_sample_dist(src)
            trg_dist_list.append(trg_dist)

        trg_dist_len = [d.shape[0] for d in trg_dist_list]
        if np.unique(trg_dist_len).shape[0] == 1:
            trg_dist = np.stack(trg_dist_list, axis=0)
        else:
            trg_dist = np.zeros([len(trg_dist_list), max(trg_dist_len), self.target_vocab_size])
            for b, trg_d in enumerate(trg_dist_list):
                trg_dist[b, :trg_d.shape[0], :] = trg_d

        trg_dist = torch.FloatTensor(trg_dist).to(src_batch.device)
        return trg_dist
