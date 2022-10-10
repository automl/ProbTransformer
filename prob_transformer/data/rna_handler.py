import collections
import numpy as np
import torch
import pandas as pd

from pathlib import Path


class RNAHandler():
    def __init__(self,
                 df_path,
                 sub_set,
                 ignore_index,
                 seed,
                 min_length,
                 max_length,
                 similarity=80,
                 device='cpu',
                 ):

        assert sub_set in ['train', 'valid', 'test']

        df_path = Path(df_path)

        if not df_path.is_file():
            raise UserWarning(f"no dataframe found on: {df_path.resolve().__str__()}")

        df = pd.read_pickle(df_path)

        df = df[df[f"non_sim_{similarity}"]]
        df = df[df['set'].str.contains(sub_set)]

        df = df[df['structure'].apply(set).apply(len) > 1]  # remove only '.' samples, should be removed already

        self.max_length = max_length

        df = df[df['sequence'].apply(lambda x: min_length <= len(x) <= max_length)]

        df = df.reset_index()

        self.datasettoint = {k: v for k, v in zip(df['dataset'].unique(), range(len(df['dataset'].unique())))}
        self.inttodataset = {v: k for k, v in self.datasettoint.items()}

        self.df = df

        self.set_size = self.df.shape[0]

        self.rng = np.random.default_rng(seed=seed)
        self.device = device

        self.ignore_index = ignore_index

        self.seq_vocab = ['A', 'C', 'G', 'U', 'N']
        self.canonical_pairs = ['GC', 'CG', 'AU', 'UA', 'GU', 'UG']

        nucs = {
            'T': 'U',
            'P': 'U',
            'R': 'A',  # or 'G'
            'Y': 'C',  # or 'T'
            'M': 'C',  # or 'A'
            'K': 'U',  # or 'G'
            'S': 'C',  # or 'G'
            'W': 'U',  # or 'A'
            'H': 'C',  # or 'A' or 'U'
            'B': 'U',  # or 'G' or 'C'
            'V': 'C',  # or 'G' or 'A'
            'D': 'A',  # or 'G' or 'U'
        }

        self.struct_vocab = ['.', '(0c', ')0c', '(1c', ')1c', '(2c', ')2c', '(0nc', ')0nc', '(1nc', ')1nc', '(2nc',
                             ')2nc']

        self.seq_stoi = dict(zip(self.seq_vocab, range(len(self.seq_vocab))))
        self.seq_itos = dict((y, x) for x, y in self.seq_stoi.items())

        for nuc, mapping in nucs.items():
            self.seq_stoi[nuc] = self.seq_stoi[mapping]

        self.struct_itos = dict(zip(range(len(self.struct_vocab)), self.struct_vocab))
        self.struct_stoi = dict((y, x) for x, y in self.struct_itos.items())

        self.seq_vocab_size = len(self.seq_vocab)
        self.struct_vocab_size = len(self.struct_vocab)

    def get_sample_by_index(self, index_iter):
        for index in index_iter:
            sample = self.df.iloc[index]

            sample = self.prepare_sample(sample, self.max_length)

            yield sample

    def batch_sort_key(self, sample):
        return sample['src_len'].detach().tolist()

    def pool_sort_key(self, sample):
        return sample['src_len'].item()

    def sequence2index_matrix(self, sequence, mapping):

        int_sequence = list(map(mapping.get, sequence))

        if self.device == 'cpu':
            tensor = torch.LongTensor(int_sequence)
        else:
            tensor = torch.cuda.LongTensor(int_sequence, device=self.device)
        return tensor

    def prepare_sample(self, input_sample, max_length=None):

        sequence = input_sample["sequence"]
        structure = input_sample["structure"]
        pos1id = input_sample["pos1id"]
        pos2id = input_sample["pos2id"]
        pdb_sample = int(input_sample['is_pdb'])
        dataset = self.datasettoint[input_sample['dataset']]

        length = len(sequence)

        with torch.no_grad():
            pair_m, pair_mat = self.get_pair_matrices(pos1id, pos2id, length, pdb_sample)

            target_structure = self.encode_target_structure(pair_mat, structure, sequence)

            src_seq = self.sequence2index_matrix(sequence, self.seq_stoi)
            trg_seq = self.sequence2index_matrix(target_structure, self.struct_stoi)

            post_seq = trg_seq.clone()

            post_pair_m = pair_m.clone()
            post_pair_mat = pair_mat.clone()

            trg_pair_m = pair_m.clone()
            trg_pair_mat = pair_mat.clone()

            if pdb_sample == 0:
                trg_pair_m.fill_(self.ignore_index)

            if self.device == 'cpu':
                torch_length = torch.LongTensor([length])[0]
                torch_pdb_sample = torch.LongTensor([pdb_sample])[0]
                torch_dataset = torch.LongTensor([dataset])[0]
            else:
                torch_length = torch.cuda.LongTensor([length], device=self.device)[0]
                torch_pdb_sample = torch.cuda.LongTensor([pdb_sample], device=self.device)[0]
                torch_dataset = torch.cuda.LongTensor([dataset], device=self.device)[0]

        torch_sample = {}

        torch_sample['src_seq'] = src_seq

        torch_sample['src_len'] = torch_length
        torch_sample['trg_len'] = torch_length
        torch_sample['pdb_sample'] = torch_pdb_sample
        torch_sample['dataset'] = torch_dataset

        torch_sample['trg_seq'] = trg_seq
        torch_sample['trg_pair_m'] = trg_pair_m
        torch_sample['trg_pair_mat'] = trg_pair_mat

        torch_sample['post_seq'] = post_seq
        torch_sample['post_pair_m'] = post_pair_m
        torch_sample['post_pair_mat'] = post_pair_mat

        return torch_sample

    def encode_target_structure(self, pair_mat, raw_structure, sequence):

        pos1, pos2 = torch.where(pair_mat == 1)
        pos = torch.concat([pos1, pos2]).unique().cpu().numpy()

        pos_dict = {i1.item(): i2.item() for i1, i2 in zip(pos1, pos2)}

        structure = []
        for s_idx, s in enumerate(raw_structure):
            if len(s) > 1:
                if s[1] != '0' and s[1] != '1':
                    s = s[0] + '2'

            if s != '.':
                if s[0] == "(":
                    counter_idx = pos_dict[s_idx]
                    assert raw_structure[counter_idx][0] == ')'
                    pair = sequence[s_idx] + sequence[counter_idx]
                    if pair in self.canonical_pairs:
                        s = s + "c"
                    else:
                        s = s + "nc"
                elif s[0] == ")":
                    counter_idx = pos_dict[s_idx]
                    assert raw_structure[counter_idx][0] == '('
                    pair = sequence[counter_idx] + sequence[s_idx]
                    if pair in self.canonical_pairs:
                        s = s + "c"
                    else:
                        s = s + "nc"
                else:
                    raise UserWarning("unknown ()")
            structure.append(s)
        structure = np.asarray(structure)

        if "." in structure[pos]:
            print("DEBUG")

        assert "." not in structure[pos]

        return structure.tolist()

    def get_pair_matrices(self, pos1id, pos2id, length, pdb_sample):

        assert len(pos1id) == len(pos2id)

        if self.device == 'cpu':
            multi_mat = torch.LongTensor(length, length).fill_(0)
            pair_mat = torch.LongTensor(length, length).fill_(0)
        else:
            multi_mat = torch.cuda.LongTensor(length, length, device=self.device).fill_(0)
            pair_mat = torch.cuda.LongTensor(length, length, device=self.device).fill_(0)

        if pdb_sample == 1:
            pos_count = collections.Counter(pos1id + pos2id)
            multiplets = [pos for pos, count in pos_count.items() if count > 1]
        else:
            multiplets = []

        for p1, p2 in zip(pos1id, pos2id):

            pair_mat[p1, p2] = 1
            pair_mat[p2, p1] = 1

            if len(multiplets) > 0:
                if p1 in multiplets or p2 in multiplets:
                    multi_mat[p1, p2] = 1
                    multi_mat[p2, p1] = 1

        return multi_mat, pair_mat
