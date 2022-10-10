from typing import List, Dict, Tuple
import math
from torch.utils.data import Dataset
import re
import pandas as pd

import random
import numpy as np
import torch
from torch.nn import functional as F
from moses.utils import get_mol
from rdkit import Chem


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample_batch(model, x, block_size, temperature=1.0, sample=False, top_k=None, props=None, ignore_index=0):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    model.eval()

    steps = block_size

    if model.probabilistic:
        sample = False

    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed

        trg_len = torch.tensor([x_cond.shape[1]] * x_cond.shape[0], device=x.device)

        logits = model(trg_shf_seq=x_cond, trg_len=trg_len, props=props)  # for liggpt

        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x


def check_novelty(gen_smiles, train_smiles):  # gen: say 788, train: 120803
    if len(gen_smiles) == 0:
        novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]  # [1]*45
        novel = len(gen_smiles) - sum(duplicates)  # 788-45=743
        novel_ratio = novel * 100. / len(gen_smiles)  # 743*100/788=94.289
    print("novelty: {:.3f}%".format(novel_ratio))
    return novel_ratio


def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


class SmilesEnumerator(object):
    """SMILES Enumerator, vectorizer and devectorizer

    #Arguments
        charset: string containing the characters for the vectorization
          can also be generated via the .fit() method
        pad: Length of the vectorization
        leftpad: Add spaces to the left of the SMILES
        isomericSmiles: Generate SMILES containing information about stereogenic centers
        enum: Enumerate the SMILES during transform
        canonical: use canonical SMILES during transform (overrides enum)
    """

    def __init__(self, charset='@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True, isomericSmiles=True, enum=True,
                 canonical=False):
        self._charset = None
        self.charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    @property
    def charset(self):
        return self._charset

    @charset.setter
    def charset(self, charset):
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = dict((c, i) for i, c in enumerate(charset))
        self._int_to_char = dict((i, c) for i, c in enumerate(charset))

    def fit(self, smiles, extra_chars=[], extra_pad=5):
        """Performs extraction of the charset and length of a SMILES datasets and sets self.pad and self.charset

        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
            extra_chars: List of extra chars to add to the charset (e.g. "\\\\" when "/" is present)
            extra_pad: Extra padding to add before or after the SMILES vectorization
        """
        charset = set("".join(list(smiles)))
        self.charset = "".join(charset.union(set(extra_chars)))
        self.pad = max([len(smile) for smile in smiles]) + extra_pad

    def randomize_smiles(self, smiles):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

    def transform(self, smiles):
        """Perform an enumeration (randomization) and vectorization of a Numpy array of smiles strings
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
        """
        one_hot = np.zeros((smiles.shape[0], self.pad, self._charlen), dtype=np.int8)

        if self.leftpad:
            for i, ss in enumerate(smiles):
                if self.enumerate: ss = self.randomize_smiles(ss)
                l = len(ss)
                diff = self.pad - l
                for j, c in enumerate(ss):
                    one_hot[i, j + diff, self._char_to_int[c]] = 1
            return one_hot
        else:
            for i, ss in enumerate(smiles):
                if self.enumerate: ss = self.randomize_smiles(ss)
                for j, c in enumerate(ss):
                    one_hot[i, j, self._char_to_int[c]] = 1
            return one_hot

    def reverse_transform(self, vect):
        """ Performs a conversion of a vectorized SMILES to a smiles strings
        charset must be the same as used for vectorization.
        #Arguments
            vect: Numpy array of vectorized SMILES.
        """
        smiles = []
        for v in vect:
            v = v[v.sum(axis=1) == 1]
            # Find one hot encoded index with argmax, translate to char and join to string
            smile = "".join(self._int_to_char[i] for i in v.argmax(axis=1))
            smiles.append(smile)
        return np.array(smiles)


class SmileDataset(Dataset):

    def __init__(self, data, content, block_size, aug_prob=0, prop=None, device='cpu'):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.debug = False
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob
        self.device = device

    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]

        if self.prop:
            prop = self.prop[idx]

        smiles = smiles.strip()

        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)

        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)

        smiles += str('<') * (self.max_len - len(regex.findall(smiles)))
        if len(regex.findall(smiles)) > self.max_len:
            smiles = smiles[:self.max_len]

        smiles = regex.findall(smiles)
        dix = [self.stoi[s] for s in smiles]

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        if self.prop:
            prop = torch.tensor([prop], dtype=torch.float)
        else:
            prop = torch.tensor([0], dtype=torch.float)
        torch_length = torch.LongTensor([x.shape[0]])[0]

        return {
            "trg_shf_seq": x.to(self.device),
            "trg_seq": y.to(self.device),
            "post_seq": y.to(self.device),
            "trg_len": torch_length.to(self.device),
            "props": prop.to(self.device),
            "src_len": torch_length.to(self.device),
        }


class MolHandler():
    def __init__(
            self,
            data_dir: str = "mol_data",
            split: str = "valid",
            props: List = [],
            min_length=10,
            max_length=100,
            seed: int = 1,
            device='cpu',
    ):
        self.rng = np.random.RandomState(seed)

        self.max_length = max_length
        self.props = props
        self.split = split
        self.device = device

        data = pd.read_csv(data_dir)

        data.columns = data.columns.str.lower()
        data = data[data['smiles'].apply(lambda x: min_length <= len(x) <= max_length)]
        data = data.dropna(axis=0).reset_index(drop=True)

        self.data = data

        if split == "generate":
            set_data = data[data['source'] != 'test'].reset_index(drop=True)
        elif split == "train":
            set_data = data[data['source'] == 'train'].reset_index(drop=True)
        elif split == "valid":
            set_data = data[data['source'] == 'val'].reset_index(drop=True)
        elif split == "test":
            set_data = data[data['source'] == 'test'].reset_index(drop=True)

        smiles = set_data['smiles']

        if props:
            prop = set_data[props].values.tolist()
            self.num_props = len(props)
        else:
            prop = []
            self.num_props = False

        self.content = ' '.join(smiles)
        self.context = "C"

        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)

        lens = [len(regex.findall(i.strip())) for i in (list(smiles.values))]
        max_len = max(lens)

        whole_string = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=',
                        'B',
                        'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]',
                        '[C+]',
                        '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]',
                        '[N+]',
                        '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]',
                        '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]',
                        '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]',
                        '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o',
                        'p', 's']

        self.stoi = {ch: i for i, ch in enumerate(whole_string)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        self.dataset = SmileDataset(smiles, whole_string, max_len, prop=prop, aug_prob=0, device=self.device)

        self.set_size = set_data.shape[0]
        self.ignore_index = self.stoi['<']

        self.vocab_size = len(whole_string)
        self.source_vocab_size = len(whole_string)
        self.target_vocab_size = len(whole_string)

    def __len__(self):
        """Allows to call len(this_dataset)."""
        return self.set_size

    def __getitem__(self, idx):
        """Allows to access samples with bracket notation"""
        return self.dataset.__getitem__(idx)

    def get_sample_by_index(self, index_iter):
        for index in index_iter:
            sample = self.dataset.__getitem__(index)
            yield sample

    def batch_sort_key(self, sample):
        return sample['trg_len'].detach()

    def pool_sort_key(self, sample):
        return sample['trg_len'].item()
