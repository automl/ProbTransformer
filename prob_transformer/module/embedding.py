import torch
import torch.nn as nn
import torch.nn.functional as F


class PosEmbedding(nn.Module):
    def __init__(self, vocab, model_dim, max_len):
        super().__init__()
        self.max_len = max_len
        self.embed_seq = nn.Embedding(vocab, model_dim)
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([model_dim // 2])), requires_grad=False)
        self.embed_pair_pos = nn.Linear(max_len + 1, model_dim)

    def relative_position_encoding(self, src_seq):
        residue_index = torch.arange(src_seq.size()[1], device=src_seq.device).expand(src_seq.size())
        rel_pos = F.one_hot(torch.clip(residue_index, min=0, max=self.max_len), self.max_len + 1).type(
            torch.float32).to(src_seq.device)
        pos_encoding = self.embed_pair_pos(rel_pos)
        return pos_encoding

    def forward(self, src_seq):
        seq_embed = self.embed_seq(src_seq) * self.scale
        seq_embed = seq_embed + self.relative_position_encoding(src_seq)
        return seq_embed
