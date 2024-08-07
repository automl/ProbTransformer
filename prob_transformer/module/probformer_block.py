import torch.nn as nn

from RnaBench.lib.rna_folding_algorithms.DL.ProbTransformer.prob_transformer.module.feed_forward import FeedForward
from RnaBench.lib.rna_folding_algorithms.DL.ProbTransformer.prob_transformer.module.probabilistic_forward import ProbabilisticForward
from RnaBench.lib.rna_folding_algorithms.DL.ProbTransformer.prob_transformer.module.attention import PreNormAttention


class ProbFormerBlock(nn.Module):

    def __init__(self, model_dim, num_head, ff_factor, z_factor, dropout, zero_init, cross_attention,
                 probabilistic, last_layer):
        super().__init__()

        ff_dim = int(ff_factor * model_dim)
        z_dim = int(model_dim * z_factor)

        self.cross_attention = cross_attention
        self.probabilistic = probabilistic

        self.dropout = nn.Dropout(p=dropout)
        self.self_attn = PreNormAttention(model_dim, num_head, encoder=False, zero_init=zero_init)

        if cross_attention:
            self.coder_attn = PreNormAttention(model_dim, num_head, encoder=True, zero_init=zero_init)

        if probabilistic:
            self.prob_layer = ProbabilisticForward(model_dim, z_dim,
                                                   last_layer=last_layer,
                                                   softplus=False, zero_init=zero_init)
            self.transition = FeedForward(model_dim, ff_dim, zero_init)
        else:
            self.transition = FeedForward(model_dim, ff_dim, zero_init)

    def forward(self, src_act, src_mask, enc_act=None, enc_mask=None, p_z=None, infer_mean=False):

        src_act = src_act + self.dropout(self.self_attn(src_act, enc_act=None, mask=src_mask))

        if self.cross_attention:
            src_act = src_act + self.dropout(self.coder_attn(src_act, enc_act=enc_act, mask=enc_mask))

        if self.probabilistic:
            src_act = src_act + self.dropout(self.transition(src_act))
            act_z, z = self.prob_layer(src_act, p_z, infer_mean)
            src_act = src_act + act_z
            return src_act, z
        else:
            src_act = src_act + self.dropout(self.transition(src_act))
            return src_act
