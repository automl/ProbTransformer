import torch
import torch.nn as nn

from RnaBench.lib.rna_folding_algorithms.DL.ProbTransformer.prob_transformer.module.probformer_block import ProbFormerBlock


class ProbFormerStack(nn.Module):

    def __init__(self, n_layers, model_dim, num_head, ff_factor, z_factor, dropout, zero_init,
                 cross_attention, posterior, prob_layer):
        """Builds Attention module.
        """
        super().__init__()

        self.posterior = posterior
        self.prob_layer = prob_layer

        module_list = []
        for idx in range(n_layers):
            last_layer = posterior and idx == max(prob_layer)
            layer = ProbFormerBlock(model_dim=model_dim, num_head=num_head, ff_factor=ff_factor, z_factor=z_factor,
                                    dropout=dropout, zero_init=zero_init,
                                    cross_attention=cross_attention, probabilistic=idx in prob_layer,
                                    last_layer=last_layer)
            module_list.append(layer)
        self.layers = nn.ModuleList(module_list)

    def forward(self, src_act, src_mask, enc_act=None, enc_mask=None, p_z_list=None, infer_mean=False):

        z_list = []
        p_z_index = 0
        mask_list = []

        for idx, layer in enumerate(self.layers):
            if idx in self.prob_layer:
                if p_z_list is not None:
                    src_act_new, z = layer(src_act, src_mask, enc_act, enc_mask,
                                           p_z=p_z_list[p_z_index],
                                           infer_mean=False)
                    p_z_index = p_z_index + 1
                else:

                    src_act_new, z = layer(src_act, src_mask, enc_act, enc_mask, infer_mean=infer_mean)
                z_list.append(z)
                mask_list.append(src_mask[:, 0, :].detach())
            else:
                src_act_new = layer(src_act, src_mask, enc_act, enc_mask)
            src_act = src_act + src_act_new

        if len(self.prob_layer) > 0:
            return src_act, z_list, torch.stack(mask_list, dim=0)
        else:
            return src_act
