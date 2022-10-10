import torch
import torch.nn as nn
import torch.nn.functional as F


class GECOLoss(nn.Module):

    def __init__(self, model, kappa, lagmul_rate, ma_decay):
        super(GECOLoss, self).__init__()

        self.register_buffer('kappa', torch.nn.Parameter(torch.FloatTensor([kappa]), requires_grad=False), persistent=True)
        self.decay = ma_decay

        self.lagmul_rate = lagmul_rate
        lagmul_init = torch.FloatTensor([1.0])
        lagmul_init = torch.log(torch.exp(torch.sqrt(lagmul_init)) - 1)  # inverse_softplus( sqrt(x) )
        lagmul = nn.Parameter(lagmul_init, requires_grad=True)
        self.lagmul = self.scale_gradients(lagmul, -lagmul_rate)
        model.register_parameter("lagmul", self.lagmul)

        self.t = 0
        self.ce_ma = 0

    @staticmethod
    def scale_gradients(v, weights):
        def hook(g):
            return g * weights
        v.register_hook(hook)
        return v

    def _moving_average(self, ce_loss):
        if self.t == 0:
            self.ce_ma = ce_loss.detach()
            self.t += 1
            return ce_loss
        else:
            self.ce_ma = self.decay * self.ce_ma + (1 - self.decay) * ce_loss.detach()
            self.t += 1
            return ce_loss + (self.ce_ma - ce_loss).detach()

    def forward(self, crit_set, trg_set):

        z_lists, ce_loss = crit_set
        trg_seq, trg_len = trg_set

        mask = torch.arange(trg_seq.size()[1], device=trg_seq.device).expand(trg_seq.size())
        mask = mask < trg_len[:, None]
        mask = mask.type(trg_seq.type())

        if z_lists[0][0].mean.shape[1] != trg_seq.size()[1]:  # correct in case of props or scaffold
            large_mask = torch.zeros(z_lists[0][0].mean.shape[:2], device=trg_seq.device).type(trg_seq.type())
            large_mask[:, -trg_seq.size()[1]:] = mask
            mask = large_mask

        z_list, p_z_list = z_lists

        kl_list = []
        mean_list = []
        mean_max_list = []
        stddev_list = []
        stddev_max_list = []

        for idx, (z, p_z) in enumerate(zip(z_list, p_z_list)):
            mean_list.append(torch.mean(z.mean).detach())
            mean_max_list.append(torch.max(torch.abs(z.mean)).detach())
            stddev_list.append(torch.mean(z.stddev).detach())
            stddev_max_list.append(torch.max(torch.abs(z.stddev)).detach())

            kl_dist = torch.distributions.kl_divergence(p_z, z)

            kl_dist = kl_dist.sum(-1)
            kl_dist = kl_dist * mask
            kl_dist = torch.sum(kl_dist, dim=-1) / trg_len
            kl_dist = kl_dist.mean()

            kl_list.append(kl_dist)

        kl_loss = torch.stack(kl_list, dim=-1).sum()
        ma_ce_loss = self._moving_average(ce_loss)
        rec_constraint = ma_ce_loss - self.kappa

        lamb = F.softplus(self.lagmul) ** 2

        loss = lamb * rec_constraint + kl_loss

        summary = {"mean_list": [k.detach() for k in mean_list],
                   "stddev_list": [k.detach() for k in stddev_list],
                   "mean_max_list": [k.detach() for k in mean_max_list],
                   "stddev_max_list": [k.detach() for k in stddev_max_list],
                   "kl_loss": kl_loss.detach(),
                   "ce_loss": ce_loss.detach(),
                   "lagmul": self.lagmul.detach(),
                   "ma_ce_loss": ma_ce_loss.detach(),
                   "rec_constraint": rec_constraint.detach(),
                   "lamb": lamb.detach(), "t": self.t, }

        return loss, summary
