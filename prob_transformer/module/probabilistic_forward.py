import numpy as np
import torch
import torch.nn as nn


class ProbabilisticForward(nn.Module):

    def __init__(self, model_dim, z_dim, last_layer=False, softplus=False, zero_init=True):
        super(ProbabilisticForward, self).__init__()

        self.last_layer = last_layer
        self.softplus = softplus
        self.zero_init = zero_init

        self.input_norm = nn.LayerNorm(model_dim)

        self.linear_z1 = nn.Linear(model_dim, z_dim)
        self.act_z = nn.SiLU()
        self.linear_z2_mean = nn.Linear(z_dim, z_dim)
        self.linear_z2_logvar = nn.Linear(z_dim, z_dim)

        if not last_layer:
            self.linear_out = nn.Linear(z_dim, model_dim)

        self.initialize()

    def initialize(self):

        nn.init.kaiming_normal_(self.linear_z1.weight)
        nn.init.constant_(self.linear_z2_mean.weight, 0.0)

        nn.init.normal_(self.linear_z2_logvar.weight, mean=0,
                        std=0.01 * np.sqrt(
                            2 / (self.linear_z2_logvar.weight.shape[0] * self.linear_z2_logvar.weight.shape[1])))

        nn.init.constant_(self.linear_z1.bias, 0.0)
        nn.init.constant_(self.linear_z2_mean.bias, 0.0)
        nn.init.constant_(self.linear_z2_logvar.bias, 0.0)

        if not self.last_layer:
            nn.init.constant_(self.linear_out.bias, 0.0)

            if self.zero_init:
                nn.init.constant_(self.linear_out.weight, 0.0)
            else:
                nn.init.xavier_normal_(self.linear_out.weight)

    def forward(self, x, p_z=None, infer_mean=False):

        z_raw_n = self.input_norm(x)
        z_raw_l = self.linear_z1(z_raw_n)
        z_raw = self.act_z(z_raw_l)

        z_mean = self.linear_z2_mean(z_raw)
        logvar = self.linear_z2_logvar(z_raw)

        z_std = torch.exp(logvar)

        z = torch.distributions.Normal(z_mean, z_std)

        if p_z is not None:
            if infer_mean:
                z_out = p_z.mean
            else:
                z_out = p_z.rsample()
        else:
            if infer_mean:
                z_out = z.mean
            else:
                z_out = z.rsample()

        if self.last_layer:
            out = torch.zeros_like(x)
        else:
            out = self.linear_out(z_out)
        return out, z
