import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, model_dim, ff_dim, zero_init=True):
        super(FeedForward, self).__init__()

        self.zero_init = zero_init

        self.input_norm = nn.LayerNorm(model_dim)
        self.linear_1 = nn.Linear(model_dim, ff_dim)
        self.linear_2 = nn.Linear(ff_dim, model_dim)
        self.act = nn.SiLU()

        self.initialize()

    def initialize(self):

        nn.init.kaiming_normal_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0.0)
        nn.init.constant_(self.linear_2.bias, 0.0)

        if self.zero_init:
            nn.init.constant_(self.linear_2.weight, 0.0)
        else:
            nn.init.xavier_normal_(self.linear_2.weight)

    def forward(self, x):

        x = self.input_norm(x)

        x = self.act(self.linear_1(x))

        return self.linear_2(x)
