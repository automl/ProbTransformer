import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, q_data_dim, m_data_dim, output_dim, num_head, zero_init=True, output_linear=True):
        super().__init__()
        assert q_data_dim % num_head == 0
        assert m_data_dim % num_head == 0
        self.key_dim = q_data_dim // num_head
        self.value_dim = m_data_dim // num_head

        self.key_dim_scaler = nn.Parameter(torch.FloatTensor([self.key_dim ** (-0.5)]), requires_grad=False)

        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([self.key_dim])), requires_grad=False)

        self.zero_init = zero_init
        self.output_linear = output_linear

        self.num_head = num_head
        self.linear_q = nn.Linear(q_data_dim, q_data_dim, bias=False)
        self.linear_k = nn.Linear(m_data_dim, m_data_dim, bias=False)
        self.linear_v = nn.Linear(m_data_dim, m_data_dim, bias=False)

        if self.output_linear:
            self.linear_o = nn.Linear(num_head * self.value_dim, output_dim, bias=True)

        self.initialize()

    def initialize(self):

        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)

        if self.zero_init:
            nn.init.constant_(self.linear_o.weight, 0.0)
        else:
            nn.init.xavier_uniform_(self.linear_o.weight)
        nn.init.constant_(self.linear_o.bias, 0.0)

    def forward(self, q_data, m_data, mask):

        batch_size = q_data.size(0)
        N_q_seq = q_data.size(1)
        N_m_seq = m_data.size(1)
        q = self.linear_q(q_data).view(batch_size, N_q_seq, self.num_head, self.key_dim).permute(0, 2, 1,
                                                                                                 3) * self.key_dim_scaler
        k = self.linear_k(m_data).view(batch_size, N_m_seq, self.num_head, self.value_dim).permute(0, 2, 3, 1)
        v = self.linear_v(m_data).view(batch_size, N_m_seq, self.num_head, self.value_dim).permute(0, 2, 1, 3)

        logits = torch.matmul(q, k) / self.scale

        if torch.is_autocast_enabled():
            bias = (1e4 * (mask - 1.))[:, None, :, :]
        else:
            bias = (1e9 * (mask - 1.))[:, None, :, :]
        logits = logits + bias

        weights = F.softmax(logits, dim=-1)

        weighted_avg = torch.matmul(weights, v).permute(0, 2, 1, 3)

        if self.output_linear:
            output = self.linear_o(weighted_avg.reshape(batch_size, N_q_seq, self.num_head * self.value_dim))
        else:
            output = weighted_avg.reshape(batch_size, N_q_seq, self.num_head * self.value_dim)

        return output


class PreNormAttention(nn.Module):

    def __init__(self, model_dim, num_head, encoder=False, zero_init=True):
        super().__init__()

        self.model_dim = model_dim
        self.num_head = num_head
        self.encoder = encoder

        self.src_ln = nn.LayerNorm(model_dim)

        if encoder:
            self.enc_ln = nn.LayerNorm(model_dim)

        self.attn = Attention(q_data_dim=model_dim, m_data_dim=model_dim, output_dim=model_dim, num_head=num_head,
                              zero_init=zero_init)
        self.initialize()

    def initialize(self):
        pass

    def forward(self, src_act, enc_act=None, mask=None):

        src_act = self.src_ln(src_act)
        if self.encoder:
            enc_act = self.enc_ln(enc_act)
        else:
            enc_act = src_act
        src_act = self.attn(src_act, enc_act, mask)
        return src_act
