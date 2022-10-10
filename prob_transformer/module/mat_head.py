import torch
import torch.nn as nn

from prob_transformer.module.embedding import PosEmbedding


class ResNetBlock(nn.Module):
    def __init__(self, in_channel_size, out_channel_size, kernel, residual):
        super(ResNetBlock, self).__init__()

        self.residual = residual
        self.norm1 = nn.InstanceNorm2d(in_channel_size)
        self.norm2 = nn.InstanceNorm2d(in_channel_size)

        self.acti = nn.SiLU()
        if kernel == 1:
            self.conv1 = nn.Conv2d(in_channel_size, in_channel_size, kernel_size=1)
            self.conv2 = nn.Conv2d(in_channel_size, out_channel_size, kernel_size=1)
        else:
            self.conv1 = nn.Conv2d(in_channel_size, in_channel_size, kernel_size=kernel, padding=(kernel - 1) // 2)
            if in_channel_size == out_channel_size:
                self.conv2 = nn.Conv2d(in_channel_size, out_channel_size, kernel_size=kernel, padding=(kernel - 1) // 2)
            else:
                self.conv2 = nn.Conv2d(in_channel_size, out_channel_size, kernel_size=1)

    def initialize(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv2.weight, 0.0)
        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.constant_(self.conv1.bias, 0.0)

    def forward(self, x):

        x_hat = self.norm1(x)
        x_hat = self.acti(x_hat)
        x_hat = self.conv1(x_hat)
        x_hat = self.norm2(x_hat)
        x_hat = self.acti(x_hat)
        x_hat = self.conv2(x_hat)

        if self.residual:
            return x_hat + x
        else:
            return x_hat


class SimpleMatrixHead(nn.Module):

    def __init__(self, src_vocab_size, latent_dim, dropout, model_dim, out_channels,
                 res_layer, kernel, max_len):

        super(SimpleMatrixHead, self).__init__()

        self.row_latent_linear = nn.Linear(latent_dim, model_dim)
        self.col_latent_linear = nn.Linear(latent_dim, model_dim)

        self.latent_normal = nn.LayerNorm(latent_dim)

        self.row_src_embed = PosEmbedding(src_vocab_size, model_dim, max_len)
        self.col_src_embed = PosEmbedding(src_vocab_size, model_dim, max_len)
        self.row_pred_embed = PosEmbedding(13, model_dim, max_len)
        self.col_pred_embed = PosEmbedding(13, model_dim, max_len)

        conv_net_list = []
        for _ in range(res_layer):
            conv_net_list.append(ResNetBlock(model_dim, model_dim, kernel, residual=True))

        self.conv_net = nn.Sequential(*conv_net_list)

        self.generator = nn.Conv2d(model_dim, out_channels, kernel_size=1)

        self.initialize()

    def initialize(self):
        nn.init.kaiming_normal_(self.row_latent_linear.weight)
        nn.init.kaiming_normal_(self.col_latent_linear.weight)
        nn.init.constant_(self.row_latent_linear.bias, 0.0)
        nn.init.constant_(self.col_latent_linear.bias, 0.0)

    def forward(self, latent, src, pred, src_len):

        src_mask = self.make_mask(src_len)

        row_seq = self.row_src_embed(src)
        col_seq = self.col_src_embed(src)

        row_pred = self.row_pred_embed(pred)
        col_pred = self.col_pred_embed(pred)

        latent = self.latent_normal(latent)
        row_latent = self.row_latent_linear(latent)
        col_latent = self.col_latent_linear(latent)

        row_seq = row_seq.transpose(1, 2)
        col_seq = col_seq.transpose(1, 2)

        row_pred = row_pred.transpose(1, 2)
        col_pred = col_pred.transpose(1, 2)

        row_latent = row_latent.transpose(1, 2)
        col_latent = col_latent.transpose(1, 2)

        row_seq = row_seq.unsqueeze(2).repeat(1, 1, row_seq.shape[2], 1)
        col_seq = col_seq.unsqueeze(3).repeat(1, 1, 1, col_seq.shape[2])

        row_pred = row_pred.unsqueeze(2).repeat(1, 1, row_pred.shape[2], 1)
        col_pred = col_pred.unsqueeze(3).repeat(1, 1, 1, col_pred.shape[2])

        row_latent = row_latent.unsqueeze(2).repeat(1, 1, row_latent.shape[2], 1)
        col_latent = col_latent.unsqueeze(3).repeat(1, 1, 1, col_latent.shape[2])

        latent = row_seq + col_seq + row_pred + col_pred + row_latent + col_latent * src_mask

        output_mat = self.conv_net(latent)

        output_mat = self.generator(output_mat)

        return output_mat.permute(0, 2, 3, 1), src_mask

    def make_mask(self, src_len):
        with torch.no_grad():
            max_len = torch.max(src_len).item()
            mask = []
            for l in src_len:
                m = torch.ones([max_len, max_len]).to(src_len.device)
                m = torch.triu(m, diagonal=1)
                m[l:, :] = 0
                m[:, l:] = 0
                mask.append(m)
        return torch.stack(mask, dim=0).unsqueeze(1)
