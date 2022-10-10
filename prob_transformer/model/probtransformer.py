import torch
import torch.nn as nn
import torch.cuda.amp as amp

from prob_transformer.module.probformer_stack import ProbFormerStack
from prob_transformer.module.embedding import PosEmbedding


class ProbTransformer(nn.Module):

    def __init__(self, model_type, seq_vocab_size, trg_vocab_size, model_dim, max_len, n_layers,
                 num_head, ff_factor, z_factor, dropout, prob_layer, props=False, zero_init=True):
        super().__init__()

        self.n_layers = n_layers
        self.max_len = max_len
        self.zero_init = zero_init

        self.props = props

        self.model_type = model_type
        assert self.model_type in ['encoder', 'prob_encoder',
                                   'encoder_decoder', 'encoder_prob_decoder',
                                   'decoder', 'prob_decoder']

        self.probabilistic = "prob" in model_type
        self.encoder = "encoder" in model_type
        self.decoder = "decoder" in model_type

        if self.probabilistic:
            if isinstance(prob_layer, str):
                if prob_layer == 'all':
                    self.prob_layer = list(range(n_layers))
                elif prob_layer == 'middle':
                    self.prob_layer = list(range(n_layers))[1:-1]
                elif prob_layer == 'first':
                    self.prob_layer = [range(n_layers)[0]]
                elif prob_layer == 'last':
                    self.prob_layer = [range(n_layers)[-1]]
            else:
                self.prob_layer = prob_layer
        else:
            self.prob_layer = []

        if 'encoder' in model_type:
            self.encoder = ProbFormerStack(n_layers=n_layers, model_dim=model_dim, num_head=num_head,
                                           ff_factor=ff_factor, z_factor=z_factor, dropout=dropout, zero_init=zero_init,
                                           cross_attention=False, posterior=False,
                                           prob_layer=self.prob_layer if 'prob_encoder' in model_type else [])

        if 'prob_encoder' == model_type:
            self.post_encoder = ProbFormerStack(n_layers=n_layers, model_dim=model_dim, num_head=num_head,
                                                ff_factor=ff_factor, z_factor=z_factor, dropout=dropout,
                                                zero_init=zero_init, cross_attention=False,
                                                posterior=True, prob_layer=self.prob_layer)

        if 'encoder_decoder' == model_type or 'encoder_prob_decoder' == model_type:
            self.decoder = ProbFormerStack(n_layers=n_layers, model_dim=model_dim, num_head=num_head,
                                           ff_factor=ff_factor, z_factor=z_factor, dropout=dropout,
                                           zero_init=zero_init, cross_attention=True,
                                           posterior=False,
                                           prob_layer=self.prob_layer if 'prob_decoder' in model_type else [])

        if 'encoder_prob_decoder' == model_type:
            self.post_decoder = ProbFormerStack(n_layers=n_layers, model_dim=model_dim, num_head=num_head,
                                                ff_factor=ff_factor, z_factor=z_factor, dropout=dropout,
                                                zero_init=zero_init, cross_attention=False,
                                                posterior=True, prob_layer=self.prob_layer)

        if 'decoder' == model_type or 'prob_decoder' == model_type:
            self.decoder = ProbFormerStack(n_layers=n_layers, model_dim=model_dim, num_head=num_head,
                                           ff_factor=ff_factor,
                                           z_factor=z_factor, dropout=dropout, zero_init=zero_init,
                                           cross_attention=False, posterior=False,
                                           prob_layer=self.prob_layer if 'prob_decoder' in model_type else [])

        if 'prob_decoder' == model_type:
            self.post_decoder = ProbFormerStack(n_layers=n_layers, model_dim=model_dim, num_head=num_head,
                                                ff_factor=ff_factor, z_factor=z_factor, dropout=dropout,
                                                zero_init=zero_init, cross_attention=False,
                                                posterior=True, prob_layer=self.prob_layer)

        if 'encoder' in model_type:
            self.src_embed = PosEmbedding(seq_vocab_size, model_dim, max_len)

        if self.decoder:
            self.trg_embed = PosEmbedding(trg_vocab_size, model_dim, max_len)

            if self.props:
                self.type_embed = nn.Embedding(2, model_dim)

        if self.props:
            self.prop_embed = nn.Linear(self.props, model_dim)

        if self.probabilistic:
            if 'prob_encoder' in model_type:
                self.post_encoder_embed_seq = nn.Embedding(seq_vocab_size, model_dim)
                self.post_encoder_embed_trg = nn.Embedding(trg_vocab_size, model_dim)

            if 'prob_decoder' in model_type:
                self.post_decoder_embed_post = nn.Embedding(trg_vocab_size, model_dim)
                self.post_decoder_embed_trg = nn.Embedding(trg_vocab_size, model_dim)

            if self.props:
                self.post_prop_embed = nn.Linear(self.props, model_dim)

        self.output_ln = nn.LayerNorm(model_dim)
        self.output = nn.Linear(model_dim, trg_vocab_size)

        self.initialize()

    def initialize(self):

        # embedding initialization based on https://arxiv.org/abs/1711.09160
        if self.encoder:
            nn.init.normal_(self.src_embed.embed_seq.weight, mean=0.0, std=0.0001)

        if self.decoder:
            nn.init.normal_(self.trg_embed.embed_seq.weight, mean=0.0, std=0.0001)

        if self.props:
            nn.init.normal_(self.prop_embed.weight, mean=0.0, std=0.001)

        if self.probabilistic:
            if 'prob_encoder' in self.model_type:
                nn.init.normal_(self.post_encoder_embed_seq.weight, mean=0.0, std=0.0001)
                nn.init.normal_(self.post_encoder_embed_trg.weight, mean=0.0, std=0.0001)

            if 'prob_decoder' in self.model_type:
                nn.init.normal_(self.post_decoder_embed_post.weight, mean=0.0, std=0.0001)
                nn.init.normal_(self.post_decoder_embed_trg.weight, mean=0.0, std=0.0001)

                if self.props:
                    nn.init.normal_(self.prop_embed.weight, mean=0.0, std=0.001)

        nn.init.xavier_uniform_(self.output.weight)
        nn.init.constant_(self.output.bias, 0.0)

    def make_src_mask(self, src, src_len):
        src_mask = torch.arange(src.shape[1], device=src.device).expand(src.shape[:2]) < src_len.unsqueeze(1)
        src_mask = src_mask.type(src.type())
        return src_mask

    def make_trg_mask(self, trg_embed, trg_len):
        mask = torch.arange(trg_embed.size()[1], device=trg_embed.device).expand(
            trg_embed.shape[:2]) < trg_len.unsqueeze(1)
        mask = mask.unsqueeze(-1)
        sub_mask = torch.triu(
            torch.ones((1, trg_embed.size()[1], trg_embed.size()[1]), dtype=torch.bool, device=trg_embed.device),
            diagonal=1)
        sub_mask = sub_mask == 0
        trg_mask = mask & sub_mask
        trg_mask = trg_mask.type(trg_embed.type())
        return trg_mask

    def forward(self, src_seq=None, src_len=None, post_trg_seq=None, trg_shf_seq=None, trg_len=None,
                props=None, infer_mean=False, output_latent=False):

        if self.encoder:
            src_mask = self.make_src_mask(src_seq, src_len)
            seq_embed = self.src_embed(src_seq)  # * src_mask[:, :, None]
            if torch.is_autocast_enabled():
                src_mask = src_mask.half()
                seq_embed = seq_embed.half()

            if props is not None:
                prop_embed = self.prop_embed(props)
                seq_embed = seq_embed + prop_embed

        if self.decoder:
            trg_shift_embed = self.trg_embed(trg_shf_seq)

            if props is not None:
                type_embed = self.type_embed(torch.zeros((trg_shf_seq.shape[0], 1), dtype=torch.long,
                                                         device=trg_shf_seq.device))
                prop_embed = self.prop_embed(props) + type_embed
                trg_shift_embed = torch.cat([prop_embed, trg_shift_embed], 1)
                trg_len = trg_len + 1

            trg_shift_mask = self.make_trg_mask(trg_shift_embed, trg_len)

            if torch.is_autocast_enabled():
                trg_shift_mask = trg_shift_mask.half()
                trg_shift_embed = trg_shift_embed.half()

        if self.probabilistic and post_trg_seq != None:
            if 'prob_encoder' in self.model_type:
                post_seq_encoder_embed = self.post_encoder_embed_seq(src_seq)
                post_trg_encoder_embed = self.post_encoder_embed_trg(post_trg_seq)
                post_encoder_embed = post_seq_encoder_embed + post_trg_encoder_embed
                if torch.is_autocast_enabled():
                    post_encoder_embed = post_encoder_embed.half()

            if 'prob_decoder' in self.model_type:
                post_trg_decoder_embed = self.post_decoder_embed_trg(trg_shf_seq)
                post_post_decoder_embed = self.post_decoder_embed_post(post_trg_seq)
                post_decoder_embed = post_trg_decoder_embed + post_post_decoder_embed

                if props is not None:
                    type_embed = self.type_embed(torch.zeros((trg_shf_seq.shape[0], 1), dtype=torch.long,
                                                             device=trg_shf_seq.device))
                    prop_embed = self.prop_embed(props) + type_embed
                    post_decoder_embed = torch.cat([prop_embed, post_decoder_embed], 1)

                if torch.is_autocast_enabled():
                    post_decoder_embed = post_decoder_embed.half()

        # use transformer stacks
        if 'prob_encoder' in self.model_type:
            if post_trg_seq is not None:  # training
                _, p_z_list, _ = self.post_encoder(post_encoder_embed, src_mask[:, None, :])
                encoder_act, z_list, mask_encoder = self.encoder(seq_embed, src_mask[:, None, :], p_z_list=p_z_list)
            else:
                encoder_act, z_list, mask_encoder = self.encoder(seq_embed, src_mask[:, None, :], infer_mean=infer_mean)

        elif 'encoder' in self.model_type:
            encoder_act = self.encoder(seq_embed, src_mask[:, None, :])

        if 'encoder_prob_decoder' == self.model_type:
            if post_trg_seq is not None:  # training
                _, p_z_list, _ = self.post_decoder(post_decoder_embed, trg_shift_mask)
                decoder_act, z_list, mask_decoder = self.decoder(trg_shift_embed, trg_shift_mask,
                                                                 enc_act=encoder_act, enc_mask=src_mask[:, None, :],
                                                                 p_z_list=p_z_list)
            else:
                decoder_act, z_list, mask_decoder = self.decoder(trg_shift_embed, trg_shift_mask,
                                                                 enc_act=encoder_act, enc_mask=src_mask[:, None, :],
                                                                 infer_mean=infer_mean)

        elif 'encoder_decoder' == self.model_type:
            decoder_act = self.decoder(trg_shift_embed, trg_shift_mask, enc_act=encoder_act,
                                       enc_mask=src_mask[:, None, :])

        elif 'prob_decoder' == self.model_type:
            if post_trg_seq is not None:  # training
                _, p_z_list, _ = self.post_decoder(post_decoder_embed, trg_shift_mask)
                decoder_act, z_list, mask_decoder = self.decoder(trg_shift_embed, trg_shift_mask, p_z_list=p_z_list)
            else:
                decoder_act, z_list, mask_decoder = self.decoder(trg_shift_embed, trg_shift_mask, infer_mean=infer_mean)
        elif 'decoder' == self.model_type:
            decoder_act = self.decoder(trg_shift_embed, trg_shift_mask)

        if torch.is_autocast_enabled():
            if self.encoder:
                assert encoder_act.dtype == torch.float16
            if self.decoder:
                assert decoder_act.dtype == torch.float16

        if self.decoder:
            output_act = decoder_act
        else:
            output_act = encoder_act

        if torch.is_autocast_enabled():
            output_act = output_act.float()

        with amp.autocast(enabled=False):
            output_pred = self.output(self.output_ln(output_act))

        if self.decoder and props is not None:
            output_pred = output_pred[:, -trg_shf_seq.shape[1]:, :]

        if self.probabilistic and post_trg_seq is not None:
            return output_pred, (z_list, p_z_list)
        elif output_latent and post_trg_seq is None:
            return output_pred, output_act
        else:
            return output_pred
