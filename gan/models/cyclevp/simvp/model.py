import imp
import torch
from torch import dropout, nn
from .modules import ConvSC, Inception, Mid_Xnet
import ipdb
import torch as t


def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]],
        )

    def forward(self, x):  # B*4, 3, 128, 128
        # for i in range(0, len(self.enc)):
        #     x = self.enc[i](x)
        return self.enc(x)

    def skip(self, x):
        return self.forward(x)

    def skip_first(self, x):
        return self.enc[0](x)


class Decoder(nn.Module):
    def __init__(
        self,
        C_in,
        C_hid,
        hid_T,
        C_out,
        N_S,
        skip_first=False,
        skip_last=False,
        skip_time_encoded=False,
        dropout=False,
    ):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            ConvSC(
                (11 if skip_first else 1) * C_in
                + (1 if skip_time_encoded else 0) * hid_T,
                C_hid,
                stride=strides[0],
                transpose=True,
                dropout=dropout,
            ),
            *[
                ConvSC(C_hid, C_hid, stride=s, transpose=True, dropout=dropout)
                for s in strides[1:-1]
            ],
            ConvSC(
                C_hid + (10 if skip_last else 0) * C_in,
                C_hid * (2 if skip_last else 1),
                stride=strides[-1],
                transpose=True,
                dropout=dropout,
            ),
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

        if skip_last:
            self.last_convs = nn.Sequential(
                ConvSC(C_hid * 2, C_hid, stride=1, transpose=False),
                ConvSC(C_hid, C_hid, stride=1, transpose=False),
            )

        self.skip_first = skip_first
        self.skip_last = skip_last
        self.skip_time_encoded = skip_time_encoded

    def forward(self, hid, embedd_skip=None, skip_all=None, skip_last=None):

        if not self.skip_first:
            embedd_skip = None
        if not self.skip_time_encoded:
            skip_all = None

        if self.skip_first:
            embedd_skip = embedd_skip.view(
                embedd_skip.shape[0],
                embedd_skip.shape[1] * embedd_skip.shape[2],
                *embedd_skip.shape[3:],
            )

        skips = [hid, embedd_skip, skip_all]
        hid = torch.cat([skip for skip in skips if skip != None], dim=1)
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)

        if self.skip_last:
            skip_last = skip_last.view(
                skip_last.shape[0],
                skip_last.shape[1] * skip_last.shape[2],
                *skip_last.shape[3:],
            )
            hid = torch.cat([hid, skip_last], dim=1)
        Y = self.dec[-1](hid)

        if self.skip_last:
            Y = self.last_convs(Y)

        Y = self.readout(Y)
        return Y


class SimVP(nn.Module):
    def __init__(
        self,
        params,
        shape_in=(10, 1),
        hid_S=64,
        hid_D=64,
        hid_T=256,
        N_S=4,
        N_T=8,
        incep_ker=[3, 5, 7, 11],
        groups=4,
        skip_first=True,
        skip_last=True,
        skip_time_encoded=True,
    ):
        super(SimVP, self).__init__()
        self.params = params
        shape_in = (params.in_seq_len, params.n_channels)
        T, C = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T * hid_S, hid_T, N_T, incep_ker, groups)
        self.embedding = nn.Embedding(T * 2 + 5, (self.params.imsize // 4) ** 2)
        self.full_size_embedding = nn.Embedding(T * 2 + 5, (self.params.imsize) ** 2)

        self.dec = Decoder(
            hid_S,
            hid_D,
            hid_T,
            C,
            N_S,
            skip_first=skip_first,
            skip_last=skip_last,
            skip_time_encoded=skip_time_encoded,
            dropout=False,
        )

        self.skip_first = skip_first
        self.skip_last = skip_last

    def pos_emb(self, x, T):
        # ipdb.set_trace()
        B, C, H, W = x.shape
        # B of T tensor

        label = t.tensor(T, device=x.device).int()
        labels = label.repeat(B, 1)
        # ipdb.set_trace()

        embedding = self.embedding(labels)
        embedding = embedding.reshape(B, 1, H, W)
        embedding = embedding.repeat(1, C, 1, 1)
        return x + embedding

    def time_embed(self, x, start=0):
        B, T, C, H, W = x.shape

        label = t.tensor([i + start for i in range(T)], device=x.device).int()
        labels = label.repeat(B, 1)

        embedding = self.embedding(labels)
        embedding = embedding.reshape(B, T, 1, H, W)
        return x + embedding

    def future_embed(self, x):
        B, T, C, H, W = x.shape

        label = t.tensor([i + 10 for i in range(T)], device=x.device).int()
        labels = label.repeat(B, 1)

        embedding = self.embedding(labels)
        embedding = embedding.reshape(B, T, 1, H, W)
        return x + embedding

    def range_embed(self, x, start=0, end=10):
        B, T, C, H, W = x.shape

        label = t.tensor([i for i in range(start, end)], device=x.device).int()
        labels = label.repeat(B, 1)

        embedding = self.embedding(labels)
        embedding = embedding.reshape(B, T, 1, H, W)
        return x + embedding

    def skip_embed(self, skips, last_time_stamp=0):
        B, T, C, H, W = skips.shape

        label = t.tensor(
            [i + last_time_stamp for i in range(T)], device=skips.device
        ).int()
        label = label.repeat(B, 1)
        embeddings = self.full_size_embedding(label)

        embeddings = embeddings.reshape(B, T, 1, H, W)
        skips = skips + embeddings

        # reverse order of T in skips
        return skips  # .flip(1)

    def forward(self, x_raw, future=True):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)

        embed = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)

        embed = embed.view(B, T, C_, H_, W_)

        z = self.time_embed(z)

        hid, skip_all = self.hid(z)

        hid = self.range_embed(hid, start=T, end=T + T // 2)

        outs = []

        if not self.skip_first:
            embed = None

        skips = self.enc.skip_first(x)
        skips = skips.view(B, T, C_, H, W)
        skips_emb = self.skip_embed(skips, 0)

        out_embeds = []

        # first predict half of the future
        for i in range(T // 2):

            if self.skip_first:
                skip_first = self.time_embed(embed, i)

            out = self.dec(hid[:, i, ...], skip_first, skip_all, skips_emb)
            outs.append(out)
            if self.skip_first:
                n_embed = self.enc.skip(out)
                out_embeds.append(self.pos_emb(n_embed, T + i))
                n_embed = n_embed.unsqueeze(1)
                embed = t.cat([embed, n_embed], dim=1)
                embed = embed[:, 1:, ...]

        preds = t.stack(out_embeds[: T // 2], dim=1)
        z = self.time_embed(embed)
        z[:, : T // 2, ...] = z[:, T // 2 :, ...]
        z[:, T // 2 :, ...] = preds

        hid, skip_all = self.hid(z)
        hid = self.future_embed(hid)

        # predict the rest of the future
        for i in range(T // 2):
            out = self.dec(hid[:, i, ...], embed, skip_all, skips_emb)
            outs.append(out)
            if self.skip_first:
                n_embed = self.enc.skip(out)
                # out_embeds.append(self.pos_emb(n_embed, T + i))
                n_embed = n_embed.unsqueeze(1)
                embed = t.cat([embed, n_embed], dim=1)
                embed = embed[:, 1:, ...]
                # out_embeds.append(embed)
            # skip_first = self.enc.skip_first(out)

        Y = torch.stack(outs, dim=1)

        Y = Y.reshape(B, T, C, H, W)

        # ipdb.set_trace()

        Y = torch.tanh(Y)
        return Y
