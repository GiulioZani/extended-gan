import imp
import torch
from torch import nn
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
        # enc1 = self.enc[0](x)
        # latent = enc1
        # for i in range(1, len(self.enc)):
        #     latent = self.enc[i](latent)
        return self.enc(x)

    def skip(self, x):
        return self.forward(x)

    def skip_first(self, x):
        return self.enc[0](x)


class Decoder(nn.Module):
    def __init__(self, C_hid, hid_T, C_out, N_S):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            ConvSC(C_hid * 2 + hid_T, C_hid, stride=strides[0], transpose=True),
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[1:-1]],
            ConvSC(C_hid + hid_T, C_hid, stride=strides[-1], transpose=True),
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None, ySkip=None, last_skip=None):
        hid = t.cat([hid, ySkip, enc1], dim=1)
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](t.cat([hid, last_skip], dim=1))
        Y = self.readout(Y)
        return Y


class SimVP(nn.Module):
    def __init__(
        self,
        params,
        shape_in=(10, 1),
        hid_S=64,
        hid_T=256,
        N_S=4,
        N_T=8,
        incep_ker=[3, 5, 7, 11],
        groups=4,
    ):
        super(SimVP, self).__init__()
        self.params = params
        shape_in = (params.in_seq_len, params.n_channels)
        T, C = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T * hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, hid_T, C, N_S)
        self.conditional_embedding = nn.Embedding(2, (self.params.imsize // 4) ** 2)
        self.time_embedding = nn.Embedding(T * 2 + 5, (self.params.imsize // 4) ** 2)
        # self.future = t.tensor(True).int()

    def time_embed(self, x, time_stamps):
        B, T, C, H, W = x.shape

        label = t.tensor(time_stamps, device=x.device).int()
        labels = label.repeat(B, 1)

        embedding = self.time_embedding(labels)
        embedding = embedding.reshape(B, T, 1, H, W)
        return x + embedding

    def embed(self, x, future: bool):
        B, T, C, H, W = x.shape

        label = t.tensor(
            [T if future else 0 + i for i in range(T)], device=x.device
        ).int()
        labels = label.repeat(B, 1)

        embedding = self.time_embedding(labels)
        embedding = embedding.reshape(B, T, 1, H, W)

        # condition = t.tensor(future, device=x.device).int()
        # conditions = condition.repeat(B, 1)
        # conditions = self.conditional_embedding(conditions)
        # conditions = conditions.view(B, 1, 1, H, W)
        # conditions = conditions.repeat(1, T, 1, 1, 1)

        return x + embedding

    def pos_emb(self, x, T: int):
        B, C, H, W = x.shape
        # B of T tensor

        label = t.tensor(T, device=x.device).int()
        labels = label.repeat(B, 1)
        # ipdb.set_trace()

        embedding = self.time_embedding(labels)
        embedding = embedding.reshape(B, 1, H, W)
        embedding = embedding.repeat(1, C, 1, 1)
        return x + embedding

    def conditional_emb(self, x, future=True):
        B, C, H, W = x.shape
        # B of T tensor
        label = t.tensor(future, device=x.device).int()
        labels = label.repeat(B, 1)
        # ipdb.set_trace()

        embedding = self.conditional_embedding(labels)
        embedding = embedding.reshape(B, 1, H, W)
        embedding = embedding.repeat(1, C, 1, 1)
        return x + embedding

    def forward(self, x_raw, future=True):

        # ipdb.set_trace()
        B, T, C, H, W = x_raw.shape

        x = x_raw.view(B * T, C, H, W)

        embed = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        # ipdb.set_trace()
        # z = self.pos_emb(z)
        z = self.time_embed(z, [0 if future else T + i for i in range(T)])
        # hid = self.hid(z)
        hid, skip_all = self.hid(z)
        # ipdb.set_trace()

        # for i in range(T):

        #     hid[:, i, :, :, :] = self.pos_emb(
        #         hid[:, i, :, :, :], i if future else T - i - 1
        #     )
        hid = self.embed(hid, future)
        # hid = hid.view(B * T, C_, H_, W_)
        # hid = self.conditional_emb(hid, future=future)
        # hid = hid.view(B, T, C_, H_, W_)

        # skip_mean = hid.mean(dim=1)

        outs = []
        embed = self.enc.skip(x_raw[:, -1 if future else 0, ...])
        embed = self.pos_emb(embed, T if future else 0)
        last_skip = self.enc.skip_first(x_raw[:, -1 if future else 0, ...])

        if future:
            for i in range(T):
                out = self.dec(hid[:, i, :, :, :], embed, skip_all, last_skip)
                outs.append(out)
                # ipdb.set_trace()
                # z = t.cat([x[:B, :1, ...], out], 1)
                # z = self.embed(out, future)
                # ipdb.set_trace()
                last_skip = self.enc.skip_first(out)
                embed = self.enc.skip(out)
                embed = self.pos_emb(embed, T + i)
        else:
            for i in range(T - 1, -1, -1):
                out = self.dec(hid[:, i, :, :, :], embed, skip_all, last_skip)
                # outs.insert(0, out)
                outs.append(out)
                # z = t.cat([x[:B, :1, ...], out], 1)
                # z = self.embed(out, future)
                last_skip = self.enc.skip_first(out)
                embed = self.enc.skip(out)
                embed = self.pos_emb(embed, T - i - 1)

            # reverse outs array
            # ipdb.set_trace()
            outs = outs[::-1]

        Y = t.stack(outs, dim=1)
        Y = Y.reshape(B, T, C, H, W)
        Y = t.tanh(Y)
        return Y
