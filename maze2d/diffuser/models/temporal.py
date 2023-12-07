import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb
from ..utils.debugging import check_nan
from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

Activations = {
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
}

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x

class ValueFunction(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        ##
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down1 = Downsample1d(mid_dim_2)
        horizon = horizon // 2
        ##
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down2 = Downsample1d(mid_dim_3)
        horizon = horizon // 2
        ##
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        ## mask out first conditioning timestep, since this is not sampled by the model
        # x[:, :, 0] = 0

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        ##
        x = self.mid_block1(x, t)
        x = self.mid_down1(x)
        ##
        x = self.mid_block2(x, t)
        x = self.mid_down2(x)
        ##
        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out

class SimpleMLPValue(nn.Module):
    def __init__(
        self,
        transition_dim,
        horizon=256,
        kernel_size = 3,
        stride = 1,
        dim=8,
        dim_mults=(8, 4, 2, 1),
        embed_dim = 8,
        activation = "mish",
    ):
        super().__init__()
        self.horizon = horizon
        self.activation = activation
        dims = [transition_dim + embed_dim, *map(lambda m: dim * m, dim_mults), 1]
        in_out = list(zip(dims[:-1], dims[1:]))

        l = horizon
        for i, _ in enumerate(in_out):
            print(l)
            #l_in = horizons[-1]
            s = stride if i > 0 else 1
            l = int((l - kernel_size)/s + 1)
            #horizons.append(l_out)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish() if self.activation == "mish" else nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.conv_blocks = [
            nn.Sequential(
                nn.Conv1d(
                    in_dim, 
                    out_dim, 
                    kernel_size=kernel_size, 
                    stride = stride if i > 0 else 1,
                    padding = "valid"
                ), 
                Activations[self.activation](),
                nn.InstanceNorm1d(out_dim, affine=True)
                #nn.GroupNorm(1, out_dim)
            ) for i, (in_dim, out_dim) in enumerate(in_out)
        ]

        self.convs = nn.Sequential(
            *self.conv_blocks,
            nn.Flatten(),
            nn.Linear(l, 1)
        )

    def forward(self, x_t, cond, t):
        x_t = einops.rearrange(x_t, 'b h t -> b t h')
        t_emb = self.time_mlp(t).unsqueeze(-1).tile((1, 1, self.horizon))
        return self.convs(torch.cat([x_t, t_emb], dim = 1))

class SingleStepRewardMLP(nn.Module):
    def __init__(
        self,
        transition_dim,
        dim=8,
        dim_mults=(8, 4, 2, 1),
        embed_dim = 8,
        horizon = 1,
        activation = "mish",
        **kwargs,
    ):
        super().__init__()
        self.activation = activation
        self.horizon = horizon
        dims = [horizon * transition_dim + embed_dim, *map(lambda m: dim * m, dim_mults), 1]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            Activations[self.activation](),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.mlp_blocks = [
            nn.Sequential(
                nn.Linear(in_dim, out_dim), 
                Activations[self.activation](),
            ) for i, (in_dim, out_dim) in enumerate(in_out)
        ]

        self.mlp = nn.Sequential(
            *self.mlp_blocks,
            nn.Linear(1, 1)
        )

    def forward(self, x_t, cond, t):
        x_t = einops.rearrange(x_t, 'b h c -> b (h c)')
        t_emb = self.time_mlp(t)
        return self.mlp(torch.cat([x_t, t_emb], dim = 1))
    

class TemporalValue(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        time_dim=None,
        out_dim=1,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = time_dim or dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])

        print(in_out)
        for dim_in, dim_out in in_out:

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            horizon = horizon // 2

        fc_dim = dims[-1] * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.ReLU(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        for i, (resnet, resnet2, downsample) in enumerate(self.blocks):
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out
