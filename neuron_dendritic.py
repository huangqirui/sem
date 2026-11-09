import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import surrogate
import einops

from neuron_spiking import MultiStepLIF, LIF


class MultiStepDendriticNeuron2DSpace(nn.Module):
    def __init__(
            self,
            in_channels, num_branches=2, T=8,
            tau=2.0, threshold=1.0, init_v=0.0, learnable_tau=True, last=-1,
    ):
        super().__init__()
        self.num_branches = 2
        self.learnable_tau = learnable_tau
        self.downsample_space_1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.upsample_space_1 = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False, groups=in_channels)
        # dendritic
        self.dendritic_tau_0 = nn.Parameter(torch.zeros(in_channels))
        self.dendritic_tau_space_1 = nn.Parameter(torch.zeros(in_channels))
        self.dendritic_act = nn.ReLU()
        # soma
        self.soma_norm = nn.BatchNorm3d(in_channels)
        self.soma = MultiStepLIF(
            tau=tau, threshold=threshold, init_v=init_v, learnable_tau=True, learnable_vth=True, T=T, in_channels=in_channels,
        )

    def forward(self, x):  # (B,C,T,H,W)
        B, C, T, H, W = x.shape
        x_space_1 = self.downsample_space_1(x)
        # dendrite
        x_0 = self.dendritic_act(x * self.dendritic_tau_0.view(1, C, 1, 1, 1).sigmoid())
        x_space_1 = self.dendritic_act(x_space_1 * self.dendritic_tau_space_1.view(1, C, 1, 1, 1).sigmoid())
        x_space_1 = self.upsample_space_1(x_space_1)
        x = torch.stack([x_0, x_space_1])
        x = einops.rearrange(x, 'nb b c t h w -> b t h w c nb')
        x = einops.rearrange(x, 'b t h w c nb -> nb b c t h w')
        x = x.mean(0)
        # soma
        x = self.soma_norm(x)
        x = einops.rearrange(x, 'b c t h w -> t b c h w')
        x = self.soma(x)
        x = einops.rearrange(x, 't b c h w -> b c t h w')
        return x


class MultiStepRepeatedDendriticNeuron1DSpace(nn.Module):
    def __init__(
            self,
            in_channels, num_branches=2,
            tau=2.0, threshold=1.0, init_v=0.0, learnable_tau=True,
    ):
        super().__init__()
        self.num_branches = 2
        self.learnable_tau = learnable_tau
        self.downsample_space_1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.upsample_space_1 = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False, groups=in_channels)
        # dendritic
        self.dendritic_tau_0 = nn.Parameter(torch.zeros(in_channels))
        self.dendritic_tau_space_1 = nn.Parameter(torch.zeros(in_channels))
        self.dendritic_act = nn.ReLU()
        # soma
        self.soma_norm = nn.BatchNorm1d(in_channels)
        self.soma = MultiStepLIF(
            tau=tau, threshold=threshold, init_v=init_v, learnable_tau=True, learnable_vth=True, in_channels=in_channels,
        )

    def forward(self, x):  # (T,B,C,N)
        T, B, C, N = x.shape
        H = W = int(math.sqrt(N))
        x = einops.rearrange(x, 't b c (h w) -> b c t h w', h=H, w=W)
        x_space_1 = self.downsample_space_1(x)
        # dendrite
        x_0 = self.dendritic_act(x * self.dendritic_tau_0.view(1, C, 1, 1, 1).sigmoid())
        x_space_1 = self.dendritic_act(x_space_1 * self.dendritic_tau_space_1.view(1, C, 1, 1, 1).sigmoid())
        x_space_1 = self.upsample_space_1(x_space_1)
        x = torch.stack([x_0, x_space_1])
        x = einops.rearrange(x, 'nb b c t h w -> t b c nb (h w)')
        x = einops.rearrange(x, 't b c nb n -> nb t b c n')
        x = x.mean(0)
        # soma
        x = einops.rearrange(x, 't b c n -> (t b) c n')
        x = self.soma_norm(x)
        x = einops.rearrange(x, '(t b) c n -> t b c n', t=T, b=B)
        x = self.soma(x)
        return x


class MultiStepDendriticNeuron2DTime(nn.Module):
    def __init__(
            self,
            in_channels, num_branches=2, T=8,
            tau=2.0, threshold=1.0, init_v=0.0, learnable_tau=True, last=False,
    ):
        super().__init__()
        self.num_branches = 2
        self.learnable_tau = learnable_tau
        self.upsample_time_1 = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=False, groups=in_channels)
        self.upsample_time_2 = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(4, 1, 1), stride=(4, 1, 1), bias=False, groups=in_channels)
        # dendritic
        self.dendritic_tau_0 = nn.Parameter(torch.zeros(in_channels))
        self.dendritic_tau_time_1 = nn.Parameter(torch.zeros(in_channels))
        self.dendritic_tau_time_2 = nn.Parameter(torch.zeros(in_channels))
        self.dendritic_act = nn.ReLU()
        # soma
        self.soma_norm = nn.BatchNorm3d(in_channels)
        self.soma = MultiStepLIF(
            tau=tau, threshold=threshold, init_v=init_v, learnable_tau=True, learnable_vth=True, T=T, in_channels=in_channels,
        )

    def forward(self, x):  # (B,C,T,H,W)
        B, C, T, H, W = x.shape
        x_time_1 = einops.rearrange(x, 'b c (t g) h w -> b c t g h w', t=T//2, g=2)
        x_time_1 = x_time_1.mean(3)
        x_time_2 = einops.rearrange(x, 'b c (t g) h w -> b c t g h w', t=T//4, g=4)
        x_time_2 = x_time_2.mean(3)
        # dendrite
        x_0 = self.dendritic_act(x * self.dendritic_tau_0.view(1, C, 1, 1, 1).sigmoid())
        x_time_1 = self.dendritic_act(x_time_1 * self.dendritic_tau_time_1.view(1, C, 1, 1, 1).sigmoid())
        x_time_2 = self.dendritic_act(x_time_2 * self.dendritic_tau_time_2.view(1, C, 1, 1, 1).sigmoid())
        x_time_1 = self.upsample_time_1(x_time_1)
        x_time_2 = self.upsample_time_2(x_time_2)
        x = torch.stack([x_0, x_time_1, x_time_2])
        x = einops.rearrange(x, 'nb b c t h w -> b t h w c nb')
        x = einops.rearrange(x, 'b t h w c nb -> nb b c t h w')
        x = x.mean(0)
        # soma
        x = self.soma_norm(x)
        x = einops.rearrange(x, 'b c t h w -> t b c h w')
        x = self.soma(x)
        x = einops.rearrange(x, 't b c h w -> b c t h w')
        return x


class MultiStepRepeatedDendriticNeuron1DTime(nn.Module):
    def __init__(
            self,
            in_channels, num_branches=4,
            tau=2.0, threshold=1.0, init_v=0.0, learnable_tau=True,
    ):
        super().__init__()
        self.num_branches = 2
        self.learnable_tau = learnable_tau
        self.upsample_time_1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(2, 1), stride=(2, 1), bias=False, groups=in_channels)
        # dendritic
        self.dendritic_tau_0 = nn.Parameter(torch.zeros(in_channels))
        self.dendritic_tau_1 = nn.Parameter(torch.zeros(in_channels))
        self.dendritic_act = nn.ReLU()
        # soma
        self.soma_norm = nn.BatchNorm1d(in_channels)
        self.soma = MultiStepLIF(
            tau=tau, threshold=threshold, init_v=init_v, learnable_tau=True, learnable_vth=True, in_channels=in_channels,
        )

    def forward(self, x):  # (T,B,C,N)
        T, B, C, N = x.shape
        x = einops.rearrange(x, 't b c n -> b c t n')
        x_time_1 = einops.rearrange(x, 'b c (t g) n -> b c t g n', t=T // 2, g=2)
        x_time_1 = x_time_1.mean(3)
        x_0 = self.dendritic_act(x * self.dendritic_tau_0.view(1, C, 1, 1).sigmoid())
        x_time_1 = self.dendritic_act(x_time_1 * self.dendritic_tau_1.view(1, C, 1, 1).sigmoid())
        x_time_1 = self.upsample_time_1(x_time_1)
        x = torch.stack([x_0, x_time_1])
        x = einops.rearrange(x, 'nb b c t n -> t b c nb n')
        x = einops.rearrange(x, 't b c nb n -> nb t b c n')
        x = x.mean(0)
        # soma
        x = einops.rearrange(x, 't b c n -> (t b) c n')
        x = self.soma_norm(x)
        x = einops.rearrange(x, '(t b) c n -> t b c n', t=T, b=B)
        x = self.soma(x)
        return x


class RepeatedDendriticNeuron1D(nn.Module):
    def __init__(
            self,
            in_channels, num_branches,
            tau=2.0, threshold=1.0, init_v=0.0, learnable_tau=True,
    ):
        super().__init__()
        self.num_branches = num_branches
        self.learnable_tau = learnable_tau
        # dendritic
        self.dendritic_tau = nn.Parameter(torch.zeros(in_channels, num_branches))
        self.dendritic_act = nn.ReLU()
        # soma
        self.soma_norm = nn.BatchNorm1d(in_channels)
        self.soma = LIF(
            tau=tau, threshold=threshold, init_v=init_v, learnable_tau=True, learnable_vth=True, in_channels=in_channels,
        )

    def forward(self, x):  # (B,C,N)
        B, C, N = x.shape
        x = einops.repeat(x, 'b c n -> b c nb n', nb=self.num_branches)
        # dendrite
        x = x * self.dendritic_tau.sigmoid().view(1, C, self.num_branches, 1)
        x = self.dendritic_act(x)
        x = einops.rearrange(x, 'b c nb n -> nb b c n')
        x = x.mean(0)
        # soma
        x = self.soma_norm(x)
        x = self.soma(x)
        return x


class MultiStepSplitDendriticNeuron1D(nn.Module):
    def __init__(
            self,
            in_channels, num_branches,
            tau=2.0, threshold=1.0, init_v=0.0,
    ):
        super().__init__()
        self.num_branches = num_branches
        self.dim_per_branch = in_channels // num_branches
        # dendritic
        self.dendritic_tau = nn.Parameter(torch.zeros(self.dim_per_branch, num_branches))
        self.dendritic_act = nn.ReLU()
        # soma
        self.soma_norm = nn.BatchNorm1d(in_channels)
        self.soma = MultiStepLIF(
            tau=tau, threshold=threshold, init_v=init_v, learnable_tau=True, learnable_vth=True, in_channels=in_channels,
        )

    def forward(self, x):  # (T,B,C,N)
        T, B, C, N = x.shape
        x = einops.rearrange(x, 't b (c nb) n -> t b c nb n', c=self.dim_per_branch, nb=self.num_branches)
        # dendrite
        x = x * self.dendritic_tau.sigmoid().view(1, self.dim_per_branch, self.num_branches, 1)
        x = self.dendritic_act(x)
        # soma
        x = einops.rearrange(x, 't b c nb n -> (t b) (c nb) n')
        x = self.soma_norm(x)
        x = einops.rearrange(x, '(t b) c n -> t b c n', t=T, b=B)
        x = self.soma(x)
        return x

