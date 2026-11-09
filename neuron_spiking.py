import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import surrogate
import einops


class LIF(nn.Module):
    def __init__(
            self, in_channels,
            tau=2.0, threshold=1.0, init_v=0.0,
            learnable_tau=True, learnable_vth=True,
            spike_function=surrogate.Sigmoid(),
    ):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.spike_function = spike_function
        self.learnable_tau = learnable_tau
        self.learnable_vth = learnable_vth
        self.in_channels = in_channels

        self.v = None
        self.init_v = init_v

        if self.learnable_tau:
            init_tau = torch.full((in_channels,), -math.log(tau - 1.))
            self.tau = nn.Parameter(init_tau)
        else:
            self.tau = tau
        if self.learnable_vth:
            self.threshold = nn.Parameter(torch.as_tensor(math.log(math.exp(threshold) - 1.)))
        else:
            self.threshold = threshold

    def forward(self, x):  # B C H W
        self.v = torch.zeros_like(x.detach())

        if self.init_v != 0.:
            torch.fill_(self.v, self.init_v)

        tau = self.tau.sigmoid() if self.learnable_tau else 1.0 / self.tau
        if len(x.shape) == 4 and self.learnable_tau:
            tau = tau.view(self.in_channels, 1, 1)
        elif len(x.shape) == 3 and self.learnable_tau:
            tau = tau.view(self.in_channels, 1)

        threshold = F.softplus(self.threshold) if self.learnable_vth else self.threshold

        self.v = self.v + (x - self.v) * tau
        spike = self.spike_function(self.v - threshold)
        self.v = self.v * (1.0 - spike.detach())

        return spike

    def reset(self):
        if self.v is not None:
            self.v = self.v.detach()
            self.v.zero_()


class SSMLIF(nn.Module):
    def __init__(
            self, in_channels,
            tau=2.0, threshold=1.0, init_v=0.0,
            learnable_tau=True, learnable_vth=True,
            spike_function=surrogate.Sigmoid(),
    ):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.spike_function = spike_function
        self.learnable_tau = learnable_tau
        self.learnable_vth = learnable_vth
        self.in_channels = in_channels

        self.v = None
        self.spike = None
        self.init_v = init_v

        if self.learnable_tau:
            init_tau = torch.full((in_channels,), -math.log(tau - 1.))
            self.tau = nn.Parameter(init_tau)
        else:
            self.tau = tau
        if self.learnable_vth:
            self.threshold = nn.Parameter(torch.as_tensor(math.log(math.exp(threshold) - 1.)))
        else:
            self.threshold = threshold

    def forward(self, x, deltaA_h, renew_v=True):  # B C H W
        if renew_v:
            self.v = torch.zeros_like(x.detach())
            self.spike = torch.zeros_like(x.detach())

        if self.init_v != 0.:
            torch.fill_(self.v, self.init_v)

        tau = self.tau.sigmoid() if self.learnable_tau else 1.0 / self.tau
        if len(x.shape) == 4 and self.learnable_tau:
            tau = tau.view(self.in_channels, 1, 1)
        elif len(x.shape) == 3 and self.learnable_tau:
            tau = tau.view(self.in_channels, 1)

        threshold = F.softplus(self.threshold) if self.learnable_vth else self.threshold

        self.v = (self.v * (1.0 - self.spike.detach()) + deltaA_h) * (1 - tau) + x * tau
        self.spike = self.spike_function(self.v - threshold)

        return self.spike

    def reset(self):
        if self.v is not None:
            self.v = self.v.detach()
            self.v.zero_()
            self.spike = self.spike.detach()
            self.spike.zero_()


class MultiStepLIF(nn.Module):
    def __init__(
            self, in_channels,
            tau=2.0, threshold=1.0, init_v=0.0, T=8,
            learnable_tau=True, learnable_vth=True,
            spike_function=surrogate.Sigmoid(),
    ):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.spike_function = spike_function
        self.learnable_tau = learnable_tau
        self.learnable_vth = learnable_vth
        self.in_channels = in_channels
        self.T = T

        self.v = None
        self.init_v = init_v

        if self.learnable_tau:
            init_tau = torch.full((in_channels,), -math.log(tau - 1.))
            self.tau = nn.Parameter(init_tau)
        else:
            self.tau = tau
        if self.learnable_vth:
            init_vth = torch.full((self.T,), math.log(math.exp(threshold) - 1.))
            self.threshold = nn.Parameter(init_vth)
        else:
            self.threshold = threshold

    def forward(self, x):  # T B C H W
        self.v = torch.zeros_like(x[0].detach())    # B C H W
        if self.init_v != 0.:
            torch.fill_(self.v, self.init_v)

        tau = self.tau.sigmoid() if self.learnable_tau else 1.0 / self.tau
        if len(x.shape) == 5 and self.learnable_tau:
            tau = tau.view(self.in_channels, 1, 1)
        elif len(x.shape) == 4 and self.learnable_tau:
            tau = tau.view(self.in_channels, 1)
        threshold = F.softplus(self.threshold) if self.learnable_vth else self.threshold

        spike_seq = []
        for i in range(x.shape[0]):
            self.v = self.v + (x[i] - self.v) * tau
            spike = self.spike_function(self.v - threshold[i])
            spike_seq.append(spike)
            self.v = self.v * (1.0 - spike.detach())

        spike_seq = torch.stack(spike_seq)
        return spike_seq

    def reset(self):
        if self.v is not None:
            self.v = self.v.detach()
            self.v.zero_()
