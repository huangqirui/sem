import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuron import get_neuron
import einops
from neuron_spiking import SSMLIF


def selective_scan_torch(
        u,          # (B, K * C, L)
        # delta,      # (B, K * C, L)
        A,          # (K * C, N)
        B,          # (B, K, N, L)
        C,          # (B, K, N, L)
        D,          # (K * C)
        ssm_lif,
        a_t_bn,
        b_t_bn,
        # delta_bias: torch.Tensor = None,  # (K * C)
        # delta_softplus=True,
        oflex=True,
):
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = int(KCdim / K)

    u, A, B, C = u.float(), A.float(), B.float(), C.float()
    A = A.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    B = B.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)

    deltaA = a_t_bn(A)
    deltaB = b_t_bn(B)
    deltaB_u = torch.einsum('bdnl,bdl->bdln', deltaB, u)

    x = A.new_zeros((Batch, KCdim, N))
    ys = []
    for i in range(L):
        if i == 0:
            x = ssm_lif(deltaA_h=deltaA[:, :, :, i] * x, x=deltaB_u[:, :, i, :])
        else:
            x = ssm_lif(deltaA_h=deltaA[:, :, :, i] * x, x=deltaB_u[:, :, i, :], renew_v=False)
        y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (B, C, L)

    out = y + u * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)


class mamba_init:
    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        # A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        # A_log = torch.log(A)  # Keep A_log in fp32  # Di,Ds
        A_log = torch.zeros(d_state, dtype=torch.float32).view(1, -1).repeat(d_inner, 1).contiguous()
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()   # K,Di,Ds
            if merge:
                A_log = A_log.flatten(0, 1)     # K*Di,Ds
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        # dt_projs = [
        #     cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
        #     for _ in range(k_group)
        # ]
        # dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        # dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        # del dt_projs

        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds


class S3M(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=1,
            ssm_ratio=2.0,
            k_group=4,
            # dt_rank="auto",
            # act_layer=nn.SiLU,
            # dwconv ===============
            # d_conv=3,  # < 2 means no conv
            # conv_bias=True,
            # ======================
            dropout=0.0,
            # bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # initialize="v0",
            # ======================
            # forward_type="v2",
            # channel_first=False,
            # ======================
            num_branches=2, lif_type='frame',
            backend="cupy", lif_tau=1.5,
            **kwargs,
    ):
        super().__init__()
        self.k_group = k_group * 2
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16))

        # in proj =======================================
        self.in_bn = nn.BatchNorm1d(self.d_model)
        self.in_lif = get_neuron(f'{lif_type}_multistep_repeat_1d')(in_channels=self.d_model, num_branches=num_branches, tau=lif_tau)

        self.scan_proj = nn.Conv2d(self.d_model, self.d_inner, groups=self.d_model, bias=False, kernel_size=3, padding=1)
        self.scan_bn = nn.BatchNorm2d(self.d_inner)
        self.scan_act = get_neuron(f'{lif_type}_multistep_2d')(in_channels=self.d_inner, num_branches=num_branches, tau=lif_tau)

        self.conv2d = nn.Conv2d(self.d_model, self.d_inner, groups=self.d_model, bias=False, kernel_size=3, padding=1)
        self.act_bn = nn.BatchNorm2d(self.d_inner)
        self.act = get_neuron(f'{lif_type}_multistep_2d')(in_channels=self.d_inner, num_branches=num_branches, tau=lif_tau)

        # mamba proj =======================================
        self.x_proj = nn.Conv1d(self.d_inner * self.k_group, self.k_group * (self.d_state * 3), groups=self.k_group, kernel_size=1, stride=1, bias=False)
        self.a_t_bn = nn.BatchNorm2d(self.k_group * self.d_inner)
        self.b_t_bn = nn.BatchNorm2d(self.k_group * self.d_inner)
        self.ssm_lif = SSMLIF(in_channels=self.k_group * self.d_inner, tau=lif_tau)

        self.out_norm = nn.BatchNorm1d(self.d_inner)
        # out proj =======================================
        self.out_lif = get_neuron(f'{lif_type}_multistep_repeat_1d')(in_channels=self.d_inner, num_branches=num_branches, tau=lif_tau)
        self.out_proj = nn.Conv1d(self.d_inner, self.d_model, kernel_size=1, stride=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # init ============================
        _, self.Ds = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
        )

    def forward_ssm(self, x, scan):
        T, B, D, L = x.shape
        N = self.d_state
        K, D, R = self.k_group // 2, self.d_inner, self.dt_rank
        K_all = self.k_group
        BT = B * T
        G = L // K

        x = einops.rearrange(x, 't b c n -> (b t) c n')
        scan = einops.rearrange(scan, 't b n -> (b t) n')
        sort_order = torch.argsort(scan, dim=-1, descending=True)  # [BT, L]
        P_hard = F.one_hot(sort_order, num_classes=L).to(dtype=x.dtype)  # [BT, L, L]
        sorted_score = torch.sort(scan, dim=-1, descending=True).values  # [BT, L]
        dist = torch.abs(sorted_score.unsqueeze(-1) - scan.unsqueeze(1))  # [BT, L, L]

        tau = 0.1
        P_soft = torch.softmax(-dist / tau, dim=-1)  # [BT, L, L]
        P_st = P_soft + (P_hard - P_soft).detach()  # [BT, L, L]
        x_sorted = torch.einsum('bdn,brn->bdr', x, P_st)  # x_sorted=P_hard \times x, [BT, D, L]
        xs = x_sorted.reshape(BT, D, K, G)  # [BT, D, K, G]
        shift = G // 2
        P_st_roll = torch.roll(P_st, shifts=shift, dims=1)  # [BT, L, L]
        x_sorted_roll = torch.einsum('bdn,brn->bdr', x, P_st_roll)  # [BT, D, L]
        xs_roll = x_sorted_roll.reshape(BT, D, K, G)  # [BT, D, K, G]
        xs = torch.cat([xs, xs_roll], dim=2)
        xs = xs.view(BT, -1, G)
        x_dbl = self.x_proj(xs)     # B,K*Di,L -> B,K*(R+2N),L
        As, Bs, Cs = torch.split(x_dbl.view(BT, K_all, -1, G), [N, N, N], dim=2)    # B,K,R,L; B,K,N,L; B,K,N,L
        As = As.contiguous().view(BT, K_all, N, G)
        Bs = Bs.contiguous().view(BT, K_all, N, G)
        Cs = Cs.contiguous().view(BT, K_all, N, G)
        Ds = self.Ds.to(torch.float)  # (K * c)
        ys = selective_scan_torch(xs, As, Bs, Cs, Ds, self.ssm_lif, self.a_t_bn, self.b_t_bn).view(BT, K_all, -1, G)
        y = ys[:, :K]
        y = einops.rearrange(y, 'b k c l -> b c (k l)')
        y_roll = ys[:, K:]
        y_roll = einops.rearrange(y_roll, 'b k c l -> b c (k l)')
        y_roll = torch.roll(y_roll, shifts=-shift, dims=-1)

        inv_order = torch.empty_like(sort_order)
        inv_order.scatter_(1, sort_order, torch.arange(L, device=sort_order.device).unsqueeze(0).expand(BT, L))

        y = torch.gather(y, dim=2, index=inv_order.unsqueeze(1).expand(-1, D, -1))
        y_roll = torch.gather(y_roll, dim=2, index=inv_order.unsqueeze(1).expand(-1, D, -1))
        y = y + y_roll

        # y = y.view(B, T, -1, H, W)
        return y.to(x.dtype)

    def forward(self, x):
        B, C, T, N = x.shape
        H = W = int(math.sqrt(N))

        x = self.in_bn(x.flatten(2)).reshape(B, C, T, N).permute(2, 0, 1, 3).contiguous()  # T,B,C,N
        x = self.in_lif(x).permute(1, 2, 0, 3)   # B,C,T,N
        x = einops.rearrange(x, 'b c t (h w) -> (b t) c h w', h=H, w=W)

        scan = self.scan_proj(x)
        scan = self.scan_bn(scan)
        scan = einops.rearrange(scan, '(b t) c h w -> b c t h w', b=B, t=T)
        scan = self.scan_act(scan)
        scan = einops.rearrange(scan, 'b c t h w -> t b c (h w)')
        scan = scan.mean(2)     # t b n

        x = self.conv2d(x)  # (b, di, h, w)
        x = self.act_bn(x)
        x = einops.rearrange(x, '(b t) c h w -> b c t h w', b=B, t=T)
        x = self.act(x)
        x = einops.rearrange(x, 'b c t h w -> t b c (h w)')
        y = self.forward_ssm(x, scan)
        y = einops.rearrange(y, '(b t) c n -> b c (t n)', b=B, t=T)
        y = self.out_norm(y)
        y = einops.rearrange(y, 'b c (t n) -> t b c n', t=T, n=N)
        y = self.out_lif(y)
        y = einops.rearrange(y, 't b c n -> b c (t n)')
        y = self.out_proj(y)
        y = einops.rearrange(y, 'b c (t n) -> b c t n', t=T, n=N)
        return y
