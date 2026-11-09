import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.nn as nn
from neuron import get_neuron
from spikingjelly.clock_driven import functional
# from neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
import math
import numpy as np
import einops
from s3m import S3M


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)


class SpikingTokenizer(nn.Module):
    def __init__(
            self, img_size=128, patch_size=4, in_channels=2, timestep=16, embed_dims=256, pool_state=[True, True, True, True], num_branches=2, lif_type='frame',
            backend="cupy", lif_tau=1.5):
        super().__init__()
        self.image_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.pool_state = pool_state
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.t_num_patches = timestep // 2

        self.proj_conv = nn.Conv3d(in_channels, embed_dims // 8, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm3d(embed_dims // 8)
        self.proj_lif = get_neuron(f'{lif_type}_multistep_2d')(in_channels=embed_dims // 8, tau=lif_tau, T=16)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=1, ceil_mode=False)

        self.proj1_conv = nn.Conv3d(embed_dims // 8, embed_dims // 4, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm3d(embed_dims // 4)
        self.proj1_lif = get_neuron(f'{lif_type}_multistep_2d')(in_channels=embed_dims // 4, tau=lif_tau, T=16)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=1, ceil_mode=False)

        self.proj2_conv = nn.Conv3d(embed_dims // 4, embed_dims // 2, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm3d(embed_dims // 2)
        self.proj2_lif = get_neuron(f'{lif_type}_multistep_2d')(in_channels=embed_dims // 2, tau=lif_tau)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), dilation=1, ceil_mode=False)

        self.proj3_conv = nn.Conv3d(embed_dims // 2, embed_dims, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm3d(embed_dims)
        self.proj3_lif = get_neuron(f'{lif_type}_multistep_2d')(in_channels=embed_dims, tau=lif_tau)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv3d(embed_dims, embed_dims, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)

    def forward(self, x):
        # B, C, T, H, W = x.shape

        x = self.proj_conv(x)
        if self.pool_state[0]:
            x = self.maxpool(x)
        x = self.proj_bn(x).contiguous()
        x = self.proj_lif(x)

        x = self.proj1_conv(x)
        if self.pool_state[1]:
            x = self.maxpool1(x)
        x = self.proj1_bn(x).contiguous()
        x = self.proj1_lif(x)

        x = self.proj2_conv(x)
        if self.pool_state[2]:
            x = self.maxpool2(x)
        x = self.proj2_bn(x).contiguous()
        x = self.proj2_lif(x)

        x = self.proj3_conv(x)
        if self.pool_state[3]:
            x = self.maxpool3(x)
        x = self.proj3_bn(x).contiguous()
        x = self.proj3_lif(x)

        x = self.rpe_conv(x).flatten(3)

        return x


class SpikingSelfAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, num_branches=2, lif_type='frame',
            backend="cupy", lif_tau=1.5,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = get_neuron(f'{lif_type}_multistep_repeat_1d')(in_channels=dim, num_branches=num_branches, tau=lif_tau)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = get_neuron(f'{lif_type}_multistep_repeat_1d')(in_channels=dim, num_branches=num_branches, tau=lif_tau)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = get_neuron(f'{lif_type}_multistep_repeat_1d')(in_channels=dim, num_branches=num_branches, tau=lif_tau)

        self.attn_lif = get_neuron('repeat_1d')(in_channels=dim, num_branches=num_branches, tau=lif_tau)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = get_neuron(f'{lif_type}_multistep_repeat_1d')(in_channels=dim, num_branches=num_branches, tau=lif_tau)

    def forward(self, x):
        B, C, T, N = x.shape

        x = self.proj_bn(x.flatten(2)).reshape(B, C, T, N).permute(2, 0, 1, 3).contiguous()     # T,B,C,N
        x_for_qkv = self.proj_lif(x).permute(1, 2, 0, 3).flatten(2)

        q_conv_out = self.q_conv(x_for_qkv)     # B,C,T*N
        q_conv_out = self.q_bn(q_conv_out).reshape(B, C, T, N).permute(2, 0, 1, 3).contiguous()     # T,B,C,N
        q_conv_out = self.q_lif(q_conv_out).permute(1, 0, 3, 2).flatten(1, 2)   # B,T*N,C
        q = q_conv_out.reshape(B, T*N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(B, C, T, N).permute(2, 0, 1, 3).contiguous()
        k_conv_out = self.k_lif(k_conv_out).permute(1, 0, 3, 2).flatten(1, 2)   # B,T*N,C
        k = k_conv_out.reshape(B, T*N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(B, C, T, N).permute(2, 0, 1, 3).contiguous()
        v_conv_out = self.v_lif(v_conv_out).permute(1, 0, 3, 2).flatten(1, 2)   # B,T*N,C
        v = v_conv_out.reshape(B, T*N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale

        x = x.transpose(2, 3).reshape(B, C, T * N).contiguous()  # T,B,N,C
        x = self.attn_lif(x)

        x = self.proj_conv(x).reshape(B, C, T, N)

        return x


class SpikingMLP(nn.Module):
    def __init__(
            self,
            in_features, hidden_features=None, out_features=None, num_branches=2, lif_type='frame',
            backend="cupy", lif_tau=1.5,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = get_neuron('multistep_split_1d')(in_channels=hidden_features, num_branches=num_branches, tau=lif_tau)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = get_neuron(f'{lif_type}_multistep_repeat_1d')(in_channels=out_features, num_branches=num_branches, tau=lif_tau)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        B, C, T, N = x.shape

        x = self.fc2_bn(x.flatten(2)).reshape(B, C, T, N).permute(2, 0, 1, 3).contiguous()  # T,B,C,N
        x = self.fc2_lif(x).permute(1, 2, 0, 3).flatten(2)  # B,C,T*N
        x = self.fc1_conv(x)
        x = self.fc1_bn(x).reshape(B, self.c_hidden, T, N).permute(2, 0, 1, 3).contiguous()
        x = self.fc1_lif(x).permute(1, 2, 0, 3).flatten(2)
        x = self.fc2_conv(x).reshape(B, C, T, N)

        return x


class SpikingTransformerBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., num_branches=2, lif_type='frame',
            local=False,
            backend="cupy", lif_tau=1.5,
    ):
        super().__init__()
        if local:
            k_group = 7 if lif_type == 'space' else 8
            self.attn = S3M(
                dim, d_state=1, k_group=k_group,
            )
        else:
            self.attn = SpikingSelfAttention(
                dim, num_heads=num_heads, num_branches=num_branches, lif_type=lif_type,
                backend=backend, lif_tau=lif_tau,
            )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SpikingMLP(
            in_features=dim, hidden_features=mlp_hidden_dim, num_branches=num_branches, lif_type=lif_type,
            backend=backend, lif_tau=lif_tau,
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SEM(nn.Module):
    def __init__(
            self,
            img_size=128, seq_size=128, patch_size=16, image_channels=2, pool_state=[True, True, True, True],
            mlp_ratios=4, image_timestep=4, n_win=16,
            num_branches=2,
            local_embed_dims=256, local_num_heads=4, local_image_depths=6, local_seq_depths=6,
            global_embed_dims=384, global_num_heads=4, global_depths=6,
            num_classes=11, TET=True,
            backend="cupy", lif_tau=1.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.local_embed_dims = local_embed_dims
        # self.local_depths = local_depths
        self.image_timestep = image_timestep
        self.n_win = n_win
        self.TET = TET

        self.seq_encoder = SpikingTokenizer(
            img_size=seq_size, patch_size=patch_size, in_channels=image_channels, embed_dims=local_embed_dims,
            timestep=image_timestep, num_branches=num_branches, lif_type='time',
            pool_state=pool_state, backend=backend, lif_tau=lif_tau,
        )
        self.seq_num_patches = self.seq_encoder.num_patches

        self.patch_embed = SpikingTokenizer(
            img_size=img_size, patch_size=patch_size, in_channels=image_channels, embed_dims=local_embed_dims,
            timestep=image_timestep, num_branches=num_branches, lif_type='space',
            pool_state=pool_state, backend=backend, lif_tau=lif_tau,
        )
        self.num_patches = self.patch_embed.num_patches
        self.t_num_patches = self.patch_embed.t_num_patches
        self.image_pos_embed = get_sinusoid_encoding_table(self.num_patches * self.t_num_patches, local_embed_dims)
        self.seq_pos_embed = get_sinusoid_encoding_table(self.seq_num_patches * self.t_num_patches, local_embed_dims)

        # local
        self.local_image_blocks = nn.ModuleList([
            SpikingTransformerBlock(
                dim=local_embed_dims, num_heads=local_num_heads, mlp_ratio=mlp_ratios,
                num_branches=num_branches, lif_type='space', local=True,
                backend=backend, lif_tau=lif_tau,
            )
            for j in range(local_image_depths)
        ])

        self.local_seq_blocks = nn.ModuleList([
            SpikingTransformerBlock(
                dim=local_embed_dims, num_heads=local_num_heads, mlp_ratio=mlp_ratios,
                num_branches=num_branches, lif_type='time', local=True,
                backend=backend, lif_tau=lif_tau,
            )
            for j in range(local_seq_depths)
        ])

        # global
        self.global_blocks = nn.ModuleList([
            SpikingTransformerBlock(
                dim=global_embed_dims, num_heads=global_num_heads, mlp_ratio=mlp_ratios,
                num_branches=num_branches, lif_type='time', local=False,
                backend=backend, lif_tau=lif_tau,
            )
            for j in range(global_depths)
        ])
        self.global_bn = nn.BatchNorm1d(global_embed_dims)
        self.global_head = nn.Linear(global_embed_dims, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'image_pos_embed'}

    def forward(self, pts, frames):
        # local
        frames = frames.transpose(1, 2)

        images = self.patch_embed(frames)   # B,D,T,N
        T = images.shape[2]
        pos_embed = self.image_pos_embed.transpose(1, 2).reshape(1, self.local_embed_dims, T, self.num_patches).type_as(images).to(images.device).clone().detach()
        images = images + pos_embed

        for blk in self.local_image_blocks:
            images = blk(images)

        seqs = pts.transpose(1, 2)
        seqs = self.seq_encoder(seqs)  # B,D,T,N
        seq_pos_embed = self.seq_pos_embed.transpose(1, 2).reshape(1, self.local_embed_dims, T, self.seq_num_patches).type_as(seqs).to(seqs.device).clone().detach()
        seqs = seqs + seq_pos_embed

        for blk in self.local_seq_blocks:
            seqs = blk(seqs)  # B,D,T,N

        # global
        x = torch.cat((images, seqs), dim=-1)
        for blk in self.global_blocks:
            x = blk(x)

        if self.TET:
            x = self.global_bn(x.mean(-1)).permute(2, 0, 1)
            x = self.global_head(x)
        else:
            x = self.global_bn(x.mean(-1).mean(-1))
            x = self.global_head(x)

        return {
            'global': x,
        }


@register_model
def sem_small(pretrained=False, **kwargs):
    model = SEM(
        img_size=224, seq_size=128, patch_size=16, image_channels=2, pool_state=[True, True, True, True],
        mlp_ratios=4, image_timestep=kwargs['image_timestep'], n_win=kwargs['n_win'],
        num_branches=kwargs['num_branches'],
        local_embed_dims=256, local_num_heads=8, local_image_depths=1, local_seq_depths=1,
        global_embed_dims=256, global_num_heads=8, global_depths=4,
        num_classes=kwargs['num_classes'], TET=kwargs['TET'],
        backend=kwargs['backend'], lif_tau=kwargs['lif_tau'],
    )
    return model


