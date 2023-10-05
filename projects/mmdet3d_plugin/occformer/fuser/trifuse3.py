import random
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from mmdet3d.models.builder import FUSION_LAYERS
from mmcv.cnn import build_norm_layer
from projects.mmdet3d_plugin.utils import VanillaNeRFRadianceField, MLP
import numpy as np


@FUSION_LAYERS.register_module()
class TriFuser(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if norm_cfg is None:
            norm_cfg = dict(type='BN3d', eps=1e-3, momentum=0.01)

        self.img_enc = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.pts_enc = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.vis_enc = nn.Sequential(
            nn.Conv3d(2*out_channels, 16, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, 16)[1],
            # nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv3d(16, 1, 1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        # self.plane_yz_mlp = MLP(input_dim=128, output_dim=1, net_depth=3,skip_layer=1)
        # self.plane_xz_mlp = MLP(input_dim=128, output_dim=1, net_depth=3,skip_layer=1)
        # self.plane_xy_mlp = MLP(input_dim=16, output_dim=1, net_depth=3,skip_layer=1)
        # self.plane_yz_mlp = nn.Sequential(
        #     nn.Conv3d(128, 1, 7, padding=3, bias=False),
        #     build_norm_layer(norm_cfg, 1)[1],
        #     # nn.BatchNorm3d(1),
        #     nn.ReLU(True),
        # )
        # self.plane_xz_mlp = nn.Sequential(
        #     nn.Conv3d(128, 1, 7, padding=3, bias=False),
        #     build_norm_layer(norm_cfg, 1)[1],
        #     # nn.BatchNorm3d(1),
        #     nn.ReLU(True),
        # )
        # self.plane_xy_mlp = nn.Sequential(
        #     nn.Conv3d(16, 1, 7, padding=3, bias=False),
        #     build_norm_layer(norm_cfg, 1)[1],
        #     # nn.BatchNorm3d(1),
        #     nn.ReLU(True),
        # )
        self.plane_xy_attn = LayerAttentionModule(num_channels=in_channels, attention_dim=2)
        self.plane_yz_attn = LayerAttentionModule(num_channels=in_channels, attention_dim=0)
        self.plane_xz_attn = LayerAttentionModule(num_channels=in_channels, attention_dim=1)



    def forward(self, img_voxel_feats, pts_voxel_feats):
        img_voxel_feats = self.img_enc(img_voxel_feats)
        pts_voxel_feats = self.pts_enc(pts_voxel_feats)
        vis_weight = self.vis_enc(torch.cat([img_voxel_feats, pts_voxel_feats], dim=1))
        voxel_feats = vis_weight * img_voxel_feats + (1 - vis_weight) * pts_voxel_feats # B, C, H, W, L
        B, C, H, W, L = voxel_feats.shape
        voxel_feats = voxel_feats.view(B,H,W,L,C)

        plane_xy = self.plane_xy_attn(voxel_feats).permute(0,-1,1,2)
        plane_yz = self.plane_yz_attn(voxel_feats).permute(0,-1,1,2)
        plane_xz = self.plane_xz_attn(voxel_feats).permute(0,-1,1,2)
        # print(plane_xy.shape, plane_xz.shape, plane_yz.shape)
        return plane_xy, plane_yz, plane_xz


class LayerAttentionModule(nn.Module):
    def __init__(
        self,
        num_channels: int,
        attention_dim: int
        ):
        super().__init__()

        self.Q_linear = nn.Linear(num_channels, num_channels)
        self.K_linear = nn.Linear(num_channels, num_channels)
        self.V_linear = nn.Linear(num_channels, num_channels)

        self.attention_dim = attention_dim

    def forward(self, x): # x: torch.Tensor # [B, H, W, L, C]
        x = x.transpose(1, self.attention_dim + 1) # [B, d_att, d_1, d_2, C]
        B, d_att, d_1, d_2, C = x.shape

        Q = self.Q_linear(x)
        if self.attention_dim in [0, 1]: # H or W
            Q = Q[:, d_att//2, ...] # [B, d_1, d_2, C]
        else: # L
            Q = Q[:, 0, ...] # [B, d_1, d_2, C]
        Q = Q.reshape(B, 1, d_1 * d_2 * C) # [B, 1, d_1 * d_2 * C]

        K = self.K_linear(x).reshape(B, d_att, d_1 * d_2 * C) # [B, d_att, d_1 * d_2 * C]
        weights = torch.bmm(Q, K.transpose(1, 2)) #! [B, 1, d_1 * d_2 * C] * [B, d_1 * d_2 * C, d_att] -> [B, 1, d_att]
        weights = torch.softmax(weights / np.sqrt(d_1 * d_2 * C), dim=-1)

        V = self.V_linear(x).reshape(B, d_att, d_1 * d_2 * C) # [B, d_att, d_1 * d_2 * C]
        ret = torch.bmm(weights, V) #! [B, 1, d_att] * [B, d_att, d_1 * d_2 * C] -> [B, 1, d_1 * d_2 * C]
        ret = ret.reshape(B, d_1, d_2, C)
        return ret
