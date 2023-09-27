import random
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from mmdet3d.models.builder import FUSION_LAYERS
from mmcv.cnn import build_norm_layer

from projects.mmdet3d_plugin.utils.torch_moe_layer_nobatch import moe_layer, SingleExpert, Mlp as Mlp_torch
from projects.mmdet3d_plugin.utils.moe import MoE


@FUSION_LAYERS.register_module()
class MoEFuser(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if norm_cfg is None:
            norm_cfg = dict(type='BN3d', eps=1e-3, momentum=0.01)
        self.multi=multi
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
        self.moe_layer = MoE(input_size=128, output_size=128, num_experts=3, hidden_size=128, k=1, noisy_gating=True)
    
       
        self.multi_enc = nn.Sequential(
            nn.Conv3d(2*out_channels, 16, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, 16)[1],
            # nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv3d(16, 1, 1, padding=0, bias=False),
            nn.Sigmoid(),
        )
       

    def forward(self, img_voxel_feats=None, pts_voxel_feats=None):
        img_voxel_feats = self.img_enc(img_voxel_feats)
        pts_voxel_feats = self.pts_enc(pts_voxel_feats)
        vis_weight = self.multi_enc(torch.cat([img_voxel_feats, pts_voxel_feats], dim=1))
        voxel_feats = vis_weight * img_voxel_feats + (1 - vis_weight) * pts_voxel_feats
        B, C, H, W, L = voxel_feats.shape
        voxel_feats, balance_loss = self.moe_layer(voxel_feats.reshape(-1, C))
        
        return voxel_feats.reshape(B, C, H, W, L)
