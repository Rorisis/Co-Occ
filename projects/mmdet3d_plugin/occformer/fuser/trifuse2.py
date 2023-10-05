import random
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from mmdet3d.models.builder import FUSION_LAYERS
from mmcv.cnn import build_norm_layer
from projects.mmdet3d_plugin.utils import VanillaNeRFRadianceField, MLP


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
        self.plane_xy_conv = SPPModule(in_channels=in_channels*16, out_channels=out_channels)
        self.plane_yz_conv = SPPModule(in_channels=in_channels*128, out_channels=out_channels)
        self.plane_xz_conv = SPPModule(in_channels=in_channels*128, out_channels=out_channels)



    def forward(self, img_voxel_feats, pts_voxel_feats):
        img_voxel_feats = self.img_enc(img_voxel_feats)
        pts_voxel_feats = self.pts_enc(pts_voxel_feats)
        vis_weight = self.vis_enc(torch.cat([img_voxel_feats, pts_voxel_feats], dim=1))
        voxel_feats = vis_weight * img_voxel_feats + (1 - vis_weight) * pts_voxel_feats # B, C, H, W, L
        B, C, H, W, L = voxel_feats.shape

        plane_xy = self.plane_xy_conv(voxel_feats.view(B, C*L, H, W))
        plane_yz = self.plane_yz_conv(voxel_feats.view(B, C*H, W, L))
        plane_xz = self.plane_xz_conv(voxel_feats.view(B, C*W, H, L))
        # print(plane_xy.min(), plane_xy.min())
        return plane_xy, plane_yz, plane_xz


class SPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(SPPModule, self).__init__(**kwargs)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.dilated_conv3x3_rate6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.dilated_conv3x3_rate12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        # self.dilated_conv3x3_rate18 = nn.Sequential(
        #     nn.Conv2d(512+256, 256, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
        #     nn.BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        #     nn.ReLU()
        # )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        # x2 = self.dilated_conv3x3_rate18(x)
        x3 = self.dilated_conv3x3_rate6(x)
        x4 = self.dilated_conv3x3_rate12(x)
        ret = self.fuse(torch.cat([x1, x2, x3, x4], dim=1))
        
        return ret
