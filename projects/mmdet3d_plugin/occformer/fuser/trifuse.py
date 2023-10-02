import random
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from mmdet3d.models.builder import FUSION_LAYERS
from mmcv.cnn import build_norm_layer
@FUSION_LAYERS.register_module()
class TriFuser(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if norm_cfg is None:
            norm_cfg = dict(type='BN2d', eps=1e-3, momentum=0.01)
        self.img_enc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.pts_enc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.vis_enc = nn.Sequential(
            nn.Conv2d(2*out_channels, 16, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, 16)[1],
            # nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 1, padding=0, bias=False),
            nn.Sigmoid(),
        )

        self.plane_yz_mlp = nn.Sequential(
            nn.Conv3d(128, 1, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, 1)[1],
            # nn.BatchNorm3d(1),
            nn.ReLU(True),
        )
        self.plane_xz_mlp = nn.Sequential(
            nn.Conv3d(128, 1, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, 1)[1],
            # nn.BatchNorm3d(1),
            nn.ReLU(True),
        )
        self.plane_xy_mlp = nn.Sequential(
            nn.Conv3d(16, 1, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, 1)[1],
            # nn.BatchNorm3d(1),
            nn.ReLU(True),
        )

    def forward(self, img_voxel_feats, pts_voxel_feats):
        # print("img_voxel_feats",img_voxel_feats.shape)
        img_plane_xy = self.plane_xy_mlp(img_voxel_feats.permute(0,-1,1,2,3)).squeeze(1)
        img_plane_yz = self.plane_yz_mlp(img_voxel_feats.permute(0,2,1,3,4)).squeeze(1)
        img_plane_xz = self.plane_xz_mlp(img_voxel_feats.permute(0,3,1,2,4)).squeeze(1)
        pts_plane_xy = self.plane_xy_mlp(pts_voxel_feats.permute(0,-1,1,2,3)).squeeze(1)
        pts_plane_yz = self.plane_yz_mlp(pts_voxel_feats.permute(0,2,1,3,4)).squeeze(1)
        pts_plane_xz = self.plane_xz_mlp(pts_voxel_feats.permute(0,3,1,2,4)).squeeze(1)
        # print("img_plane_xy", img_plane_xy.shape, "img_plane_yz", img_plane_yz.shape, "img_plane_xz", img_plane_xz.shape)
        img_plane_xy = self.img_enc(img_plane_xy)
        img_plane_yz = self.img_enc(img_plane_yz)
        img_plane_xz = self.img_enc(img_plane_xz)
        pts_plane_xy = self.pts_enc(pts_plane_xy)
        pts_plane_yz = self.pts_enc(pts_plane_yz)
        pts_plane_xz = self.pts_enc(pts_plane_xz)
        vis_weight_xy = self.vis_enc(torch.cat([img_plane_xy, pts_plane_xy], dim=1))
        vis_weight_yz = self.vis_enc(torch.cat([img_plane_yz, pts_plane_yz], dim=1))
        vis_weight_xz = self.vis_enc(torch.cat([img_plane_xz, pts_plane_xz], dim=1))
        plane_xy = vis_weight_xy * img_plane_xy + (1 - vis_weight_xy) * pts_plane_xy
        plane_yz = vis_weight_yz * img_plane_yz + (1 - vis_weight_yz) * pts_plane_yz
        plane_xz = vis_weight_xz * img_plane_xz + (1 - vis_weight_xz) * pts_plane_xz

        return plane_xy, plane_yz, plane_xz