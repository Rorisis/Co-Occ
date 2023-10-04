import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import FUSION_LAYERS
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_attention
from projects.mmdet3d_plugin.utils.point_generator import MlvlPointGenerator
from .modules.cross_attention import CrossAttention

@FUSION_LAYERS.register_module()
class AttnFuser(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dims, positional_encoding,
                 real_h=102.4, strides=[1, 1, 1, 1], norm_cfg=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.real_h = real_h
        self.embed_dims = embed_dims
        self.strides = strides
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.cross_attention = CrossAttention()

        if norm_cfg is None:
            norm_cfg = dict(type='BN3d', eps=1e-3, momentum=0.01)
        
        # self.img_enc = nn.Sequential(
        #     nn.Conv3d(in_channels, out_channels, 7, padding=3, bias=False),
        #     build_norm_layer(norm_cfg, out_channels)[1],
        #     # nn.BatchNorm3d(out_channels),
        #     nn.ReLU(True),
        # )
        # self.pts_enc = nn.Sequential(
        #     nn.Conv3d(in_channels, out_channels, 7, padding=3, bias=False),
        #     build_norm_layer(norm_cfg, out_channels)[1],
        #     # nn.BatchNorm3d(out_channels),
        #     nn.ReLU(True),
        # )
        self.vis_enc = nn.Sequential(
            nn.Conv3d(2*out_channels, out_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(16),
            nn.ReLU(True),
        )

        # self.vis_enc2 = nn.Sequential(
        #     nn.Conv3d(192, out_channels, 3, padding=1, bias=False),
        #     build_norm_layer(norm_cfg, out_channels)[1],
        #     # nn.BatchNorm3d(16),
        #     nn.ReLU(True),
        # )
        self.point_generator = MlvlPointGenerator(strides)

    def get_ref_3d(self):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
            ref_3d (Array): 3D reference points
        """
        scene_size = (102.4, 102.4, 8)
        vox_origin = np.array([-51.2, -51.2, -5])
        voxel_size = self.real_h / self.bev_h

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1), idx], axis=0).astype(int).T

        # Normalize the voxels centroids in lidar cooridnates
        ref_3d = np.concatenate([(xv.reshape(1,-1)+0.5)/self.bev_h, (yv.reshape(1,-1)+0.5)/self.bev_w, (zv.reshape(1,-1)+0.5)/self.bev_z,], axis=0).astype(np.float64).T 

        return torch.tensor(vox_coords), torch.tensor(ref_3d)

    def forward(self, img_voxel_feats, pts_voxel_feats):
        # print(img_voxel_feats.shape, pts_voxel_feats.shape)
        if img_voxel_feats != None:
            bs, C, H, W, L = img_voxel_feats.shape
            dtype = img_voxel_feats.dtype
        else:
            bs, C, H, W, L = pts_voxel_feats.shape
            dtype = pts_voxel_feats.dtype
        # bev_queries = self.bev_embed.weight.to(dtype) #[128*128*16, dim]
        x = pts_voxel_feats
        kv = self.vis_enc(torch.cat([img_voxel_feats, pts_voxel_feats], dim=1))
        padding_mask = x.new_zeros((bs, ) + x.shape[-3:], dtype = torch.bool)
        pos_embed = self.positional_encoding(padding_mask)
        padding_mask_kv = kv.new_zeros((bs, ) + kv.shape[-3:], dtype = torch.bool)
        pos_embed_kv = self.positional_encoding(padding_mask_kv)
        # level_embed = self.level_encoding.weight[0]
        # pos_embed = level_embed.view(1, -1, 1, 1, 1) + pos_embed
        
        reference_points = self.point_generator.single_level_grid_priors(
                x.shape[-3:], 0, device=x.device)
        factor = x.new_tensor([[L, H, W]]) * self.strides[0]
        reference_points = reference_points / factor
        
        # shape (batch_size, c, x_i, y_i, z_i) -> (x_i * y_i * z_i, batch_size, c)
        x_projected = x.flatten(2).permute(2,0,1)
        pos_embed = pos_embed.flatten(2).permute(2,0,1)
        kv_projected = kv.flatten(2).permute(2,0,1)
        pos_embed_kv = pos_embed_kv.flatten(2).permute(2,0,1)
        padding_mask = padding_mask.flatten(1)
        spatial_shapes = [x.shape[-3:]]
        # shape (num_encoder_levels, 2), from low
        # resolution to high resolution
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=x.device)
        # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = reference_points[None, :, None].repeat(
            bs, 1, 1, 1)
        valid_radios = reference_points.new_ones(
            (bs, 1, 2))
        # Generate bev postional embeddings for cross and self attention
        # bev_pos_cross_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=bev_queries.device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]
        # bev_pos_self_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=bev_queries.device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]

        # vox_coords, ref_3d = self.get_ref_3d()
        # ref_3d = ref_3d.to(dtype)
        # # compute fused features by deformable cross_attention 
        # x = self.vis_enc(torch.cat([img_voxel_feats, pts_voxel_feats], dim=1)).reshape(-1, self.embed_dims)

        # queries = pts_voxel_feats.reshape(-1, self.embed_dims)

        voxel_feats = self.cross_attention(query=x_projected, key=kv_projected, value=kv_projected, 
                                           query_pos=None, key_pos=None, attn_masks=None, 
                                           query_key_padding_mask=padding_mask, spatial_shapes=spatial_shapes,
                                           reference_points=reference_points, level_start_index=level_start_index,
                                           valid_radios=valid_radios,)
        voxel_feats = voxel_feats.reshape(bs, -1, H, W, L)
        # voxel_feats = self.vis_enc2(voxel_feats)
        return voxel_feats

