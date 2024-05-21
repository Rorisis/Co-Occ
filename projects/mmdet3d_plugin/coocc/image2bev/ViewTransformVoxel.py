import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from mmdet3d.models.builder import NECKS
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_attention

@NECKS.register_module()
class ViewTransformVoxel(nn.Module):
    def __init__(
        self,
        bev_h, 
        bev_w,
        bev_z,
        dataset_type='nus',
        cross_transform=None,
        positional_encoding=None,
        embed_dims=128,
        **kwargs,
    ):

        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        
        self.dataset_type = dataset_type
        if self.dataset_type == 'nus':
            self.real_w = 100
            self.real_h = 100
            self.real_z = 8
        else:
            self.real_h = 51.2
            self.real_w = 51.2
            self.real_z = 6.4

        self.embed_dims = embed_dims
        self.bev_embed = nn.Embedding((self.bev_h * self.bev_w * self.bev_z), self.embed_dims)
        self.mask_embed = nn.Embedding(1, self.embed_dims)
        if positional_encoding:
            self.positional_encoding = build_positional_encoding(positional_encoding)
        if cross_transform:
            self.cross_transformer = build_transformer(cross_transformer)

    def forward(self, img_feats, img_metas):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embed.weight.to(dtype) #[128*128*16, dim]

        # Generate bev postional embeddings for cross and self attention
        bev_pos_cross_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=bev_queries.device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]
        bev_pos_self_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=bev_queries.device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]

        # Load query proposals
        proposal =  img_metas[0]['proposal'].reshape(self.bev_h, self.bev_w, self.bev_z)
        unmasked_idx = np.asarray(np.where(proposal.reshape(-1)>0)).astype(np.int32)
        masked_idx = np.asarray(np.where(proposal.reshape(-1)==0)).astype(np.int32)
        vox_coords, ref_3d = self.get_ref_3d()

        # Compute seed features of query proposals by deformable cross attention
        seed_feats = self.cross_transformer.get_vox_features(
            mlvl_feats, 
            bev_queries,
            self.bev_h,
            self.bev_w,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos_cross_attn,
            img_metas=img_metas,
            prev_bev=None,
        )

        # Complete voxel features by adding mask tokens
        vox_feats = torch.empty((self.bev_h, self.bev_w, self.bev_z, self.embed_dims), device=bev_queries.device)
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = seed_feats[0]
        vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = self.mask_embed.weight.view(1, self.embed_dims).expand(masked_idx.shape[1], self.embed_dims).to(dtype)

        print("voxel:", vox_feats_flatten.shape) # B, C, H, W, D
        return vox_feats_flatten
    
    def get_ref_3d(self):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
            ref_3d (Array): 3D reference points
        """
        if self.dataset_type == 'nus':
            scene_size = (100, 100, 8)
            vox_origin = np.array([0, 0, -1])
        else:
            scene_size = (51.2, 51.2, 6.4)
            vox_origin = np.array([0, -25.6, -2])
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

        return vox_coords, ref_3d