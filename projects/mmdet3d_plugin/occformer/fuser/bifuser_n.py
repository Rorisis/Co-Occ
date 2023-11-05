import random
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from mmdet3d.models.builder import FUSION_LAYERS
from mmcv.cnn import build_norm_layer

from mmdet3d.ops import furthest_point_sample, gather_points, ball_query

@FUSION_LAYERS.register_module()
class BiFuser_N(nn.Module):
    def __init__(self, in_channels, out_channels, knum=1, norm_cfg=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.knum = knum
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
        # self.vis_enc = nn.Sequential(
        #     nn.Conv3d(2*out_channels, 16, 3, padding=1, bias=False),
        #     build_norm_layer(norm_cfg, 16)[1],
        #     # nn.BatchNorm3d(16),
        #     nn.ReLU(True),
        #     nn.Conv3d(16, 1, 1, padding=0, bias=False),
        #     nn.Sigmoid(),
        # )

        self.con_enc = nn.Sequential(
            nn.Conv3d(in_channels*4, out_channels*2, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels*2),
            nn.ReLU(True),
            nn.Conv3d(in_channels*2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
            )

        self.knn_enc = nn.Sequential(
            nn.Linear(in_channels*knum, out_channels),
            # nn.BatchNorm1d(self.in_channels_2D[i]),
            nn.ReLU()
            )

    def fps_NN_fast(self, query, key, fps_num, radius, max_cluster_samples, dist_thresh, num):
        """Efficient NN search for huge amounts of query and key (suppose queries are redundant)
        
            Behaivor:
            1. apply FPS on query and generate representative queries (repr_query)
            2. calculate repr_queries' distances with all keys, and get the NN key
            3. apply ball query to assign the same NN key with the group center 

        """
        if num == 1:
            query_NN_key_idx = torch.zeros_like(query[:, 0]).long() - 1
        else:
            query_NN_key_idx = (torch.zeros_like(query[:, 0]).long() - 1).repeat(num, 1)
        query = query[:, 1:].unsqueeze(0)
        key = key[:, 1:].unsqueeze(0)

        if num == 1:
            if query.shape[1] <= fps_num:
                dist = torch.norm(query.float().unsqueeze(2) - key.float().unsqueeze(1), p=2, dim=-1)
                val, NN_key_idx = dist.squeeze(0).min(-1)
                valid_mask = val < dist_thresh
                query_NN_key_idx[valid_mask] = NN_key_idx[valid_mask]
                return query_NN_key_idx

            else:
                repr_query_idx = furthest_point_sample(query.float().contiguous(), fps_num)
                repr_query = query[:, repr_query_idx[0].long(), :]
                # repr_query = gather_points(query.permute(0,2,1).float().contiguous(), repr_query_idx)

                dist = torch.norm(repr_query.float().unsqueeze(2) - key.float().unsqueeze(1), p=2, dim=-1)
                val, NN_key_idx = dist.squeeze(0).min(-1)
                valid_mask = val < dist_thresh

                query_group_idx = ball_query(0, radius, max_cluster_samples, query.float(), repr_query.float())
                query_group_idx = query_group_idx.squeeze(0).long()
                
                # ipdb.set_trace()
                # tmp = query_group_idx.reshape(-1).unique()

                expanded_NN_key_idx = NN_key_idx.unsqueeze(1).repeat(1, max_cluster_samples).reshape(-1)
                expanded_valid_mask = valid_mask.unsqueeze(1).repeat(1, max_cluster_samples).reshape(-1)
                query_group_idx = query_group_idx.reshape(-1)

                # select valid NN key assign
                valid_NN_key_idx = expanded_NN_key_idx[expanded_valid_mask]
                valid_query_group_idx = query_group_idx[expanded_valid_mask]
                
                query_NN_key_idx[valid_query_group_idx] = valid_NN_key_idx
            
        else:
            if query.shape[1] <= fps_num:
                dist = torch.norm(query.float().unsqueeze(2) - key.float().unsqueeze(1), p=2, dim=-1)
                for i in range(num):
                    val, NN_key_idx = torch.topk(dist.squeeze(0), i+1, dim=-1, largest=False)
                    valid_mask = val < dist_thresh
                    query_NN_key_idx[i][valid_mask] = NN_key_idx[valid_mask]
                return query_NN_key_idx

            else:
                repr_query_idx = furthest_point_sample(query.float().contiguous(), fps_num)
                repr_query = query[:, repr_query_idx[0].long(), :]
                # repr_query = gather_points(query.permute(0,2,1).float().contiguous(), repr_query_idx)

                dist = torch.norm(repr_query.float().unsqueeze(2) - key.float().unsqueeze(1), p=2, dim=-1)
                
                val_, NN_key_idx_ = torch.topk(dist.squeeze(0), num, dim=-1, largest=False)
                for i in range(num):
                    val = val_[:, i]
                    NN_key_idx = NN_key_idx_[:, i]
                    valid_mask = val < dist_thresh

                    query_group_idx = ball_query(0, radius, max_cluster_samples, query.float(), repr_query.float())
                    query_group_idx = query_group_idx.squeeze(0).long()
                    
                    # ipdb.set_trace()
                    # tmp = query_group_idx.reshape(-1).unique()

                    expanded_NN_key_idx = NN_key_idx.unsqueeze(1).repeat(1, max_cluster_samples).reshape(-1)
                    expanded_valid_mask = valid_mask.unsqueeze(1).repeat(1, max_cluster_samples).reshape(-1)
                    query_group_idx = query_group_idx.reshape(-1)

                    # select valid NN key assign
                    valid_NN_key_idx = expanded_NN_key_idx[expanded_valid_mask]
                    valid_query_group_idx = query_group_idx[expanded_valid_mask]
                    
                    query_NN_key_idx[i][valid_query_group_idx] = valid_NN_key_idx

            return query_NN_key_idx

    def forward(self, img_voxel_feats, pts_voxel_feats): # img_feats: N, C, H, W, L 

        B, C, H, W, L = img_voxel_feats.shape
        inds_img = torch.nonzero(img_voxel_feats.sum(1))
        inds_pts = torch.nonzero(pts_voxel_feats.sum(1))

        img_voxel_feats, pts_voxel_feats = img_voxel_feats.permute(0, 2, 3, 4, 1), pts_voxel_feats.permute(0, 2, 3, 4, 1)
        # print(inds_pts.shape)
        # print(pts_voxel_feats.shape)
        # print(pts_voxel_feats[inds_pts[:2]].shape)
        pts_voxel_feats = pts_voxel_feats
        selected_pts_feats = pts_voxel_feats[inds_pts[:, 0], inds_pts[:, 1], inds_pts[:, 2], inds_pts[:, 3]].contiguous()

        nearest_img_coords = self.fps_NN_fast(inds_pts.contiguous(), inds_img.contiguous(), fps_num=2048, radius=6, max_cluster_samples=200, dist_thresh=13.3, num=self.knum)
        if self.knum == 1:
            indices = inds_img[nearest_img_coords]
            nearest_img_feats = img_voxel_feats[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]].contiguous()
        else:
            nearest_img_feats = []
            for i in range(self.knum):
                indices = inds_img[nearest_img_coords[i]]
                nearest_img_feat = img_voxel_feats[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]].contiguous()
                nearest_img_feats.append(nearest_img_feat)
            nearest_img_feats = torch.cat(nearest_img_feats, dim=1)
        nearest_img_feats = self.knn_enc(nearest_img_feats) * selected_pts_feats
        # nearest_img_feats = self.knn_enc(nearest_img_feats)
        selected_img_feats = img_voxel_feats[inds_img[:, 0], inds_img[:, 1], inds_img[:, 2], inds_img[:, 3]].contiguous()
        nearest_pts_coords = self.fps_NN_fast(inds_img.contiguous(), inds_pts.contiguous(), fps_num=2048, radius=6, max_cluster_samples=200, dist_thresh=13.3, num=self.knum)
        if self.knum == 1:
            indices_pts = inds_pts[nearest_pts_coords]
            nearest_pts_feats = pts_voxel_feats[indices_pts[:, 0], indices_pts[:, 1], indices_pts[:, 2], indices_pts[:, 3]].contiguous()
        else:
            nearest_pts_feats = []
            for i in range(self.knum):
                indices = inds_img[nearest_pts_coords[i]]
                nearest_pts_feat = pts_voxel_feats[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]].contiguous()
                nearest_pts_feats.append(nearest_pts_feat)
            nearest_pts_feats = torch.cat(nearest_pts_feats, dim=1)
        nearest_pts_feats = self.knn_enc(nearest_pts_feats) * selected_img_feats

        fused_feats_img = torch.zeros(B, H, W, L, C).to(img_voxel_feats.device)
        fused_feats_img[inds_pts[:, 0], inds_pts[:, 1], inds_pts[:, 2], inds_pts[:, 3]] = nearest_img_feats
        

        fused_feats_pts = torch.zeros(B, H, W, L, C).to(img_voxel_feats.device)
        fused_feats_pts[inds_img[:, 0], inds_img[:, 1], inds_img[:, 2], inds_img[:, 3]] = nearest_pts_feats

        # print("fused_pts:", fused_feats_pts.shape, "fused_img:", fused_feats_img.shape, "minmax", fused_feats_img.max(), fused_feats_pts.max())

        all_feats = torch.cat([img_voxel_feats, pts_voxel_feats, fused_feats_img, fused_feats_pts], dim=-1)
        output_feats = self.con_enc(all_feats.permute(0, 4, 1, 2, 3))
        return output_feats
        #! 
        # inds_pts_nn_img = torch.zeros_like(inds_pts[:, 0]).long() - 1
        # # print("inds_pts_nn_img.shape",inds_pts_nn_img.shape)
        # base = 0
        # for idx in range(len(torch.unique(inds_pts[:, 0]))):
        #     mask_pts = inds_pts[:, 0] == idx
        #     mask_img = inds_img[:, 0] == idx
        #     query_pts = inds_pts[mask_pts]
        #     key_img = inds_img[mask_img]
        #     # print("query_pts",query_pts.shape)
        #     key_idx = self.fps_NN_fast(query_pts, key_img, fps_num=2048, radius=6, max_cluster_samples=200, dist_thresh=13.3)
        #     non_empty_mask = (key_idx != -1)
        #     # print("non_mask:", non_empty_mask.shape)
        #     key_idx[non_empty_mask] = key_idx[non_empty_mask] + base
        #     inds_pts_nn_img[mask_pts] = key_idx
        #     base = mask_img.sum()
        #     # print("key_idx:", key_idx, "base:", base, "inds_pts_nn_img:", inds_pts_nn_img.shape

        # img_voxel_feats_nn = img_voxel_feats.clone().permute(0,2,3,4,1)
        # pts_voxel_feats_nn = pts_voxel_feats.clone().permute(0,2,3,4,1)

        # print(inds_img.shape, inds_pts_nn_img.shape, inds_pts)
        # # pts_voxel_feats_sparse = pts_voxel_feats.reshape(-1, B*C)[coord_3d_pts]
        # # print(inds_pts_nn_img.shape, inds_pts[inds_pts_nn_img].shape) # inds_pts[inds_pts_nn_img] img_coord
    
        # index = inds_img[inds_pts_nn_img]

        # img_voxel_feats_nn = img_voxel_feats_nn[index]

        # img_voxel_feats_sparse_nn = self.knn_enc(img_voxel_feats_nn)
        

        # pts_voxel_feats_nn[inds_pts] *= img_voxel_feats_sparse_nn

        # pts_voxel_feats_nn = pts_voxel_feats_nn.reshape(B, C, H, W, L)
        
        # # (pts_voxel_feats_nn.reshape(-1, B*C))[coord_3d_pts] = img_voxel_feats_sparse_nn * (pts_voxel_feats_nn.reshape(-1, B*C))[coord_3d_pts]
        # # print("img_nn:", img_voxel_feats_sparse_nn.shape, "pts_voxel_feats_nn", pts_voxel_feats_nn.shape)
        # voxel_feats = self.con_enc(torch.cat([img_voxel_feats, pts_voxel_feats, pts_voxel_feats_nn], dim=1))

        # return voxel_feats
