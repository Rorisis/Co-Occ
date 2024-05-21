import torch
import torch.nn.functional as F
import collections 
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import torch.nn as nn

from mmdet.models import DETECTORS
from mmcv.runner import force_fp32
from mmdet3d.models import builder
from einops import rearrange

from projects.mmdet3d_plugin.utils import fast_hist_crop
from .bevdepth import BEVDepth

from projects.mmdet3d_plugin.utils import render_rays, sample_along_camera_ray, get_ray_direction_with_intrinsics, get_rays, sample_along_rays, \
    grid_generation, unproject_image_to_rect, compute_alpha_weights, construct_ray_warps
from projects.mmdet3d_plugin.utils import VanillaNeRFRadianceField, MLP
from projects.mmdet3d_plugin.utils import save_rendered_img, compute_psnr
from projects.mmdet3d_plugin.utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from ..dense_heads.lovasz_softmax import lovasz_softmax
from projects.mmdet3d_plugin.utils import per_class_iu, fast_hist_crop

import numpy as np
import time
import pdb
import cv2
import copy

@DETECTORS.register_module()
class COOCC_Ray_L(BEVDepth):
    def __init__(self, 
                voxel_size,
                n_voxels,
                loss_cfg=None,
                aabb=None,
                near_far_range=None,
                N_samples=40,
                N_rand=4096,
                depth_supervise=False,
                use_nerf_mask=True,
                nerf_sample_view=3,
                nerf_mode='volume',
                squeeze_scale=4,
                rgb_supervise=True,
                nerf_density=False,
                rendering_test=False,
                disable_loss_depth=False,
                empty_idx=0,
                scale=16,
                white_bkgd=True,
                occ_fuser=None,
                occ_encoder_backbone=None,
                occ_encoder_neck=None,
                density_encoder=None,
                color_encoder=None,
                semantic_encoder=None,
                density_neck=None, 
                color_neck=None,
                semantic_neck=None,
                loss_norm=False,
                use_rendering=False,
                loss_voxel_ce_weight=1.0,
                loss_voxel_sem_scal_weight=1.0,
                loss_voxel_geo_scal_weight=1.0,
                loss_voxel_lovasz_weight=1.0,
                test_rendering=False,
                **kwargs):
        super().__init__(**kwargs)
        

        self.loss_cfg = loss_cfg
        self.disable_loss_depth = disable_loss_depth
        self.loss_norm = loss_norm
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
        self.empty_idx = empty_idx
       
        self.occ_fuser = builder.build_fusion_layer(occ_fuser) if occ_fuser is not None else None
        
        self.semantic_encoder = builder.build_backbone(semantic_encoder)
        self.semantic_neck = builder.build_neck(semantic_neck)

        self.scale = scale
        self.voxel_size = voxel_size
        self.n_voxels = n_voxels
        self.aabb = aabb
        self.near_far_range = near_far_range
        self.N_samples = N_samples
        self.N_rand = N_rand
        self.depth_supervise = depth_supervise
        self.use_nerf_mask = use_nerf_mask
        self.nerf_sample_view = nerf_sample_view
        self.nerf_mode = nerf_mode
        self.squeeze_scale = squeeze_scale
        self.rgb_supervise = rgb_supervise
        self.nerf_density = nerf_density
        self.white_bkgd = white_bkgd

        self.loss_voxel_ce_weight = loss_voxel_ce_weight
        self.loss_voxel_sem_scal_weight = loss_voxel_sem_scal_weight
        self.loss_voxel_geo_scal_weight = loss_voxel_geo_scal_weight
        self.loss_voxel_lovasz_weight = loss_voxel_lovasz_weight
        

        self.rendering_test = rendering_test
        self.use_rendering = use_rendering
        self.test_rendering = test_rendering

        if use_rendering:
            self.sigma_head = MLP(input_dim=128, output_dim=1, net_depth=1, skip_layer=None)
            self.rgb_head = MLP(input_dim=128, output_dim=3, net_depth=3, skip_layer=None)
            
        coord_x, coord_y, coord_z = torch.meshgrid(torch.arange(self.n_voxels[0]),torch.arange(self.n_voxels[1]), torch.arange(self.n_voxels[2]))
        self.sample_coordinates = torch.stack([coord_x, coord_y, coord_z], dim=0)
        # self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=16.)
            
    
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return {'x':x,
                'img_feats':[x.clone()]}
    
    @force_fp32()
    def occ_encoder(self, x):
        x = self.occ_encoder_backbone(x)
        x = self.occ_encoder_neck(x)
        return x

    @force_fp32()
    def bev_encoder(self, x):
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        x = self.img_bev_encoder_backbone(x)
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['bev_encoder'].append(t1 - t0)
        
        x = self.img_bev_encoder_neck(x)
        
        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['bev_neck'].append(t2 - t1)
        
        return x
    
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
                
        img_enc_feats= self.image_encoder(img[0])
        x = img_enc_feats['x']
        img_feats = img_enc_feats['img_feats']
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['img_encoder'].append(t1 - t0)

        # img: imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors
        rots, trans, intrins, post_rots, post_trans, bda = img[1:7]
        
        mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
        
        x, depth, geom, volume = self.img_view_transformer([x] + geo_inputs)

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['view_transformer'].append(t2 - t1)
        
        # x = self.bev_encoder(x)
        # if type(x) is not list:
        #     x = [x]
        
        return x, depth, img_feats, geom

    def get_weights(self, sigma, z_vals):
        sigma = sigma.squeeze(-1)
        sigma2alpha = lambda sigma, dists: 1. - torch.exp(-sigma)

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat((dists, dists[:, -1:]), dim=-1)  # [N_rays, N_samples]

        alpha = sigma2alpha(sigma, dists)  # [N_rays, N_samples]

        T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]   # [N_rays, N_samples-1]
        T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)

        weights = alpha * T 

        return weights

    def extract_pts_feat(self, pts):
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        pts_enc_feats = self.pts_middle_encoder(voxel_features, coors, batch_size)
        if self.with_pts_backbone:
            x = self.pts_backbone(pts_enc_feats)
        if self.with_pts_neck:
            x = self.pts_neck(x)

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['pts_encoder'].append(t1 - t0)
        
        pts_feats = [x]

        return x.permute(0,1,4,3,2), pts_feats
    
    def extract_pts_feat(self, pts):
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        pts_enc_feats = self.pts_middle_encoder(voxel_features, coors, batch_size)
        if self.with_pts_backbone:
            x = self.pts_backbone(pts_enc_feats)
        if self.with_pts_neck:
            x = self.pts_neck(x)

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['pts_encoder'].append(t1 - t0)
        
        pts_feats = [x]

        return x.permute(0,1,4,3,2), pts_feats

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_voxel_feats = None
        pts_voxel_feats, pts_feats = None, None
        depth, img_feats, geom = None, None, None
        if img is not None:
            img_voxel_feats, depth, img_feats, geom = self.extract_img_feat(img, img_metas)
        if points is not None:
            pts_voxel_feats, pts_feats = self.extract_pts_feat(points)

        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()


        if self.occ_fuser is not None:
            voxel_feats = self.occ_fuser(img_voxel_feats, pts_voxel_feats)
        else:
            assert (img_voxel_feats is None) or (pts_voxel_feats is None)
            voxel_feats = img_voxel_feats if pts_voxel_feats is None else pts_voxel_feats

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_fuser'].append(t1 - t0)

        return (voxel_feats, img_feats, pts_feats, depth, geom, img_voxel_feats)
    
    @force_fp32(apply_to=('voxel_feats'))
    def forward_pts_train(
            self,
            voxel_feats,
            gt_occ=None,
            points_occ=None,
            img_metas=None,
            transform=None,
            img_feats=None,
            pts_feats=None,
            visible_mask=None,
        ):
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        outs = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            target_points=points_occ,
            transform=transform,
        )
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_head'].append(t1 - t0)
        losses = self.pts_bbox_head.loss(
            output_voxels=outs['output_voxels'],
            output_voxels_fine=outs['output_voxels_fine'],
            output_coords_fine=outs['output_coords_fine'],
            target_voxels=gt_occ,
            target_points=points_occ,
            img_metas=img_metas,
            visible_mask=visible_mask,
        )
        
        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['loss_occ'].append(t2 - t1)
        
        return losses

    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            points_occ=None,
            visible_mask=None,
            gt_depths=None,
            **kwargs,
        ):

        # extract bird-eye-view features from perspective images
        voxel_feats, img_feats, pts_feats, depth, gemo, img_voxel_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
      
        mid_voxel = self.semantic_encoder(voxel_feats)
        semantic_voxel = self.semantic_neck(mid_voxel)

        
        
        # training losses
        losses = dict()
        if self.record_time:        
            torch.cuda.synchronize()
            t0 = time.time()
        
        if not self.disable_loss_depth and depth is not None:
            losses['loss_depth'] = self.img_view_transformer.get_depth_loss(img_inputs[7], depth)
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['loss_depth'].append(t1 - t0)


        transform = img_inputs[1:] if img_inputs is not None else None
        losses_occupancy = self.forward_pts_train(semantic_voxel, gt_occ,
                        points_occ, img_metas, img_feats=img_feats, pts_feats=pts_feats, transform=transform, 
                        visible_mask=visible_mask)
        losses.update(losses_occupancy)
        if self.loss_norm:
            for loss_key in losses.keys():
                if loss_key.startswith('loss'):
                    losses[loss_key] = losses[loss_key] / (losses[loss_key].detach() + 1e-9)

        if self.use_rendering:
            rgbs = []
            depths = []
            gt_imgs = []
            weight = []
            if img_feats is not None:
                B, N, D, H, W, _ = gemo.shape
                assert B == 1
                gemo = gemo.reshape(B * N, D, H, W, 3)
        
                
                for i in range(gemo.shape[0]):
                    geom = gemo[i] # [D, H, W, 3] [112, 24, 80, 3]
                    voxel_feats = voxel_feats.squeeze(0) # [C, X, Y, Z] [128, 128, 128, 16]
                    xbound, ybound, zbound = [-50., 50., 1.], [-50., 50., 1.], [-5., 3., 1.0]
                    # xbound, ybound, zbound = [-51.2, 51.2, 0.8], [-51.2, 51.2, 0.8], [-5., 3., 0.8]
                    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]).to(geom.device)
                    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]]).to(geom.device)
                    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]).to(geom.device)
                    geom = ((geom - (bx - dx / 2.)) / dx)
                    inside_mask = (geom[..., 0] >= 0) & (geom[..., 0] < nx[0]) \
                    & (geom[..., 1] >= 0) & (geom[..., 1] < nx[1]) \
                    & (geom[..., 2] >= 0) & (geom[..., 2] < nx[2]) # [D, H, W, 3]
                    geom[~inside_mask] *= 0
                    
                    D, H, W, _ = geom.shape
                    pts = geom.long().permute(1, 2, 0, 3) # [H, W, D, 3]
                    pts_feature = voxel_feats[:, pts[..., 0], pts[..., 1], pts[..., 2]] # [C, H, W, D]
                    pts_feature = pts_feature.permute(1, 2, 3, 0) # [H, W, D, C]
                    mask = inside_mask.permute(1, 2, 0) # [H, W, D]
                    
                    rgb = self.rgb_head(pts_feature)
                    rgb[~mask] = 0
                    rgb = torch.sigmoid(rgb) # [H, W, D, 3]
                    sigma = self.sigma_head(pts_feature).squeeze(-1) # [H, W, D]
                    sigma = F.relu(sigma)
                    
                    pts = pts.float()
                    dists = torch.norm(pts[:, :, 1:, :] - pts[:, :, :-1, :], dim=-1) # [H, W, D - 1]
                    dists = torch.cat(
                        [dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(dists.device)], 
                        dim=-1
                    ) # [H, W, D]
                    alpha = 1. - torch.exp(-F.relu(sigma * dists)) # [H, W, D]
                    weights = alpha * torch.cumprod(
                        torch.cat(
                            [torch.ones(H, W, 1).to(alpha.device), 1.-alpha + 1e-10], -1
                        ), dim=-1
                    )[:, :, :-1] # [H, W, D]
                    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2) # [H, W, 3]
                    
                    z_vals = torch.linspace(0, D, D).reshape(1, 1, D).to(rgb_map.device)
                    depth_map = torch.sum(weights * z_vals, dim=-1)
                    # depthNet_pred = depth.permute(0, 2, 3, 1).squeeze(0).argmax(-1).float() # [H, W]
                    depth_map = F.interpolate(
                        depth_map.unsqueeze(0).unsqueeze(1), scale_factor=16, mode='bilinear'
                    ).squeeze(1).squeeze(0) # [16 * H, 16 * W]
                    rgb_map = F.interpolate(
                    rgb_map.permute(2, 0, 1).unsqueeze(0), scale_factor=16, mode='bilinear'
                        ).permute(0, 2, 3, 1).squeeze(0)
                    rgbs.append(rgb_map)
                    depths.append(depth_map)
                rgbs = torch.stack(rgbs) # [B, H, W, 3]
                depths = torch.stack(depths)

                depth_gt = img_inputs[7][0] # [16 * H, 16 * W] [B, H, W]
                rgb_gt = img_inputs[0][0].permute(0, 2, 3, 1)
                d_bound = [2., 58., 0.5]
                depth_gt = (depth_gt - (d_bound[0] - d_bound[2] / 2.)) / d_bound[2]
                depth_gt = depth_gt.clip(0, D)
                fg_mask = (depth_gt > 0)
                losses["loss_depth_render"] = F.mse_loss(
                    depths[fg_mask] / D, depth_gt[fg_mask] / D
                )
                
                losses["loss_rgb"] = F.mse_loss(rgbs, rgb_gt)
            
            else:
                gemo = get_frustum(gt_depths[0], gt_depths[1], gt_depths[2], gt_depths[3], gt_depths[4], gt_depths[5], gt_depths[-1], 16)
                B, N, D, H, W, _ = gemo.shape
                assert B == 1
                gemo = gemo.reshape(B * N, D, H, W, 3)
                for i in range(gemo.shape[0]):
                    geom = gemo[i] # [D, H, W, 3] [112, 24, 80, 3]
                    voxel_feats = voxel_feats.squeeze(0) # [C, X, Y, Z] [128, 128, 128, 16]
                    xbound, ybound, zbound = [-50., 50., 1.], [-50., 50., 1.], [-5., 3., 1.0]
                    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]).to(geom.device)
                    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]]).to(geom.device)
                    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]).to(geom.device)
                    geom = ((geom - (bx - dx / 2.)) / dx)
                    inside_mask = (geom[..., 0] >= 0) & (geom[..., 0] < nx[0]) \
                    & (geom[..., 1] >= 0) & (geom[..., 1] < nx[1]) \
                    & (geom[..., 2] >= 0) & (geom[..., 2] < nx[2]) # [D, H, W, 3]
                    geom[~inside_mask] *= 0
                    
                    D, H, W, _ = geom.shape
                    pts = geom.long().permute(1, 2, 0, 3) # [H, W, D, 3]
                    pts_feature = voxel_feats[:, pts[..., 0], pts[..., 1], pts[..., 2]] # [C, H, W, D]
                    pts_feature = pts_feature.permute(1, 2, 3, 0) # [H, W, D, C]
                    mask = inside_mask.permute(1, 2, 0) # [H, W, D]
                    
                    sigma = self.sigma_head(pts_feature).squeeze(-1) # [H, W, D]
                    sigma = F.relu(sigma)
                    
                    pts = pts.float()
                    dists = torch.norm(pts[:, :, 1:, :] - pts[:, :, :-1, :], dim=-1) # [H, W, D - 1]
                    dists = torch.cat(
                        [dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(dists.device)], 
                        dim=-1
                    ) # [H, W, D]
                    alpha = 1. - torch.exp(-F.relu(sigma * dists)) # [H, W, D]
                    weights = alpha * torch.cumprod(
                        torch.cat(
                            [torch.ones(H, W, 1).to(alpha.device), 1.-alpha + 1e-10], -1
                        ), dim=-1
                    )[:, :, :-1] # [H, W, D]
         
                    z_vals = torch.linspace(0, D, D).reshape(1, 1, D).to(alpha.device)
                    depth_map = torch.sum(weights * z_vals, dim=-1)
                    # depthNet_pred = depth.permute(0, 2, 3, 1).squeeze(0).argmax(-1).float() # [H, W]
                    depth_map = F.interpolate(
                        depth_map.unsqueeze(0).unsqueeze(1), scale_factor=16, mode='bilinear'
                    ).squeeze(1).squeeze(0) # [16 * H, 16 * W]
              
                    depths.append(depth_map)
 
                depths = torch.stack(depths)

                depth_gt = gt_depths[6][0] # [16 * H, 16 * W] [B, H, W]

                d_bound = [2., 58., 0.5]
                depth_gt = (depth_gt - (d_bound[0] - d_bound[2] / 2.)) / d_bound[2]
                depth_gt = depth_gt.clip(0, D)
                fg_mask = (depth_gt > 0)
                losses["loss_depth_render"] = F.mse_loss(
                    depths[fg_mask] / D, depth_gt[fg_mask] / D
                )
                
        

    
        
        def logging_latencies():
            # logging latencies
            avg_time = {key: sum(val) / len(val) for key, val in self.time_stats.items()}
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)
            
            print(out_res)
        
        if self.record_time:
            logging_latencies()
        
        return losses
        
    def forward_test(self,
            img_metas=None,
            img_inputs=None,
            **kwargs,
        ):
        
        return self.simple_test(img_metas, img_inputs, **kwargs)
    
    def simple_test(self, img_metas, img=None, gt_depths=None, points=None, rescale=False, points_occ=None, 
            gt_occ=None, visible_mask=None):
        
        voxel_feats, img_feats, pts_feats, depth, gemo, img_voxel_feats = self.extract_feat(points, img=img, img_metas=img_metas)

        mid_voxel = self.semantic_encoder(voxel_feats)
        semantic_voxel = self.semantic_neck(mid_voxel)

        transform = img[1:] if img is not None else None
        output = self.pts_bbox_head(
            voxel_feats=semantic_voxel,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            target_points=points_occ,
            transform=transform,
        )

        pred_c = output['output_voxels'][0]
        SC_metric, _ = self.evaluation_semantic(pred_c, gt_occ, eval_type='SC', visible_mask=visible_mask)
        SSC_metric, SSC_occ_metric = self.evaluation_semantic(pred_c, gt_occ, eval_type='SSC', visible_mask=visible_mask)

        pred_f = None
        SSC_metric_fine = None
        if output['output_voxels_fine'] is not None:
            if output['output_coords_fine'] is not None:
                fine_pred = output['output_voxels_fine'][0]  # N ncls
                fine_coord = output['output_coords_fine'][0]  # 3 N
                pred_f = self.empty_idx * torch.ones_like(gt_occ)[:, None].repeat(1, fine_pred.shape[1], 1, 1, 1).float()
                pred_f[:, :, fine_coord[0], fine_coord[1], fine_coord[2]] = fine_pred.permute(1, 0)[None]
            else:
                pred_f = output['output_voxels_fine'][0]
            SC_metric, _ = self.evaluation_semantic(pred_f, gt_occ, eval_type='SC', visible_mask=visible_mask)
            SSC_metric_fine, SSC_occ_metric_fine = self.evaluation_semantic(pred_f, gt_occ, eval_type='SSC', visible_mask=visible_mask)
        # evaluate nusc lidar-seg
        if output['output_points'] is not None and points_occ is not None:
            output['output_points'] = torch.argmax(output['output_points'][:, 1:], dim=1) + 1
            target_points = torch.cat(points_occ, dim=0)
            output['evaluation_semantic'] = self.simple_evaluation_semantic(output['output_points'], target_points, img_metas)
            output['target_points'] = target_points

        if self.use_rendering and self.test_rendering:
            rgbs = []
            depths = []
            gt_imgs = []
            gt_depths = []
            weight = []
            psnr_total = 0

            B, N, D, H, W, _ = gemo.shape
            assert B == 1
            gemo = gemo.reshape(B * N, D, H, W, 3)
            
            for i in range(gemo.shape[0]):
                geom = gemo[i] # [D, H, W, 3] [112, 24, 80, 3]
                voxel_feats = voxel_feats.squeeze(0) # [C, X, Y, Z] [128, 128, 128, 16]
                xbound, ybound, zbound = [-50., 50., 1.], [-50., 50., 1.], [-5., 3., 1.0]
                # xbound, ybound, zbound = [-51.2, 51.2, 0.8], [-51.2, 51.2, 0.8], [-5., 3., 0.8]
                dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]).to(geom.device)
                bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]]).to(geom.device)
                nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]).to(geom.device)
                geom = ((geom - (bx - dx / 2.)) / dx)
                inside_mask = (geom[..., 0] >= 0) & (geom[..., 0] < nx[0]) \
                & (geom[..., 1] >= 0) & (geom[..., 1] < nx[1]) \
                & (geom[..., 2] >= 0) & (geom[..., 2] < nx[2]) # [D, H, W, 3]
                geom[~inside_mask] *= 0
                
                D, H, W, _ = geom.shape
                pts = geom.long().permute(1, 2, 0, 3) # [H, W, D, 3]
                pts_feature = voxel_feats[:, pts[..., 0], pts[..., 1], pts[..., 2]] # [C, H, W, D]
                pts_feature = pts_feature.permute(1, 2, 3, 0) # [H, W, D, C]
                mask = inside_mask.permute(1, 2, 0) # [H, W, D]
                
                rgb = self.rgb_head(pts_feature)
                rgb[~mask] = 0
                rgb = torch.sigmoid(rgb) # [H, W, D, 3]
                sigma = self.sigma_head(pts_feature).squeeze(-1) # [H, W, D]
                sigma = F.relu(sigma)

                pts = pts.float()
                dists = torch.norm(pts[:, :, 1:, :] - pts[:, :, :-1, :], dim=-1) # [H, W, D - 1]
                dists = torch.cat(
                    [dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(dists.device)], 
                    dim=-1
                ) # [H, W, D]
                alpha = 1. - torch.exp(-F.relu(sigma * dists)) # [H, W, D]
                weights = alpha * torch.cumprod(
                    torch.cat(
                        [torch.ones(H, W, 1).to(alpha.device), 1.-alpha + 1e-10], -1
                    ), dim=-1
                )[:, :, :-1] # [H, W, D]
                rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2) # [H, W, 3]
                
                z_vals = torch.linspace(0, D, D).reshape(1, 1, D).to(rgb_map.device)
                depth_map = torch.sum(weights * z_vals, dim=-1)
                # depthNet_pred = depth.permute(0, 2, 3, 1).squeeze(0).argmax(-1).float() # [H, W]
                depth_map = F.interpolate(
                    depth_map.unsqueeze(0).unsqueeze(1), scale_factor=16, mode='bilinear'
                ).squeeze(1).squeeze(0) # [16 * H, 16 * W]
                rgb_map = F.interpolate(
                rgb_map.permute(2, 0, 1).unsqueeze(0), scale_factor=16, mode='bilinear'
                    ).permute(0, 2, 3, 1).squeeze(0)
                rgbs.append(rgb_map)
                depths.append(depth_map)

            rgbs = torch.stack(rgbs)
            depths = torch.stack(depths)
            gt_img = img[0][0]

            for v in range(rgbs.shape[0]):
                depth_ = ((depths[v]-depths[v].min()) / (depths[v].max() - depths[v].min()+1e-8)).unsqueeze(-1).repeat(1, 1, 3)
                img_to_save = torch.cat([rgbs[v], gt_img[v].permute(1,2,0), depth_], dim=1).clip(0, 1)
                img_to_save = np.uint8(img_to_save.cpu().numpy()* 255.0)
                psnr = compute_psnr(rgbs[v], gt_img[v].permute(1,2,0), mask=None)
                psnr_total += psnr
                cv2.imwrite("./img_"+str(v)+'.png', img_to_save)
            print("psnr:", psnr_total/rgbs.shape[0])

        test_output = {
            'SC_metric': SC_metric,
            'SSC_metric': SSC_metric,
            'pred_c': pred_c,
            'pred_f': pred_f,
            'output_voxels': pred_c,
            'target_voxels': gt_occ,
        }
        
        if output['output_points'] is not None and points_occ is not None:
            test_output['output_points'] = output['output_points']
            test_output['target_points'] = output['target_points']
            test_output['evaluation_semantic'] = output['evaluation_semantic']
            
        if SSC_metric_fine is not None:
            test_output['SSC_metric_fine'] = SSC_metric_fine

        return test_output


    def evaluation_semantic(self, pred, gt, eval_type, visible_mask=None):
        _, H, W, D = gt.shape
        pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        pred = torch.argmax(pred[0], dim=0).cpu().numpy()
        gt = gt[0].cpu().numpy()
        gt = gt.astype(np.int)

        # ignore noise
        noise_mask = gt != 255

        if eval_type == 'SC':
            # 0 1 split
            gt[gt != self.empty_idx] = 1
            pred[pred != self.empty_idx] = 1
            return fast_hist(pred[noise_mask], gt[noise_mask], max_label=2), None


        if eval_type == 'SSC':
            hist_occ = None
            if visible_mask is not None:
                visible_mask = visible_mask[0].cpu().numpy()
                mask = noise_mask & (visible_mask!=0)
                hist_occ = fast_hist(pred[mask], gt[mask], max_label=17)

            hist = fast_hist(pred[noise_mask], gt[noise_mask], max_label=17)
            return hist, hist_occ

    def post_process_semantic(self, pred_occ):
        if type(pred_occ) == list:
            pred_occ = pred_occ[-1]
        
        score, clses = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
        return clses

    def simple_evaluation_semantic(self, pred, gt, img_metas):
        # pred = torch.argmax(pred, dim=1).cpu().numpy()
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()
        gt = gt[:, 3].astype(np.int)
        unique_label = np.arange(16)
        hist = fast_hist_crop(pred, gt, unique_label)
        
        return hist

    def forward_dummy(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            points_occ=None,
            **kwargs,
        ):

        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(points, img=img_inputs, img_metas=img_metas)

        transform = img_inputs[1:] if img_inputs is not None else None
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )
        
        return output
    
    
def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)

def get_frustum(rots, trans, intrins, post_rots, post_trans, bda, input_size, scale):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape
        ogfH, ogfW = input_size[0].item(), input_size[1].item()
        
        fH, fW = ogfH // scale, ogfW // scale
        ds = torch.arange(2.0, 58.0, 0.5, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        
        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        frustum = torch.nn.Parameter(frustum, requires_grad=False).to(post_trans.device)

        # undo post-transformation
        # B x N x D x H x W x 3
        points = frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        
        if intrins.shape[3] == 4: # for KITTI
            shift = intrins[:, :, :3, 3]
            points = points - shift.view(B, N, 1, 1, 1, 3, 1)
            intrins = intrins[:, :, :3, :3]
        
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        
        
        if bda.shape[-1] == 4:
            points = torch.cat((points, torch.ones(*points.shape[:-1], 1).type_as(points)), dim=-1)
            points = bda.view(B, 1, 1, 1, 1, 4, 4).matmul(points.unsqueeze(-1)).squeeze(-1)
            points = points[..., :3]
        else:
            points = bda.view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        
        return points

class Density(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)

class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, beta=0.1, beta_min=0.0001):
        super().__init__(beta=beta)
        self.beta_min = torch.tensor(beta_min).cuda()

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta

