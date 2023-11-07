import torch
import torch.nn.functional as F
import torch.nn as nn
import collections 
import matplotlib.pyplot as plt
import matplotlib.patches as pch

from mmdet.models import DETECTORS
from mmcv.runner import force_fp32
from mmdet3d.models import builder
from einops import rearrange

from projects.mmdet3d_plugin.utils import fast_hist_crop
from .bevdepth import BEVDepth

from projects.mmdet3d_plugin.utils import render_rays, sample_along_camera_ray, get_ray_direction_with_intrinsics, get_rays, sample_along_rays, \
    grid_generation, unproject_image_to_rect, compute_alpha_weights, construct_ray_warps
from projects.mmdet3d_plugin.utils import VanillaNeRFRadianceField, MLP, ResnetFC, SSIM, NerfMLP
from projects.mmdet3d_plugin.utils import save_rendered_img, compute_psnr
from projects.mmdet3d_plugin.utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from ..dense_heads.lovasz_softmax import lovasz_softmax
from projects.mmdet3d_plugin.utils import per_class_iu, fast_hist_crop

import numpy as np
import time
import pdb
import cv2
import copy
import pandas as pd
# import mayavi.mlab as mlab

@DETECTORS.register_module()
class MoEOccupancyScale_Test(BEVDepth):
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
        # self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        # self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        self.occ_fuser = builder.build_fusion_layer(occ_fuser) if occ_fuser is not None else None
        
        # self.semantic_encoder = builder.build_backbone(semantic_encoder)
        # self.semantic_neck = builder.build_neck(semantic_neck)

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

        self.ssim = SSIM(size_average=True).cuda()
        

        self.rendering_test = rendering_test
        self.use_rendering = use_rendering
        self.test_rendering = test_rendering

        if use_rendering:
            # self.density_encoder = builder.build_neck(density_encoder)
            # self.density_neck = builder.build_neck(density_neck)

                # self.color_encoder = builder.build_neck(color_encoder)
            # layer1 = torch.nn.Linear(192, 192)
            # layer2 = torch.nn.Linear(192, 192)
            # layer3 = torch.nn.Linear(192, 3)
            num_c = 192
            self.color_head = MLP(input_dim=128, output_dim=3, net_depth=4, skip_layer=None)
            self.color_head2 = MLP(input_dim=128, output_dim=3, net_depth=4, skip_layer=None)
            # self.color_head = torch.nn.Linear(128, 3)
            # self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
            # self.color_head = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)#torch.nn.Linear(256, 3)#MLP(input_dim=256, output_dim=3,net_depth=3,skip_layer=0)
            # self.color_head = ResnetFC(
            # d_in=128,
            # d_out=3,
            # n_blocks=3,
            # d_hidden=512,
            # d_latent=0)
            # self.color_head = VanillaNeRFRadianceField(
            # net_depth=4,  # The depth of the MLP.
            # net_width=256,  # The width of the MLP.
            # skip_layer=None,  # The layer to add skip layers to.
            # feature_dim=128, # + RGB original img
            # net_depth_condition=1,  # The depth of the second part of MLP.
            # net_width_condition=128
            # )
            # self.color_head2 = torch.nn.Linear(3, 3)
            # self.color_neck = builder.build_neck(color_neck)
            # self.density_head = torch.nn.Linear(128, 1)
            # self.density = LaplaceDensity(beta=0.01)
            # self.sdf_conv = nn.Sequential(
            #     nn.Conv3d(num_c, num_c, 3, 1, 1, bias=True),
            #     nn.Softplus(beta=100),
            #     nn.Conv3d(num_c, num_c, 3, 1, 1, bias=True),
            #     nn.Softplus(beta=100),
            #     nn.Conv3d(num_c, num_c + 1, 3, 1, 1, bias=True),
            # )
            # self.rgb_conv = nn.Sequential(
            #     nn.Conv3d(num_c, 3, 3, 1, 1, bias=False),
            #     nn.Sigmoid(),
            # )
    
        coord_x, coord_y, coord_z = torch.meshgrid(
            torch.arange(self.n_voxels[0]),
            torch.arange(self.n_voxels[1]), 
            torch.arange(self.n_voxels[2])
        )
        self.sample_coordinates = torch.stack([coord_x, coord_y, coord_z], dim=0)
    
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, output_H, output_W = x.shape
        x = x.view(B, N, output_dim, output_H, output_W)
        
        return x
    
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
                
        x = self.image_encoder(img[0]) # [B, N, C, 128, 128]
        img_feats = x.clone()
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['img_encoder'].append(t1 - t0)

        # img: imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors
        rots, trans, intrins, post_rots, post_trans, bda = img[1:7]
        
        mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
        
        x, depth, geom, volume = self.img_view_transformer([x] + geo_inputs)
        # x: [B, C, H, W, D]
        # depth: [B, N, H, W, D]
        # geom: [B, N, D, H, W, 3]
        # volume: [B, N, D, H, W, C]

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['view_transformer'].append(t2 - t1)
        
        # x = self.bev_encoder(x)
        # if type(x) is not list:
        #     x = [x]
        
        return x, depth, img_feats, geom, volume

    def extract_pts_feat(self, pts):
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        pts_enc_feats = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = pts_enc_feats['x']
        # print(pts_enc_feats.shape)
        if self.with_pts_backbone:
            x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['pts_encoder'].append(t1 - t0)
        
        pts_feats = pts_enc_feats['pts_feats']

        return x, pts_feats

    # def extract_pts_feat(self, pts):
    #     if self.record_time:
    #         torch.cuda.synchronize()
    #         t0 = time.time()
    #     voxels, num_points, coors = self.voxelize(pts)
    #     voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
    #     batch_size = coors[-1, 0] + 1
    #     pts_enc_feats = self.pts_middle_encoder(voxel_features, coors, batch_size)
    #     if self.with_pts_backbone:
    #         x = self.pts_backbone(pts_enc_feats)
    #     if self.with_pts_neck:
    #         x = self.pts_neck(x)

    #     if self.record_time:
    #         torch.cuda.synchronize()
    #         t1 = time.time()
    #         self.time_stats['pts_encoder'].append(t1 - t0)
        
    #     pts_feats = [x]

    #     return x.permute(0,1,4,3,2), pts_feats

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_voxel_feats = None
        pts_voxel_feats, pts_feats = None, None
        depth, img_feats, geom = None, None, None
        if img is not None:
            img_voxel_feats, depth, img_feats, geom, volume = self.extract_img_feat(img, img_metas)
        if points is not None:
            pts_voxel_feats, pts_feats = self.extract_pts_feat(points)

        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()

        if self.occ_fuser is not None:
            voxel_feats1 = self.occ_fuser(img_voxel_feats, pts_voxel_feats)
        else:
            assert (img_voxel_feats is None) or (pts_voxel_feats is None)
            voxel_feats1 = img_voxel_feats if pts_voxel_feats is None else pts_voxel_feats

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_fuser'].append(t1 - t0)

        voxel_feats = self.bev_encoder(voxel_feats1)

        if type(voxel_feats) is not list:
            voxel_feats = [voxel_feats]

        # voxel_feats_enc = self.occ_encoder(voxel_feats)
        # if type(voxel_feats_enc) is not list:
        #     voxel_feats_enc = [voxel_feats_enc]

        # if self.record_time:
        #     torch.cuda.synchronize()
        #     t2 = time.time()
        #     self.time_stats['occ_encoder'].append(t2 - t1)

        return (voxel_feats, img_feats, pts_feats, depth, geom, volume, img_voxel_feats)
    
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
    
    @force_fp32(apply_to=('pts_feats'))
    def forward_pts_train(
            self,
            pts_feats,
            gt_occ=None,
            points_occ=None,
            img_metas=None,
            img_feats=None,
            points_uv=None,
            **kwargs,
        ):
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        losses = self.pts_bbox_head.forward_train(
            voxel_feats=pts_feats,
            img_metas=img_metas,
            gt_occ=gt_occ,
            points=points_occ,
            img_feats=img_feats,
            points_uv=points_uv,
            **kwargs,
        )
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['mask2former_head'].append(t1 - t0)
        
        return losses
    
    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            points_occ=None,
            points_uv=None,
            **kwargs,
        ):

        # extract bird-eye-view features from perspective images
        voxel_feats, img_feats, pts_feats, depth, gemo, volume, img_voxel_feat = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas
        )
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
        losses_occupancy = self.forward_pts_train(voxel_feats, gt_occ, 
                        points_occ, img_metas, img_feats=img_feats, points_uv=points_uv, **kwargs)
        losses.update(losses_occupancy)

        if self.use_rendering:
            rgbs = []

            B, N, D, H, W, _ = gemo.shape
            assert B == 1
            gemo = gemo.reshape(B * N, D, H, W, 3)
            volume = F.interpolate(
                volume.permute(0, 3, 1, 2), 
                scale_factor=16
            ).permute(0, 2, 3, 1)
            
            geom = gemo.long().squeeze(0) # [D, H, W, 3] [112, 24, 80, 3]
            img_voxel_feat = img_voxel_feat.squeeze(0) # [C, X, Y, Z] [128, 128, 128, 16]
            
            xbound, ybound, zbound = [0, 51.2, 0.4], [-25.6, 25.6, 0.4], [-2, 4.4, 0.4]
            dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
            bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
            geom = ((geom - (bx - dx / 2.)) / dx).long()
            
            C, X, Y, Z = img_voxel_feat.shape
            D, H, W, _ = geom.shape
            color_feature = img_voxel_feat[:, geom[..., 0], geom[..., 1], geom[..., 2]] # [C, D, H, W]
            # print('img_voxel_feat', img_voxel_feat.shape)
            # print('geom', geom.shape)
            # print('color_feature', color_feature.shape)
            # raise ValueError('')
            
            color_feature_2d = color_feature.sum(dim=1).permute(1, 2, 0) # [H, W, C]
            rgbs = torch.sigmoid(self.color_head2(color_feature_2d)) # [H, W, 3]
            rgbs = rgbs.unsqueeze(0) # [1, H, W, 3]
            rgbs = F.interpolate(
                rgbs.permute(0, 3, 1, 2), scale_factor=16, mode='bilinear'
            ).permute(0, 2, 3, 1)
            losses["loss_rgb"] = F.mse_loss(
                rgbs, img_inputs[0][0].permute(0, 2, 3, 1)
            )
            
            gt_vis = img_inputs[0][0][0].permute(1, 2, 0).cpu().numpy()
            pred_vis = rgbs[0].detach().cpu().numpy()
            vis = np.concatenate((gt_vis, pred_vis), axis=1)
            vis = (vis * 255).astype(np.uint8)
            cv2.imwrite("./vis_save/" + str(time.time()) + 'vis.png', vis)
            print("rgb_loss: ", losses["loss_rgb"].item())
            
            # for b in range(gemo.shape[0]):
            #     geom_b = gemo[b, ...].long()            
            #     color_feature = img_voxel_feat[..., geom_b[..., 1], geom_b[..., 0], geom_b[..., 2]]
            #     color_feature = color_feature.permute(0, 2, 3, 4, 1).squeeze(0)
            #     print(color_feature.shape)
            #     color_2d = torch.sigmoid(self.color_head2(color_feature.sum(0)))
                
            #     rgb_volume = torch.sigmoid(self.color_head(volume[b]))

            #     rgbs.append(color_2d)
            #     rgb_volumes.append(rgb_volume)
            # rgbs = torch.stack(rgbs)
            # rgb_volumes = torch.stack(rgb_volumes)

            # rgbs = F.interpolate(
            #     rgbs.permute(0, 3, 1, 2), scale_factor=16
            # ).permute(0, 2, 3, 1)
            
            # gt_vis = np.uint8(img_inputs[0][0][0].permute(1,2,0).cpu().numpy()* 255.0)
            # volume_vis = np.uint8(rgb_volumes[0].detach().cpu().numpy()* 255.0)
            # rgbs_vis = np.uint8(rgbs[0].detach().cpu().numpy()* 255.0)
            # vis = np.concatenate((gt_vis, rgbs_vis, volume_vis), axis=1)
            # cv2.imwrite("./vis_save/"+str(time.time())+'vis.png', vis)
   
            # losses["loss_color"] = F.mse_loss(
            #     rgbs, img_inputs[0][0].permute(0, 2, 3, 1)
            # )
            # losses["loss_rgb"] = F.mse_loss(
            #     rgb_volume, img_inputs[0][0].permute(0, 2, 3, 1)
            # )

            # print("color:", losses['loss_color'].item(), "rgb:", losses["loss_rgb"].item())


        if self.loss_norm:
            for loss_key in losses.keys():
                if loss_key.startswith('loss'):
                    losses[loss_key] = losses[loss_key] / (losses[loss_key].detach() + 1e-9)

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
    
    def simple_test(self, img_metas, img=None, points=None, rescale=False, points_occ=None, 
            gt_occ=None, visible_mask=None, points_uv=None):
        
        voxel_feats, img_feats, pts_feats, depth, gemo, volume = self.extract_feat(points, img=img, img_metas=img_metas)       
        output = self.pts_bbox_head.simple_test(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            points_uv=points_uv,
        )

        if self.use_rendering and self.test_rendering:
            t_to_s, s_to_t = construct_ray_warps(self.near_far_range[0], self.near_far_range[1], uniform=True)
            B,N,D,H,W,_ = gemo.shape
            assert B == 1
            gemo = gemo.reshape(B*N, D, H, W, 3)

            rays_o_all = gemo[:, 0, : , :, :]
            rays_d_all = gemo[:, 1, : , :, :] - gemo[:, 0, : , :, :]
            magnitude = torch.norm(rays_d_all, dim=-1, keepdim=True)  # calculate magnitude
            rays_d_all = rays_d_all / magnitude

            sdf_output = self.sdf_conv(voxel_feats[0])
            sdf = sdf_output[:,:1]
            feature_vectors = sdf_output[:,1:]

            rgb = self.rgb_conv(feature_vectors)
            density_ = self.density(sdf)
            volume = F.interpolate(volume.permute(0,3,1,2), scale_factor=16).permute(0,2,3,1)

            rgbs_part = []
            depths_part = []
            rgbs = []
            depths = []
            rgb_volumes = []
            for b in range(rays_o_all.shape[0]):
                rays_d = rays_d_all[b].reshape(-1, 3) #N, H, W, 3
                rays_o = rays_o_all[b].reshape(-1, 3)
                # print("rays_o:", rays_o.shape, "rays_d:", rays_d.shape)
                for i in range(0, rays_o.shape[0], self.N_rand):
                    rays_o_chunck = rays_o[i: i+self.N_rand]
                    rays_d_chunck = rays_d[i: i+self.N_rand]
                    pts, z_vals = sample_along_camera_ray(ray_o=rays_o_chunck,   
                                                    ray_d=rays_d_chunck,
                                                    depth_range=self.near_far_range,
                                                    N_samples=self.N_samples,
                                                    inv_uniform=False,
                                                    det=False)

                    # xyz_min = torch.tensor(np.array([0, -25.6, -2])).to(ray_o.device)
                    # xyz_max = torch.tensor(np.array([51.2, 25.6, 4.4])).to(ray_o.device)
                    # xyz_range = (xyz_max - xyz_min).float()
                    # radius = 39
                    # scene_center = (xyz_min + xyz_max) * 0.5
                    # scene_radius = torch.tensor(np.array([radius, radius, radius])).to(ray_o.device)
                    # bg_len = (xyz_range[0]//2-radius)/radius
                    # world_len = 256
                    # step_size = 0.5

                    # ray_o = (ray_o - scene_center) / scene_radius       # normalization
                    # ray_d = ray_o / ray_d.norm(dim=-1, keepdim=True)
                    # N_inner = int(2 / (2+2*bg_len) * world_len / step_size) + 1
                    # N_outer = N_inner//15   # hardcode: 15
                    # b_inner = torch.linspace(0, 2, N_inner+1)
                    # b_outer = 2 / torch.linspace(1, 1/64, N_outer+1)
                    # z_vals = torch.cat([
                    #     (b_inner[1:] + b_inner[:-1]) * 0.5,
                    #     (b_outer[1:] + b_outer[:-1]) * 0.5,
                    # ]).to(ray_o)
                    # ray_pts = ray_o[:,None,:] + ray_d[:,None,:] * z_vals[None,:,None]

                    # norm = ray_pts.norm(dim=-1, keepdim=True)
                    # inner_mask = (norm<=1)
                    # norm_pts = torch.where(
                    #     inner_mask,
                    #     ray_pts,
                    #     ray_pts / norm * ((1+bg_len) - bg_len/norm)
                    # )

                    # # reverse bda-aug 
                    # bda = img_inputs[6][0]
                    # norm_pts = bda[:3,:3].matmul(norm_pts.float().unsqueeze(-1)).squeeze(-1)

                    aabb = torch.tensor([[0, -25.6, -2], [51.2, 25.6, 4.4]]).to(pts.device)

                    pts = pts.reshape(1, pts.shape[0],pts.shape[1],1,3)

                    aabbSize = aabb[1] - aabb[0]
                    invgridSize = 1.0/aabbSize * 2
                    norm_pts = (pts-aabb[0]) * invgridSize - 1
                    # print(norm_pts.shape, norm_pts.max(), norm_pts.min())

                    density = F.grid_sample(density_.permute(0,1,4,3,2), norm_pts, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(0).squeeze(-1).permute(1,2,0)
                    color = F.grid_sample(rgb.permute(0,1,4,3,2), norm_pts, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(0).squeeze(-1).permute(1,2,0)
                    # print("color_features:", color_feature.shape)
                    # mlp_out = self.color_head(color_feature)
                    # density = F.relu(self.density_head(density_feature))
                    # color = torch.sigmoid(self.color_head(color_feature))
                    
                    weights = self.get_weights(density, z_vals)
                    
                    color_2d = torch.sum(weights.unsqueeze(2) * color, dim=1)
                    # color_2d = torch.sigmoid(torch.sum(color, dim=1))
                    

                    if self.white_bkgd:
                        color_2d = color_2d + (1. - torch.sum(weights, dim=-1, keepdim=True))

                    depth_2d = torch.sum(weights * z_vals, dim=-1) / (torch.sum(weights, dim=-1) + 1e-8)
                    depth_2d = torch.clamp(depth_2d, z_vals.min(), z_vals.max())
                    rgbs.append(color_2d)
                    depths.append(depth_2d)
                rgb_volume = torch.sigmoid(self.color_head(volume[b]))
                rgb_volumes.append(rgb_volume)

            rgbs = torch.cat(rgbs, dim=0).view(self.nerf_sample_view,H,W,3)
            depths = torch.cat(depths, dim=0).view(self.nerf_sample_view,H,W,1)
            rgb_volumes = torch.stack(rgb_volumes, dim=0)

            psnr_total = 0
            for v in range(rgbs.shape[0]):
                gt_img = img[0][0][v].permute(1,2,0)
                # print("pred:", rgbs[v].var(), "gt:", img[-5][0][v].var())
                depth_ = ((depths[v]-depths[v].min()) / (depths[v].max() - depths[v].min()+1e-8)).repeat(1, 1, 3)
                img_to_save = torch.cat([rgbs[v], rgb_volumes[v], gt_img, depth_], dim=1).clip(0, 1)
                img_to_save = np.uint8(img_to_save.cpu().numpy()* 255.0)
                psnr = compute_psnr(rgbs[v], gt_img, mask=None)
                psnr_total += psnr
                cv2.imwrite("./img_"+str(v)+'.png', img_to_save)
                # plt.imsave("./img_"+str(v)+'.png', img_to_save.cpu().numpy())
            print("psnr:", psnr_total/rgbs.shape[0])
    
        # evaluate nusc lidar-seg
        if output['output_points'] is not None and points_occ is not None:
            output['output_points'] = torch.argmax(output['output_points'][:, 1:], dim=1) + 1
            target_points = torch.cat(points_occ, dim=0)
            output['evaluation_semantic'] = self.simple_evaluation_semantic(output['output_points'], target_points, img_metas)
            output['target_points'] = target_points
        
        # evaluate voxel 
        output_voxels = output['output_voxels'][0]
        target_occ_size = img_metas[0]['occ_size']
        
        if (output_voxels.shape[-3:] != target_occ_size).any():
            output_voxels = F.interpolate(output_voxels, size=tuple(target_occ_size), 
                            mode='trilinear', align_corners=True)
        
        output['output_voxels'] = output_voxels
        output['target_voxels'] = gt_occ
        
        return output


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


# @DETECTORS.register_module()
# class OccupancyFormer4D(OccupancyFormer):
#     def prepare_voxel_feat(self, img, rot, tran, intrin, 
#                 post_rot, post_tran, bda, mlp_input):
        
#         x = self.image_encoder(img)
#         img_feats = x.clone()
        
#         voxel_feat, depth = self.img_view_transformer([x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        
#         return voxel_feat, depth, img_feats

#     def extract_img_feat(self, img, img_metas):
#         inputs = img
#         """Extract features of images."""
#         B, N, _, H, W = inputs[0].shape
#         N = N//2
#         imgs = inputs[0].view(B, N, 2, 3, H, W)
#         imgs = torch.split(imgs, 1, 2)
#         imgs = [t.squeeze(2) for t in imgs]
#         rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
#         extra = [rots.view(B, 2, N, 3, 3),
#                  trans.view(B, 2, N, 3),
#                  intrins.view(B, 2, N, 3, 3),
#                  post_rots.view(B, 2, N, 3, 3),
#                  post_trans.view(B, 2, N, 3)]
#         extra = [torch.split(t, 1, 1) for t in extra]
#         extra = [[p.squeeze(1) for p in t] for t in extra]
#         rots, trans, intrins, post_rots, post_trans = extra
#         voxel_feat_list = []
#         img_feat_list = []
#         depth_list = []
#         key_frame = True # back propagation for key frame only
        
#         for img, rot, tran, intrin, post_rot, \
#             post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
                
#             mlp_input = self.img_view_transformer.get_mlp_input(
#                 rots[0], trans[0], intrin,post_rot, post_tran, bda)
#             inputs_curr = (img, rot, tran, intrin, post_rot, post_tran, bda, mlp_input)
#             if not key_frame:
#                 with torch.no_grad():
#                     voxel_feat, depth, img_feats = self.prepare_voxel_feat(*inputs_curr)
#             else:
#                 voxel_feat, depth, img_feats = self.prepare_voxel_feat(*inputs_curr)
            
#             voxel_feat_list.append(voxel_feat)
#             img_feat_list.append(img_feats)
#             depth_list.append(depth)
#             key_frame = False
        
#         voxel_feat = torch.cat(voxel_feat_list, dim=1)
#         x = self.bev_encoder(voxel_feat)
#         if type(x) is not list:
#             x = [x]

#         return x, depth_list[0], img_feat_list[0]
        