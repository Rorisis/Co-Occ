import torch
import torch.nn.functional as F
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
class NeRFOcc_KITTI(BEVDepth):
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
            self.density_encoder = builder.build_neck(density_encoder)
            # self.density_neck = builder.build_neck(density_neck)
            if color_encoder:
                self.color_encoder = builder.build_neck(color_encoder)
                self.color_head = MLP(input_dim=256, output_dim=3,net_depth=3,skip_layer=None)#torch.nn.Linear(256, 3)#MLP(input_dim=256, output_dim=3,net_depth=3,skip_layer=0)
            # self.color_neck = builder.build_neck(color_neck)
            self.density_head = torch.nn.Linear(256, 1)
            
        
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
        
        x, depth, geom = self.img_view_transformer([x] + geo_inputs)

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['view_transformer'].append(t2 - t1)
        
        # x = self.bev_encoder(x)
        # if type(x) is not list:
        #     x = [x]
        
        return x, depth, img_feats, geom

    def get_weights(self, sigma, z_vals):
        sigma = sigma.reshape(-1,1)
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
            x = self.pts_backbone(pts_enc_feats['x'])
        if self.with_pts_neck:
            x = self.pts_neck(x)

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['pts_encoder'].append(t1 - t0)
        
        pts_feats = pts_enc_feats['pts_feats']

        return pts_enc_feats['x'], pts_feats

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

        # print("img_voxels:", img_voxel_feats.shape)
        # print("pts_voxels:", pts_voxel_feats.shape)
        if self.occ_fuser is not None:
            voxel_feats = self.occ_fuser(img_voxel_feats, pts_voxel_feats)
        else:
            assert (img_voxel_feats is None) or (pts_voxel_feats is None)
            voxel_feats = img_voxel_feats if pts_voxel_feats is None else pts_voxel_feats
        # print("voxel_feat:", voxel_feats.shape)

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_fuser'].append(t1 - t0)

        # voxel_feats_enc = self.occ_encoder(voxel_feats)
        # if type(voxel_feats_enc) is not list:
        #     voxel_feats_enc = [voxel_feats_enc]

        # if self.record_time:
        #     torch.cuda.synchronize()
        #     t2 = time.time()
        #     self.time_stats['occ_encoder'].append(t2 - t1)

        return (voxel_feats, img_feats, pts_feats, depth, geom)
    
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
        voxel_feats, img_feats, pts_feats, depth, gemo = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)

        
        mid_voxel = self.semantic_encoder(voxel_feats)
        # for i in range(len(mid_voxel)):
        #     print("mid_voxel:", mid_voxel[i].shape)
        
        semantic_voxel = self.semantic_neck(mid_voxel)
        # for i in range(len(semantic_voxel)):
        #     print("semantic_voxel:", semantic_voxel[i].shape)
        
        
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
        # losses["loss_voxel_ce"] = self.loss_voxel_ce_weights * CE_ssc_loss(semantic_preds, gt_occ, self.class_weights.type_as(semantic_preds), ignore_index=255)
        # losses['loss_voxel_sem'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(semantic_preds, gt_occ, ignore_index=255)
        # losses['loss_voxel_geo'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(semantic_preds, gt_occ, ignore_index=255, non_empty_idx=self.empty_idx)
        # losses['loss_voxel'] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(semantic_preds, dim=1), gt_occ, ignore=255)

        transform = img_inputs[1:] if img_inputs is not None else None
        losses_occupancy = self.forward_pts_train(semantic_voxel, gt_occ,
                        points_occ, img_metas, img_feats=img_feats, pts_feats=pts_feats, transform=transform, 
                        visible_mask=visible_mask)
        losses.update(losses_occupancy)
        if self.use_rendering:
            # rays_d = []
            # rays_o = []
            # rgbs = []
            # depths = []
            # gt_imgs = []
            # gt_depths = []
            
            density_voxel = self.density_encoder(mid_voxel)
            if img_feats:
                color_voxel = self.color_encoder(mid_voxel)
            t_to_s, s_to_t = construct_ray_warps(self.near_far_range[0], self.near_far_range[1], uniform=True)
            if gemo == None:
                gemo = get_frustum(gt_depths[0], gt_depths[1], gt_depths[2], gt_depths[3], gt_depths[4], gt_depths[5], gt_depths[6], self.scale)
            B,N,D,H,W,_ = gemo.shape
            gemo = gemo.reshape(B*N, D, H, W, 3)

            s_vals = sample_along_rays(gemo.shape[0], self.N_samples, randomized=self.training)

            norm_coord_2d, dir_coord_2d = grid_generation(gemo.shape[-3], gemo.shape[-2])
            # print("norm_coord_2d:", norm_coord_2d.shape)
            norm_coord_2d = norm_coord_2d[None, :, :, None, :].repeat(gemo.shape[0], 1, 1, self.N_samples, 1)  # (b, h, w, d, 2)
            sampled_disparity = s_vals[:, :-1][:, None, None, :, None].repeat(1, gemo.shape[-3],
                                                                            gemo.shape[-2], 1, 1)
            norm_coord_frustum = torch.cat([norm_coord_2d, sampled_disparity], dim=-1).cuda()  # (b, h, w, d, 3)
            # print("norm_coord_frustum:", norm_coord_frustum.max(), norm_coord_frustum.min(), norm_coord_frustum.shape)

            density_features = []
            color_features = []
            for i in range(norm_coord_frustum.shape[0]):
                density_feature = F.grid_sample(density_voxel[0].permute(0,1,4,3,2), norm_coord_frustum[i].unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False).permute(0,4,2,3,1)
                if img_feats:
                    color_feature = F.grid_sample(color_voxel[0].permute(0,1,4,3,2), norm_coord_frustum[i].unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False).permute(0,4,2,3,1) # b, h, w, d, c
                    color_features .append(color_feature)
                density_features.append(density_feature)
                
            
            density_features = torch.cat(density_features, dim=0)
            if img_feats:
                color_features = torch.cat(color_features, dim=0)

            density = F.relu(self.density_head(density_features))
            if img_feats:
                rgb = torch.sigmoid(self.color_head(color_features))
            
            directions = []
            if img_feats:
                cam_intrins = img_inputs[3][0]
            else:
                cam_intrins = gt_depths[3][0]
            
            for i in range(gemo.shape[0]):
                if img_feats:
                    dir_coord_2d = dir_coord_2d * img_inputs[0].shape[-1] // gemo.shape[-2] #img: b, n, c, h, w gemo: b, d, h, w, c
                else:
                    dir_coord_2d = dir_coord_2d * gt_depths[-2][0][-1].item() // gemo.shape[-2]
                # print(cam_intrins[i].shape)
                if cam_intrins.shape[-1] == 4:
                    dir_coord_3d = unproject_image_to_rect(dir_coord_2d, torch.cat((cam_intrins[i][:3, :3], torch.zeros(3, 1).to(cam_intrins.device)), dim=1).float())
                else:
                    dir_coord_3d = unproject_image_to_rect(dir_coord_2d, torch.cat((cam_intrins[i], torch.zeros(3, 1).to(cam_intrins.device)), dim=1).float())
                direction = dir_coord_3d[:, :, 1, :] - dir_coord_3d[:, :, 0, :]
                if img_feats:
                    direction /= img_inputs[0].shape[-1] // gemo.shape[-2]
                else:
                    direction /= gt_depths[-2][0][-1].item() // gemo.shape[-2]
                directions.append(direction)

            directions = torch.stack(directions, dim=0)
            weights, tdist = compute_alpha_weights(density, s_vals, directions, s_to_t)
            acc = weights.sum(dim=1)

            # reconstruct depth and rgb image
            t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
            background_rgb = rearrange((1 - acc)[..., None] * torch.tensor([1.0, 1.0, 1.0]).float().cuda(),
                                       'b h w c -> b c h w')
            # print("weights:", weights.shape, "rgb:", rgb.shape, "background_rgb", background_rgb.shape)
            if img_feats:
                rgb_values = torch.sum(weights.unsqueeze(1) * rgb.permute(0,-1,1,2,3), dim=2) + background_rgb
            background_depth = (1 - acc) * torch.tensor([1.0]).float().cuda() * self.near_far_range[1]
            depth_values = (weights * t_mids[..., None, None]).sum(dim=1) + background_depth
            depth_values = depth_values.unsqueeze(1)

            if img_feats:
                rgb_values = F.interpolate(rgb_values, scale_factor=self.scale)
            depth_values = F.interpolate(depth_values, scale_factor=self.scale).squeeze(1)
            # depth_values = self.upsample(depth_values)
            # print("color:", rgb_values.shape, "depth:", depth_values.shape)
            # print(img_inputs[0][0].shape, img_inputs[-7][0].shape)
            
            if img_feats:
                gt_depths = img_inputs[7][0]
                gt_imgs = img_inputs[0][0]
                losses["loss_color"] = F.mse_loss(rgb_values, gt_imgs)
                print(losses["loss_color"].item())
            else: 
                gt_depths = gt_depths[-1][0]

            if self.depth_supervise:
                losses["loss_render_depth"] = 0.0 
                for idx in range(gt_depths.shape[0]):
                    fg_mask = gt_depths[idx] > 0.0
                    target = gt_depths[idx][fg_mask]
                    pred = depth_values[idx][fg_mask]
                    losses["loss_render_depth"] += F.smooth_l1_loss(pred, target, reduction='none').mean()
            if pts_feats != None:
                rendered_opacity = F.interpolate(weights, scale_factor=self.scale).sum(dim=1)
                gt_opacity = (gt_depths != 0).to(gt_depths.dtype)
                losses["loss_opacity"] = torch.mean(-gt_opacity * torch.log(rendered_opacity + 1e-6) - (1 - gt_opacity) * torch.log(1 - rendered_opacity +1e-6)) # BCE loss

                # rand_indices = np.random.choice(range(points[0].shape[0]), self.N_rand)
                # selected_lidar = points[0][rand_indices][:, :3]
                # lidar_density = F.grid_sample(density_voxel[0].permute(0,1,4,3,2), selected_lidar.reshape(1,1,1,self.N_rand,3), mode='bilinear', padding_mode='zeros', align_corners=False)
                # losses['loss_density'] = F.mse_loss(lidar_density.reshape(self.N_rand, -1), 1e2)
            
            
            # for b in range(img_inputs[0].shape[0]):
            #     cam_intrin = img_inputs[-3][b]
            #     c2w = img_inputs[-2][b]
            #     directions = get_ray_direction_with_intrinsics(img_inputs[0].shape[-2], img_inputs[0].shape[-1], cam_intrin)
            #     rays_d, rays_o = get_rays(directions, c2w)
            # # rays_d = torch.stack(rays_d)
            # # rays_o = torch.stack(rays_o)
            #     rays_d = rays_d.reshape(-1, 3) #N, H, W, 3
            #     rays_o = rays_o.reshape(-1, 3)

            #     # print("rays_o:", rays_o.shape, "rays_d:", rays_d.shape)
            #     rand_indices = np.random.choice(range(rays_o.shape[0]), self.N_rand)
            #     rays_o, rays_d = rays_o[rand_indices], rays_d[rand_indices]
            #     gt_img = img_inputs[-5][b].reshape(-1,3)[rand_indices]
            #     gt_depth = img_inputs[-7][b].reshape(-1)[rand_indices]
            #     gt_imgs.append(gt_img)
            #     gt_depths.append(gt_depth)

            #     pts, z_vals = sample_along_camera_ray(ray_o=rays_o,   
            #                                         ray_d=rays_d,
            #                                         depth_range=self.near_far_range,
            #                                         N_samples=self.N_samples,
            #                                         inv_uniform=False,
            #                                         det=False)


            #     aabb = img_inputs[-4][b] # batch size
                
            # #     density_preds = self.density_head(density_voxel[0]) # [H_o, W_o, L_o, 1]
            # #     density_preds = F.relu(density_preds)
            # # # print("density_preds", density_preds.shape)
            #     # print("cor", rays_o[10], rays_d[10], pts[10])
            #     # fig = plt.figure()
            #     # ax = fig.add_subplot()
            #     # pts_ = pts.cpu().numpy()
            #     # # rect = pch.Rectangle(xy=(aabb[0,0], aabb[0,1]), width=aabb[1,0]-aabb[0,0], height=aabb[1,1]-aabb[0,1], fill=False, color='y')
            #     # rect = pch.Rectangle(xy=(aabb[0,1], aabb[0,2]), width=aabb[1,1]-aabb[0,1], height=aabb[1,2]-aabb[0,2], fill=False, color='y')
            #     # ax.add_patch(rect)
            #     # for i in range(20):
            #     #     j=np.random.randint(pts.shape[0])
            #     # # print(pts_[0,:,0], pts_[0,:,1], pts_[0,:,2])
            #     #     ax.scatter(pts_[j,:,1], pts_[j,:,2], color='b')
            #     # # ax.scatter(x, y, z, color='r')
            #     # plt.title('sample pts')
            #     # plt.draw()
            #     # plt.savefig('./pts.png')
            #     # plt.show()


            #     pts = pts.reshape(1, pts.shape[0],pts.shape[1],1,3)
            #     # print("pts:", pts.shape)

            #     aabbSize = aabb[1] - aabb[0]
            #     invgridSize = 1.0/aabbSize * 2
            #     norm_pts = (pts-aabb[0]) * invgridSize - 1
            #     # fig = plt.figure()
            #     # ax = fig.add_subplot(111, projection='3d')
            #     # norm_pts_ = norm_pts.clone().cpu().numpy()
            #     # ax.scatter(norm_pts_[:,0], norm_pts_[:,1], norm_pts_[:,2])
            #     # plt.savefig('./norm.png')
            #     # print("norm_0",norm_pts[:, 0].min(), norm_pts[:, 0].max())
            #     # print("norm_1",norm_pts[:, 1].min(), norm_pts[:, 1].max())
            #     # print("norm_2",norm_pts[:, 2].min(), norm_pts[:, 2].max())
            #     density_feature = F.grid_sample(density_voxel[0][b].unsqueeze(0).permute(0,1,4,3,2), norm_pts, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0).squeeze(-1).permute(1,2,0)
            #     color_feature = F.grid_sample(color_voxel[0][b].unsqueeze(0).permute(0,1,4,3,2), norm_pts, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0).squeeze(-1).permute(1,2,0)

            #     density = F.relu(self.density_head(density_feature))
            #     color = torch.sigmoid(self.color_head(color_feature))
                
            #     weights = self.get_weights(density, z_vals)

            #     color_2d = torch.sum(weights.unsqueeze(2) * color, dim=1)

            #     if self.white_bkgd:
            #         color_2d = color_2d + (1. - torch.sum(weights, dim=-1, keepdim=True))

            #     depth_2d = torch.sum(weights * z_vals, dim=-1) / (torch.sum(weights, dim=-1) + 1e-8)
            #     depth_2d = torch.clamp(depth_2d, z_vals.min(), z_vals.max())
            # # print("depth_2d:", depth_2d.shape)
            #     rgbs.append(color_2d)
            #     depths.append(depth_2d)
            # rgbs = torch.stack(rgbs)
            # depths = torch.stack(depths)
            # gt_imgs = torch.stack(gt_imgs)
            # gt_depths = torch.stack(gt_depths)
    
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
    
    def simple_test(self, img_metas, img=None, gt_depths=None, points=None, rescale=False, points_occ=None, 
            gt_occ=None, visible_mask=None):
        
        voxel_feats, img_feats, pts_feats, depth, gemo = self.extract_feat(points, img=img, img_metas=img_metas)

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
            density_voxel = self.density_encoder(mid_voxel)
            color_voxel = self.color_encoder(mid_voxel)
            t_to_s, s_to_t = construct_ray_warps(self.near_far_range[0], self.near_far_range[1], uniform=True)
            B,N,D,H,W,_ = gemo.shape
            gemo = gemo.reshape(B*N, D, H, W, 3)

            s_vals = sample_along_rays(gemo.shape[0], self.N_samples, randomized=self.training)

            norm_coord_2d, dir_coord_2d = grid_generation(gemo.shape[-3], gemo.shape[-2])
            # print("norm_coord_2d:", norm_coord_2d.shape)
            norm_coord_2d = norm_coord_2d[None, :, :, None, :].repeat(gemo.shape[0], 1, 1, self.N_samples, 1)  # (b, h, w, d, 2)
            sampled_disparity = s_vals[:, :-1][:, None, None, :, None].repeat(1, gemo.shape[-3],
                                                                            gemo.shape[-2], 1, 1)
            norm_coord_frustum = torch.cat([norm_coord_2d, sampled_disparity], dim=-1).cuda()  # (b, h, w, d, 3)
            # print("norm_coord_frustum:", norm_coord_frustum.max(), norm_coord_frustum.min())

            density_features = []
            color_features = []
            for i in range(norm_coord_frustum.shape[0]):
                density_feature = F.grid_sample(density_voxel[0].permute(0,1,4,3,2), norm_coord_frustum[i].unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False).permute(0,4,2,3,1)
                color_feature = F.grid_sample(color_voxel[0].permute(0,1,4,3,2), norm_coord_frustum[i].unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False).permute(0,4,2,3,1) # b, h, w, d, c
                density_features.append(density_feature)
                color_features .append(color_feature)
            
            density_features = torch.cat(density_features, dim=0)
            color_features = torch.cat(color_features, dim=0)

            density = F.relu(self.density_head(density_features))
            rgb = torch.sigmoid(self.color_head(color_features))
            
            directions = []
            cam_intrins = img[3][0]
            
            for i in range(gemo.shape[0]):
                dir_coord_2d = dir_coord_2d * img[0].shape[-1] // gemo.shape[-2] #img: b, n, c, h, w gemo: b, d, h, w, c
                dir_coord_3d = unproject_image_to_rect(dir_coord_2d, torch.cat((cam_intrins[i], torch.zeros(3, 1).to(cam_intrins.device)), dim=1).float())
                direction = dir_coord_3d[:, :, 1, :] - dir_coord_3d[:, :, 0, :]
                direction /= img[0].shape[-1] // gemo.shape[-2]
                directions.append(direction)

            directions = torch.stack(directions, dim=0)
            weights, tdist = compute_alpha_weights(density, s_vals, directions, s_to_t)
            acc = weights.sum(dim=1)

            # reconstruct depth and rgb image
            t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
            background_rgb = rearrange((1 - acc)[..., None] * torch.tensor([1.0, 1.0, 1.0]).float().cuda(),
                                       'b h w c -> b c h w')
            # print("weights:", weights.shape, "rgb:", rgb.shape, "background_rgb", background_rgb.shape)
            rgb_values = torch.sum(weights.unsqueeze(1) * rgb.permute(0,-1,1,2,3), dim=2) + background_rgb
            background_depth = (1 - acc) * torch.tensor([1.0]).float().cuda() * self.near_far_range[1]
            depth_values = (weights * t_mids[..., None, None]).sum(dim=1) + background_depth
            depth_values = depth_values.unsqueeze(1)
    
            gt_imgs = F.interpolate(img[-5][0], scale_factor=0.0625)
            # depth_values = self.upsample(depth_values)
            # print("color:", rgb_values.shape, "depth:", depth_values.shape)
            # print(img_inputs[0][0].shape, img_inputs[-7][0].shape)
            # gt_depths = img[7][0]
            # gt_imgs = img[0][0]

            psnr_total = 0
            for v in range(rgb_values.shape[0]):
                # print("pred:", rgbs[v].var(), "gt:", img[-5][0][v].var())
                # depth_ = ((depth_values[v].unsqueeze(-1)-depth_values[v].unsqueeze(-1).min()) / (depth_values[v].unsqueeze(-1).max() - depth_values[v].unsqueeze(-1).min()+1e-8)).repeat(1, 1, 3)
                
                img_to_save = torch.cat([rgb_values[v].permute(1,2,0), gt_imgs[v].permute(1,2,0)], dim=1).clip(0, 1)
                img_to_save = np.uint8(img_to_save.cpu().numpy()* 255.0)
                psnr = compute_psnr(rgb_values[v].permute(1,2,0), gt_imgs[v].permute(1,2,0), mask=None)
                psnr_total += psnr
                cv2.imwrite("./img_"+str(v)+'.png', img_to_save)
            print("psnr:", psnr_total/rgb_values.shape[0])

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
                hist_occ = fast_hist(pred[mask], gt[mask], max_label=20)

            hist = fast_hist(pred[noise_mask], gt[noise_mask], max_label=20)
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
        