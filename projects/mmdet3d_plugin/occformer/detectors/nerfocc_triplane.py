import torch
import torch.nn.functional as F
import collections 
import matplotlib.pyplot as plt
import matplotlib.patches as pch

from mmdet.models import DETECTORS
from mmcv.runner import force_fp32
from mmdet3d.models import builder

from projects.mmdet3d_plugin.utils import fast_hist_crop
from .bevdepth import BEVDepth

from projects.mmdet3d_plugin.utils import render_rays, sample_along_camera_ray, get_ray_direction_with_intrinsics, get_rays
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
class NeRFOcc_Triplane(BEVDepth):
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
            self.color_encoder = builder.build_neck(color_encoder)
            # self.color_neck = builder.build_neck(color_neck)
            self.density_head = torch.nn.Linear(256, 1)
            self.color_head = MLP(input_dim=256, output_dim=3,net_depth=3,skip_layer=1)#torch.nn.Linear(256, 3)#MLP(input_dim=256, output_dim=3,net_depth=3,skip_layer=0)
        
        coord_x, coord_y, coord_z = torch.meshgrid(torch.arange(self.n_voxels[0]),torch.arange(self.n_voxels[1]), torch.arange(self.n_voxels[2]))
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
        
        x, depth = self.img_view_transformer([x] + geo_inputs)

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['view_transformer'].append(t2 - t1)
        
        # x = self.bev_encoder(x)
        # if type(x) is not list:
        #     x = [x]
        
        return x, depth, img_feats

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
        depth, img_feats = None, None
        if img is not None:
            img_voxel_feats, depth, img_feats = self.extract_img_feat(img, img_metas)
        if points is not None:
            pts_voxel_feats, pts_feats = self.extract_pts_feat(points)

        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()

        # print("img_voxels:", img_voxel_feats.shape)
        # print("pts_voxels:", pts_voxel_feats.shape)
        if self.occ_fuser is not None:
            plane_xy, plane_yz, plane_xz, voxel_x, voxel_y, voxel_z = self.occ_fuser(img_voxel_feats, pts_voxel_feats)
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

        return (plane_xy, plane_yz, plane_xz, voxel_x, voxel_y, voxel_z, img_feats, pts_feats, depth)
    
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
    
    def aggregator(self, plane_xy, plane_yz, plane_xz):
        semantic_voxel = []
        for i in range(len(plane_xy)):
            plane_xy_ = plane_xy[i].unsqueeze(-1).expand(-1,-1,-1,-1,plane_xz[i].shape[-1])
            plane_yz_= plane_yz[i].unsqueeze(-3).expand(-1,-1,plane_xy[i].shape[-2],-1,-1)
            plane_xz_ = plane_xz[i].unsqueeze(-2).expand(-1,-1,-1,plane_xy[i].shape[-1],-1)
            # print("plane_xy", plane_xy_.shape, "plane_yz", plane_yz_.shape, "plane_xz", plane_xz_.shape)
            fused = plane_xy_ + plane_yz_ + plane_xz_
            # print("fused:", fused.shape)
            semantic_voxel.append(fused)
        return semantic_voxel

    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            points_occ=None,
            visible_mask=None,
            **kwargs,
        ):

        # extract bird-eye-view features from perspective images
        plane_xy, plane_yz, plane_xz, voxel_x, voxel_y, voxel_z, img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)

        mid_plane_xy = self.semantic_encoder(plane_xy)
        mid_plane_yz = self.semantic_encoder(plane_yz)
        mid_plane_xz = self.semantic_encoder(plane_xz)

        # for i in range(len(mid_plane_xy)):
        #     print("mid:", mid_plane_xy[i].shape)
        semantic_plane_xy = self.semantic_neck(mid_plane_xy)
        semantic_plane_yz = self.semantic_neck(mid_plane_yz)
        semantic_plane_xz = self.semantic_neck(mid_plane_xz)
      
        semantic_voxel = self.aggregator(semantic_plane_xy, semantic_plane_yz, semantic_plane_xz)
        # for i in range(len(semantic_voxel)):
        #     print("semantic_voxel:", semantic_voxel[i].shape)
        # training losses
        losses = dict()
        
        if self.record_time:        
            torch.cuda.synchronize()
            t0 = time.time()
        
        if not self.disable_loss_depth and depth is not None:
            losses['loss_depth'] = self.img_view_transformer.get_depth_loss(img_inputs[-7], depth)
        
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
            rgbs = []
            depths = []
            gt_imgs = []
            gt_depths = []
            density_plane_xy = self.density_encoder(mid_plane_xy)
            density_plane_yz = self.density_encoder(mid_plane_yz)
            density_plane_xz = self.density_encoder(mid_plane_xz)
            color_plane_xy = self.color_encoder(mid_plane_xy)
            color_plane_yz = self.color_encoder(mid_plane_yz)
            color_plane_xz = self.color_encoder(mid_plane_xz)
            # print("density_plane_xy:", density_plane_xy.shape, "density_plane_xy:", density_plane_yz.shape, "density_plane_xy:", density_plane_xz.shape)

            for b in range(img_inputs[0].shape[0]):
                cam_intrin = img_inputs[-3][b]
                c2w = img_inputs[-2][b]
                directions = get_ray_direction_with_intrinsics(img_inputs[0].shape[-2], img_inputs[0].shape[-1], cam_intrin)
                rays_d, rays_o = get_rays(directions, c2w)
            # rays_d = torch.stack(rays_d)
            # rays_o = torch.stack(rays_o)
                rays_d = rays_d.reshape(-1, 3) #N, H, W, 3
                rays_o = rays_o.reshape(-1, 3)

                # print("rays_o:", rays_o.shape, "rays_d:", rays_d.shape)
                rand_indices = np.random.choice(range(rays_o.shape[0]), self.N_rand)
                rays_o, rays_d = rays_o[rand_indices], rays_d[rand_indices]
                gt_img = img_inputs[-5][b].reshape(-1,3)[rand_indices]
                gt_depth = img_inputs[-7][b].reshape(-1)[rand_indices]
                gt_imgs.append(gt_img)
                gt_depths.append(gt_depth)

                pts, z_vals = sample_along_camera_ray(ray_o=rays_o,   
                                                    ray_d=rays_d,
                                                    depth_range=self.near_far_range,
                                                    N_samples=self.N_samples,
                                                    inv_uniform=False,
                                                    det=False)


                aabb = img_inputs[-4][b] # batch size
                
            #     density_preds = self.density_head(density_voxel[0]) # [H_o, W_o, L_o, 1]
            #     density_preds = F.relu(density_preds)
            # # print("density_preds", density_preds.shape)
                # print("cor", rays_o[10], rays_d[10], pts[10])
                # fig = plt.figure()
                # ax = fig.add_subplot()
                # pts_ = pts[...,[1,2]].cpu().numpy()
                # rect = pch.Rectangle(xy=(aabb[0,0], aabb[0,1]), width=aabb[1,0]-aabb[0,0], height=aabb[1,1]-aabb[0,1], fill=False, color='y')
                # rect = pch.Rectangle(xy=(aabb[0,1], aabb[0,2]), width=aabb[1,1]-aabb[0,1], height=aabb[1,2]-aabb[0,2], fill=False, color='y')
                # ax.add_patch(rect)
                # for i in range(20):
                #     j=np.random.randint(pts.shape[0])
                # # print(pts_[0,:,0], pts_[0,:,1], pts_[0,:,2])
                #     ax.scatter(pts_[j,:,0], pts_[j,:,1], color='b')
                # # ax.scatter(x, y, z, color='r')
                # plt.title('sample pts')
                # plt.draw()
                # plt.savefig('./pts.png')
                # plt.show()


                pts = pts.reshape(1, pts.shape[0],pts.shape[1],3)
                # print("pts:", pts.shape)

                aabbSize = aabb[1] - aabb[0]
                invgridSize = 1.0/aabbSize * 2
                norm_pts = (pts-aabb[0]) * invgridSize - 1
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # norm_pts_ = norm_pts.clone().cpu().numpy()
                # ax.scatter(norm_pts_[:,0], norm_pts_[:,1], norm_pts_[:,2])
                # plt.savefig('./norm.png')
                # print("norm_0",norm_pts[:, 0].min(), norm_pts[:, 0].max())
                # print("norm_1",norm_pts[:, 1].min(), norm_pts[:, 1].max())
                # print("norm_2",norm_pts[:, 2].min(), norm_pts[:, 2].max())
                density_feature_xy = F.grid_sample(density_plane_xy, norm_pts[...,[0,1]], mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0).permute(1,2,0)
                density_feature_yz = F.grid_sample(density_plane_yz, norm_pts[...,[1,2]], mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0).permute(1,2,0)
                density_feature_xz = F.grid_sample(density_plane_xz, norm_pts[...,[0,2]], mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0).permute(1,2,0)
                color_feature_xy = F.grid_sample(color_plane_xy, norm_pts[...,[0,1]], mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0).permute(1,2,0)
                color_feature_yz = F.grid_sample(color_plane_yz, norm_pts[...,[1,2]], mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0).permute(1,2,0)
                color_feature_xz = F.grid_sample(color_plane_xz, norm_pts[...,[0,2]], mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0).permute(1,2,0)
                
                density_feature = torch.mean(torch.stack([density_feature_xy, density_feature_yz, density_feature_xz]), dim=0)
                color_feature = torch.mean(torch.stack([color_feature_xy, color_feature_yz, color_feature_xz]), dim=0)

                density = F.relu(self.density_head(density_feature))
                color = torch.sigmoid(self.color_head(color_feature))
                
                weights = self.get_weights(density, z_vals)

                color_2d = torch.sum(weights.unsqueeze(2) * color, dim=1)

                if self.white_bkgd:
                    color_2d = color_2d + (1. - torch.sum(weights, dim=-1, keepdim=True))

                depth_2d = torch.sum(weights * z_vals, dim=-1) / (torch.sum(weights, dim=-1) + 1e-8)
                depth_2d = torch.clamp(depth_2d, z_vals.min(), z_vals.max())
            # print("depth_2d:", depth_2d.shape)
                rgbs.append(color_2d)
                depths.append(depth_2d)
            rgbs = torch.stack(rgbs)
            depths = torch.stack(depths)
            gt_imgs = torch.stack(gt_imgs)
            gt_depths = torch.stack(gt_depths)
    
            losses["loss_color"] = F.mse_loss(rgbs, gt_imgs)
            fg_mask = torch.max(gt_depths, dim=1).values > 0.0
            gt_depths = gt_depths[fg_mask]
            depths = depths[fg_mask]
            losses["loss_render_depth"] = F.smooth_l1_loss(depths, gt_depths, reduction='mean')

            # print(losses["loss_color"], losses["loss_render_depth"] )

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
            gt_occ=None, visible_mask=None):
        
        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(points, img=img, img_metas=img_metas)

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
            rays_d = []
            rays_o = []
            for b in range(img[0].shape[0]):
                cam_intrin = img[-3][b]
                c2w = img[-2][b]
                directions = get_ray_direction_with_intrinsics(img[0].shape[-2], img[0].shape[-1], cam_intrin)
                rays_d_, rays_o_ = get_rays(directions, c2w)
                rays_d.append(rays_d_)
                rays_o.append(rays_o_)
            rays_d = torch.stack(rays_d)
            rays_o = torch.stack(rays_o)
            rays_d = rays_d.reshape(-1, 3) # N, H, W, 3
            rays_o = rays_o.reshape(-1, 3)
            H = img[0][0].shape[-2]
            W = img[0][0].shape[-1]
    
            num_rays = rays_o.shape[0]
            batch_size = 1024 * 8
            rgbs = []
            depths = []
            for i in range(0, num_rays, batch_size):
                rays_o_chunck = rays_o[i: i+batch_size]
                rays_d_chunck = rays_d[i: i+batch_size]
                pts, z_vals = sample_along_camera_ray(ray_o=rays_o_chunck,   
                                                  ray_d=rays_d_chunck,
                                                  depth_range=self.near_far_range,
                                                  N_samples=self.N_samples,
                                                  inv_uniform=False,
                                                  det=False)

                density_voxel = self.density_encoder(mid_voxel)
                color_voxel = self.color_encoder(mid_voxel)
                aabb = img[-4][0] # batch size
                pts = pts.reshape(1, pts.shape[0],pts.shape[1],1,3)
                aabbSize = aabb[1] - aabb[0]
                invgridSize = 1.0/aabbSize * 2
                norm_pts = (pts-aabb[0]) * invgridSize - 1
          
                density_feature = F.grid_sample(density_voxel[0].permute(0,1,4,3,2), norm_pts, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0).squeeze(-1).permute(1,2,0)
                color_feature = F.grid_sample(color_voxel[0].permute(0,1,4,3,2), norm_pts, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0).squeeze(-1).permute(1,2,0)
                # print("density_feature:", density_feature.shape)

                density = F.relu(self.density_head(density_feature))
                color = torch.sigmoid(self.color_head(color_feature))
                
                weights = self.get_weights(density, z_vals)
                color_2d = torch.sum(weights.unsqueeze(2) * color, dim=1)
                depth_2d = torch.sum(weights * z_vals, dim=-1) / (torch.sum(weights, dim=-1) + 1e-8)
                depth_2d = torch.clamp(depth_2d, z_vals.min(), z_vals.max())
                rgbs.append(color_2d)
                depths.append(depth_2d)
            rgbs = torch.cat(rgbs, dim=0).view(self.nerf_sample_view,H,W,3)
            depths = torch.cat(depths, dim=0).view(self.nerf_sample_view,H,W,1)
            psnr_total = 0
            for v in range(rgbs.shape[0]):
                # print("pred:", rgbs[v].var(), "gt:", img[-5][0][v].var())
                depth_ = ((depths[v]-depths[v].min()) / (depths[v].max() - depths[v].min()+1e-8)).repeat(1, 1, 3)
                img_to_save = torch.cat([rgbs[v], img[-5][0][v].permute(1,2,0), depth_], dim=1).clip(0, 1)
                img_to_save = np.uint8(img_to_save.cpu().numpy()* 255.0)
                psnr = compute_psnr(rgbs[v], img[-5][0][v].permute(1,2,0), mask=None)
                psnr_total += psnr
                cv2.imwrite("./img_"+str(v)+'.png', img_to_save)
            # print("psnr:", psnr_total/rgbs.shape[0])

        test_output = {
            'SC_metric': SC_metric,
            'SSC_metric': SSC_metric,
            'pred_c': pred_c,
            'pred_f': pred_f,
            'output_voxels': pred_c,
            'target_voxels': gt_occ,
            'evaluation_semantic': output['evaluation_semantic'],
        }
        
        if output['output_points'] is not None and points_occ is not None:
            test_output['output_points'] = output['output_points']
            test_output['target_points'] = output['target_points']
            
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
        