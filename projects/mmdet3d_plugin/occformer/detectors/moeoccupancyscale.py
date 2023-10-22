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
import mayavi.mlab as mlab

@DETECTORS.register_module()
class MoEOccupancyScale(BEVDepth):
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
        

        self.rendering_test = rendering_test
        self.use_rendering = use_rendering
        self.test_rendering = test_rendering

        if use_rendering:
            # self.density_encoder = builder.build_neck(density_encoder)
            # self.density_neck = builder.build_neck(density_neck)

                # self.color_encoder = builder.build_neck(color_encoder)
            self.color_head = MLP(input_dim=192, output_dim=3,net_depth=3,skip_layer=None)#torch.nn.Linear(256, 3)#MLP(input_dim=256, output_dim=3,net_depth=3,skip_layer=0)
            # self.color_neck = builder.build_neck(color_neck)
            self.density_head = torch.nn.Linear(192, 1)
            
        
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
                
        x = self.image_encoder(img[0])
        img_feats = x.clone()
        
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
    def extract_pts_feat(self, pts):
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        # print(len(pts), pts[0].shape)
        voxels, num_points, coors = self.voxelize(pts)
        # print(voxels.shape, len(num_points), coors.shape)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(pts[0][:,1].cpu().numpy(), pts[0][:,0].cpu().numpy(), pts[0][:,2].cpu().numpy(), color ='b', alpha=0.2)
        # ax.scatter(voxels[:,1].cpu().numpy(), voxels[:,0].cpu().numpy(), voxels[:,2].cpu().numpy(), color ='r', marker='o')
        # plt.savefig('./pts.png')
#         fig = mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(1600, 1000))
# #       category  label   r   g   b
#         mlab.points3d(voxels[:,0], voxels[:,1], voxels[:,2], color=(0,0,0), mode='point', colormap = 'gnuplot', scale_factor=1, figure=fig)
#         mlab.savefig('./pts.png', size=(1600, 1000), figure=fig)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        pts_enc_feats = self.pts_middle_encoder(voxel_features, coors, batch_size)
        if self.with_pts_backbone:
            pts_enc_feats['x'] = self.pts_backbone(pts_enc_feats['x'])
        if self.with_pts_neck:
            pts_enc_feats['x'] = self.pts_neck(pts_enc_feats['x'])

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

        if self.occ_fuser is not None:
            voxel_feats = self.occ_fuser(img_voxel_feats, pts_voxel_feats)
        else:
            assert (img_voxel_feats is None) or (pts_voxel_feats is None)
            voxel_feats = img_voxel_feats if pts_voxel_feats is None else pts_voxel_feats

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_fuser'].append(t1 - t0)

        voxel_feats = self.bev_encoder(voxel_feats)

        if type(voxel_feats) is not list:
            voxel_feats = [voxel_feats]

        # voxel_feats_enc = self.occ_encoder(voxel_feats)
        # if type(voxel_feats_enc) is not list:
        #     voxel_feats_enc = [voxel_feats_enc]

        # if self.record_time:
        #     torch.cuda.synchronize()
        #     t2 = time.time()
        #     self.time_stats['occ_encoder'].append(t2 - t1)

        return (voxel_feats, img_feats, pts_feats, depth, geom)
    
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
        voxel_feats, img_feats, pts_feats, depth, gemo = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
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
            # rays_d = []
            # rays_o = []
            # rgbs = []
            # depths = []
            # gt_imgs = []
            # gt_depths = []
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
            # print("norm_coord_frustum:", norm_coord_frustum.max(), norm_coord_frustum.min())

            density_features = []
            color_features = []

            for i in range(norm_coord_frustum.shape[0]):
                density_feature = F.grid_sample(voxel_feats[0].permute(0,1,4,3,2), norm_coord_frustum[i].unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False).permute(0,4,2,3,1)
 
                color_feature = F.grid_sample(voxel_feats[0].permute(0,1,4,3,2), norm_coord_frustum[i].unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False).permute(0,4,2,3,1) # b, h, w, d, c
                color_features .append(color_feature)
                density_features.append(density_feature)
                
            
            density_features = torch.cat(density_features, dim=0)

            color_features = torch.cat(color_features, dim=0)

            density = F.relu(self.density_head(density_features))

            rgb = torch.sigmoid(self.color_head(color_features))
            
            directions = []

            cam_intrins = img_inputs[3][0]

            
            for i in range(gemo.shape[0]):

                dir_coord_2d = dir_coord_2d * img_inputs[0].shape[-1] // gemo.shape[-2] #img: b, n, c, h, w gemo: b, d, h, w, c

                # print(cam_intrins[i].shape)
                if cam_intrins.shape[-1] == 4:
                    dir_coord_3d = unproject_image_to_rect(dir_coord_2d, torch.cat((cam_intrins[i][:3, :3], torch.zeros(3, 1).to(cam_intrins.device)), dim=1).float())
                else:
                    dir_coord_3d = unproject_image_to_rect(dir_coord_2d, torch.cat((cam_intrins[i], torch.zeros(3, 1).to(cam_intrins.device)), dim=1).float())
                direction = dir_coord_3d[:, :, 1, :] - dir_coord_3d[:, :, 0, :]
                direction /= img_inputs[0].shape[-1] // gemo.shape[-2]

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

            rgb_values = F.interpolate(rgb_values, scale_factor=self.scale)
            depth_values = F.interpolate(depth_values, scale_factor=self.scale).squeeze(1)
            # depth_values = self.upsample(depth_values)
            # print("color:", rgb_values.shape, "depth:", depth_values.shape)
            # print(img_inputs[0][0].shape, img_inputs[-7][0].shape)
            

            gt_depths = img_inputs[7][0]
            gt_imgs = img_inputs[0][0]
            losses["loss_color"] = F.mse_loss(rgb_values, gt_imgs)

            if self.depth_supervise:
                fg_mask = torch.max(gt_depths.reshape(gt_depths.shape[0], -1), dim=1).values > 0.0
                gt_depths = gt_depths[fg_mask]
                depth_values = depth_values[fg_mask]
                losses["loss_render_depth"] = F.smooth_l1_loss(depth_values, gt_depths, reduction='none').mean()
            # if pts_feats != None:
            rendered_opacity = F.interpolate(weights, scale_factor=self.scale).sum(dim=1)
            gt_opacity = (gt_depths != 0).to(gt_depths.dtype)
            losses["loss_opacity"] = torch.mean(-gt_opacity * torch.log(rendered_opacity + 1e-6) - (1 - gt_opacity) * torch.log(1 - rendered_opacity +1e-6)) # BCE lo



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
        
        voxel_feats, img_feats, pts_feats, depth, gemo = self.extract_feat(points, img=img, img_metas=img_metas)       
        output = self.pts_bbox_head.simple_test(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            points_uv=points_uv,
        )
        
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
        