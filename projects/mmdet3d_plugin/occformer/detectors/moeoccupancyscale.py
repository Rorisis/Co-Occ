import torch
import torch.nn.functional as F
import collections 

from mmdet.models import DETECTORS
from mmcv.runner import force_fp32
from mmdet3d.models import builder

from projects.mmdet3d_plugin.utils import fast_hist_crop
from .bevdepth import BEVDepth

import numpy as np
import time
import pdb
import copy

@DETECTORS.register_module()
class MoEOccpancyScale(BEVDepth):
    def __init__(self, 
                loss_cfg=None,
                disable_loss_depth=False,
                empty_idx=0,
                occ_fuser=None,
                occ_encoder_backbone=None,
                occ_encoder_neck=None,
                loss_norm=False,
                **kwargs):
        super().__init__(**kwargs)
        

        self.loss_cfg = loss_cfg
        self.disable_loss_depth = disable_loss_depth
        self.loss_norm = loss_norm
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
        self.empty_idx = empty_idx
        self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        self.occ_fuser = builder.build_fusion_layer(occ_fuser) if occ_fuser is not None else None
            
    
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

        if self.occ_fuser is not None:
            voxel_feats = self.occ_fuser(img_voxel_feats, pts_voxel_feats)
        else:
            assert (img_voxel_feats is None) or (pts_voxel_feats is None)
            voxel_feats = img_voxel_feats if pts_voxel_feats is None else pts_voxel_feats

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

        return (voxel_feats, img_feats, pts_feats, depth)
    
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
        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        
        # training losses
        losses = dict()
        
        if self.record_time:        
            torch.cuda.synchronize()
            t0 = time.time()
        
        if not self.disable_loss_depth and depth is not None:
            losses['loss_depth'] = self.img_view_transformer.get_depth_loss(img_inputs[-3], depth)
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['loss_depth'].append(t1 - t0)
        
        transform = img_inputs[1:] if img_inputs is not None else None
        losses_occupancy = self.forward_pts_train(voxel_feats, gt_occ, 
                        points_occ, img_metas, img_feats=img_feats, points_uv=points_uv, **kwargs)
        losses.update(losses_occupancy)
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

        transform = img[1:] if img is not None else None
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
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

        test_output = {
            'SC_metric': SC_metric,
            'SSC_metric': SSC_metric,
            'pred_c': pred_c,
            'pred_f': pred_f,
            'output_voxels': pred_c,
            'target_voxels': gt_occ,
            'evaluation_semantic': output['evaluation_semantic'],
        }

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
        