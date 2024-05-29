#import open3d as o3d
import numpy as np
import torch
import os
from mmdet.datasets.builder import PIPELINES
import pdb
from pyquaternion import Quaternion
from PIL import Image

@PIPELINES.register_module()
class CreateDepthFromLiDAR(object):
    def __init__(self, data_root=None, dataset='kitti', data_config=None, is_train=False):
        self.data_root = data_root
        self.dataset = dataset
        self.data_config = data_config
        self.is_train = is_train
        assert self.dataset in ['kitti', 'nusc']
        
    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        # from lidar to camera
        points = points.view(-1, 1, 3)
        points = points - trans.view(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        
        # the intrinsic matrix is [4, 4] for kitti and [3, 3] for nuscenes 
        if intrins.shape[-1] == 4:
            points = torch.cat((points, torch.ones((points.shape[0], trans.shape[0], 1, 1))), dim=2)
            points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        else:
            points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd

    def __call__(self, results):
        # loading LiDAR points
        if self.dataset == 'kitti':
            img_filename = results['img_filename'][0]
            seq_id, _, filename = img_filename.split("/")[-3:]
            # lidar_filename = os.path.join(self.data_root, 'dataset/sequences', 
            #                 seq_id, "velodyne", filename.replace(".png", ".bin"))
            lidar_points = np.fromfile(results['pts_filename'], dtype=np.float32).reshape(-1, 4)
            lidar_points = torch.from_numpy(lidar_points[:, :3]).float()
        else:
            lidar_points = np.fromfile(results['pts_filename'], dtype=np.float32, count=-1).reshape(-1, 5)[..., :3]
            lidar_points = torch.from_numpy(lidar_points).float()
        
        # project LiDAR to monocular / multi-view images
        if 'img_inputs' in results:
            imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][:6]
            # [num_point, num_img, 3] in format [u, v, d]
            projected_points = self.project_points(lidar_points, rots, trans, intrins, post_rots, post_trans)
            
            # create depth map
            img_h, img_w = imgs[0].shape[-2:]
            valid_mask = (projected_points[..., 0] >= 0) & \
                        (projected_points[..., 1] >= 0) & \
                        (projected_points[..., 0] <= img_w - 1) & \
                        (projected_points[..., 1] <= img_h - 1) & \
                        (projected_points[..., 2] > 0)
            
            gt_depths = []
            for img_index in range(imgs.shape[0]):
                gt_depth = torch.zeros((img_h, img_w))
                projected_points_i = projected_points[:, img_index]
                valid_mask_i = valid_mask[:, img_index]
                valid_points_i = projected_points_i[valid_mask_i]
                # sort
                depth_order = torch.argsort(valid_points_i[:, 2], descending=True)
                valid_points_i = valid_points_i[depth_order]
                # fill in
                gt_depth[valid_points_i[:, 1].round().long(), 
                        valid_points_i[:, 0].round().long()] = valid_points_i[:, 2]
                
                gt_depths.append(gt_depth)
            
            gt_depths = torch.stack(gt_depths)
            
            imgs, rots, trans, intrins, post_rots, post_trans, _, sensor2sensors, denorm_imgs, intrin_nerf, c2ws, img_size = results['img_inputs']
            results['img_inputs'] = imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors, denorm_imgs, intrin_nerf, c2ws, img_size

        else:
            imgs = []
            rots = []
            trans = []
            intrins = []
            post_rots = []
            post_trans = []
            gt_depths = []
            sensor2sensors = []
            cam_names = self.choose_cams()
            for cam_name in cam_names:
                img = np.zeros(self.data_config['input_size'])
                img = Image.fromarray(img)
                sensor2lidar = torch.tensor(results['lidar2cam_dic'][cam_name]).inverse().float()
                cam_data = results['curr']['cams'][cam_name]
                intrin = torch.Tensor(cam_data['cam_intrinsic'])

                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                rot = sensor2lidar[:3, :3]
                tran = sensor2lidar[:3, 3]
        
                scale, flip = None, None
                # image view augmentation (resize, crop, horizontal flip, rotate)
                img_augs = self.sample_augmentation(H=img.height,
                                                    W=img.width,
                                                    flip=flip,
                                                    scale=scale)
                
                resize, resize_dims, crop, flip, rotate = img_augs
                img, post_rot2, post_tran2 = \
                    self.img_transform(img, post_rot,
                                    post_tran,
                                    resize=resize,
                                    resize_dims=resize_dims,
                                    crop=crop,
                                    flip=flip,
                                    rotate=rotate)
                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2
                imgs.append(torch.tensor(np.array(img)).float())
                intrins.append(intrin)
                rots.append(rot)
                trans.append(tran)
                post_rots.append(post_rot)
                post_trans.append(post_tran)
                gt_depths.append(torch.zeros(1))
                # only placeholder currently, to be used for video-based methods
                sensor2sensors.append(sensor2lidar)
            imgs = torch.stack(imgs)
            rots = torch.stack(rots)
            trans = torch.stack(trans)
            intrins = torch.stack(intrins)
            post_rots = torch.stack(post_rots)
            post_trans = torch.stack(post_trans)
            gt_depths = torch.stack(gt_depths)
            sensor2sensors = torch.stack(sensor2sensors)

            projected_points = self.project_points(lidar_points, rots, trans, intrins, post_rots, post_trans)
            
            # create depth map
            img_h, img_w = imgs[0].shape[-2:]
            valid_mask = (projected_points[..., 0] >= 0) & \
                        (projected_points[..., 1] >= 0) & \
                        (projected_points[..., 0] <= img_w - 1) & \
                        (projected_points[..., 1] <= img_h - 1) & \
                        (projected_points[..., 2] > 0)
            
            gt_depths = []
            for img_index in range(imgs.shape[0]):
                gt_depth = torch.zeros((img_h, img_w))
                projected_points_i = projected_points[:, img_index]
                valid_mask_i = valid_mask[:, img_index]
                valid_points_i = projected_points_i[valid_mask_i]
                # sort
                depth_order = torch.argsort(valid_points_i[:, 2], descending=True)
                valid_points_i = valid_points_i[depth_order]
                # fill in
                gt_depth[valid_points_i[:, 1].round().long(), 
                        valid_points_i[:, 0].round().long()] = valid_points_i[:, 2]
                
                gt_depths.append(gt_depth)
            
            gt_depths = torch.stack(gt_depths)

            results['gt_depths'] = imgs, rots, trans, intrins, post_rots, post_trans, sensor2sensors, gt_depths, imgs.shape[-2:]

        
        # visualize image with overlayed depth
        # self.visualize(results['canvas'], gt_depths)
        
        return results
        
    def visualize(self, imgs, img_depths):
        out_path = 'debugs/lidar2depth'
        os.makedirs(out_path, exist_ok=True)
        
        import matplotlib.pyplot as plt
        
        # convert depth-map to depth-points
        for img_index in range(imgs.shape[0]):
            img_i = imgs[img_index][..., [2, 1, 0]]
            depth_i = img_depths[img_index]
            depth_points = torch.nonzero(depth_i)
            depth_points = torch.stack((depth_points[:, 1], depth_points[:, 0], depth_i[depth_points[:, 0], depth_points[:, 1]]), dim=1)
            
            plt.figure(dpi=300)
            plt.imshow(img_i)
            plt.scatter(depth_points[:, 0], depth_points[:, 1], s=1, c=depth_points[:, 2], alpha=0.2)
            plt.axis('off')
            plt.title('Image Depth')
            
            plt.savefig(os.path.join(out_path, 'demo_depth_{}.png'.format(img_index)))
            plt.close()
        
        pdb.set_trace()

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(self.data_config['cams']):
            cam_names = np.random.choice(self.data_config['cams'], self.data_config['Ncams'],
                                    replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW)/float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate