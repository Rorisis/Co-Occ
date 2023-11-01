# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES

import torch
from PIL import Image
# from .loading_nusc_imgs import mmlabNormalize
from torchvision import transforms
from skimage import io
import cv2
import matplotlib.pyplot as plt

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_SemanticKitti(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, data_config, is_train=False, img_norm_cfg=None):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.img_norm_cfg = img_norm_cfg
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    
        self.to_tensor_normalized = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
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
            resize = float(fW) / float(W)
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

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        gt_depths = []
        sensor2sensors = []
        denorm_imgs = []
        intrin_nerf = []
        c2ws = []
        # load the monocular image for semantic kitti
        img_filenames = results['img_filename']
        # assert len(img_filenames) == 1
        
        for i in range(len(img_filenames)):
            img_filename = img_filenames[i]
            
            # img = io.imread(img_filename)
            # img = mmcv.imread(img_filename, 'unchanged')
            img = Image.open(img_filename).convert("RGB")
  
            results['raw_img'] = img
            
            # img = Image.fromarray(img)
            
            # perform image-view augmentation
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            
            img_augs = self.sample_augmentation(H=img.height, W=img.width, 
                            flip=flip, scale=scale)

            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot, post_tran, resize=resize, 
                    resize_dims=resize_dims, crop=crop,flip=flip, rotate=rotate)

            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # intrins
            intrin = torch.Tensor(results['cam_intrinsic'][0])
            intrin_nerf_ = torch.Tensor(results['cam_intrinsic'][0])
            intrin_nerf_[:2] = intrin_nerf_[:2] * resize
            intrin_nerf_[0,2] -= crop[0]
            intrin_nerf_[1,2] -= crop[1]
            
            # extrins
            lidar2cam = torch.Tensor(results['lidar2cam'][0])
            cam2lidar = lidar2cam.inverse()
            rot = cam2lidar[:3, :3]
            tran = cam2lidar[:3, 3]
            
            results['canvas'] = np.array(img)[None]
            
            # img = np.array(img) 
            # denorm_img = torch.tensor(np.array(img)).float().permute(2, 0, 1).contiguous()/ 255.0
            
            # img = self.normalize_img(img, img_norm_cfg=self.img_norm_cfg)
            # img = torch.tensor(img).float().permute(2, 0, 1).contiguous()/ 255.0
            # denorm_img = mmcv.imdenormalize(img, self.mean, self.std, to_bgr=True).astype(np.uint8) 
            # print(denorm_img.shape, denorm_img.mean(), img.mean())
            # cv2.imwrite('check.png', denorm_img)
            
            # img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
            # denorm_img = torch.tensor(np.array(denorm_img)).float().permute(2, 0, 1).contiguous() / 255.

            # print(denorm_img.shape, img.mean())
            img = np.array(img, dtype=np.float32, copy=False) / 255.0 
            plt.imsave('./check.png', img)
            denorm_img = self.to_tensor(img)
            img = self.to_tensor(img) 
            # print(denorm_img.mean(), img.mean(), denorm_img.shape)

            
            # print(img.shape, img.mean())
            depth = torch.zeros(1)

            imgs.append(img)
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            gt_depths.append(torch.zeros(1))
            denorm_imgs.append(denorm_img)
            intrin_nerf.append(intrin_nerf_)
            c2ws.append(torch.zeros(1))
            # only placeholder currently, to be used for video-based methods
            sensor2sensors.append(cam2lidar)
        
        imgs = torch.stack(imgs)
        rots = torch.stack(rots)
        trans = torch.stack(trans)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        gt_depths = torch.stack(gt_depths)
        denorm_imgs = torch.stack(denorm_imgs)
        intrin_nerf = torch.stack(intrin_nerf)
        c2ws = torch.stack(c2ws)
        sensor2sensors = torch.stack(sensor2sensors)
        
        # res = [imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors, imgs.shape[-2:]]
        # res = [x[None] for x in res]
        
        return imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors, denorm_imgs, intrin_nerf, c2ws, imgs.shape[-2:]


    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        
        return results

def mmlabNormalize(img, img_norm_cfg=None):
        from mmcv.image.photometric import imnormalize
        if img_norm_cfg is None:
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            to_rgb = True
        else:
            mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
            std = np.array(img_norm_cfg['std'], dtype=np.float32)
            to_rgb = img_norm_cfg['to_rgb']
        
        img = imnormalize(np.array(img), mean, std, to_rgb)
        # img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    
        return img