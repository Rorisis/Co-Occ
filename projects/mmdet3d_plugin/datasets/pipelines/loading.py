#import open3d as o3d
import trimesh
import mmcv
import numpy as np
import numba as nb

from mmdet.datasets.builder import PIPELINES
import yaml, os
import torch
from scipy import stats
from scipy.ndimage import zoom
from skimage import transform
import pdb
import torch.nn.functional as F
import copy
from pyquaternion import Quaternion

@PIPELINES.register_module()
class LoadOccupancy(object):

    def __init__(self, to_float32=True, use_semantic=True, occ_path=None, grid_size=[512, 512, 40], unoccupied=0,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], gt_resize_ratio=1, cal_visible=False, use_vel=False, data_root='data/nuscenes',
            is_train=False, is_test_submit=False, bda_aug_conf=None, cls_metas='nuscenes.yaml'):
        self.to_float32 = to_float32
        self.use_semantic = use_semantic
        self.occ_path = occ_path
        self.cal_visible = cal_visible
        
        self.is_train = is_train
        self.is_test_submit = is_test_submit
        
        self.cls_metas = cls_metas
        with open(cls_metas, 'r') as stream:
            nusc_cls_metas = yaml.safe_load(stream)
            self.learning_map = nusc_cls_metas['learning_map']

        self.data_root = data_root
        self.bda_aug_conf = bda_aug_conf

        self.grid_size = np.array(grid_size)
        self.unoccupied = unoccupied
        self.pc_range = np.array(pc_range)
        self.voxel_size = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        self.gt_resize_ratio = gt_resize_ratio
        self.use_vel = use_vel
    
    def sample_3d_augmentation(self):
        """Generate 3d augmentation values based on bda_config."""
        
        # Currently, we only use the flips along three directions. The rotation and scaling are not fully experimented.
        rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
        scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
        flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
        flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        flip_dz = np.random.uniform() < self.bda_aug_conf.get('flip_dz_ratio', 0.0)
        
        return rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz

    def __call__(self, results):
        if self.is_test_submit:
            imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors = results['img_inputs']
            bda_rot = torch.eye(3).float()
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors)
            
            pts_filename = results['pts_filename']
            points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(-1, 5)[..., :3]
            points_label = np.zeros((points.shape[0], 1)) # placeholder
            lidarseg = np.concatenate([points, points_label], axis=-1)
            results['points_occ'] = torch.from_numpy(lidarseg).float()
            
            return results
        
        ''' load lidarseg points '''
        # print(results.keys())
        # print(results['lidarseg'], results['occ_path'])
        lidarseg_labels_filename = os.path.join(self.data_root, results['lidarseg'])
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        pts_filename = results['pts_filename']
        
        points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(-1, 5)[..., :3]
        lidarseg = np.concatenate([points, points_label], axis=-1)
        
        pointsT = points.copy().T

        pointsT = np.array(Quaternion(results['lidar2ego_rotation']).rotation_matrix) @ pointsT
        pointsT = pointsT + np.array(results['lidar2ego_translation'])[:, np.newaxis]

        pointsT = np.array(Quaternion(results['ego2global_rotation']).rotation_matrix) @ pointsT
        pointsT = pointsT + np.array(results['ego2global_translation'])[:, np.newaxis]
        pointsT = pointsT.T

        aabb_min = torch.Tensor([pointsT[:, 0].min(), pointsT[:, 1].min(), pointsT[:, 2].min()])
        aabb_max = torch.Tensor([pointsT[:, 0].max(), pointsT[:, 1].max(), pointsT[:, 2].max()])
        aabb = torch.stack([aabb_min, aabb_max])

        ''' create multi-view projections '''
        ''' apply 3D augmentation for lidar_points (and the later generated occupancy) '''
        if self.is_train:
            rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz = self.sample_3d_augmentation()
            _, bda_rot = voxel_transform(None, rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz)
        else:
            bda_rot = torch.eye(3).float()
        
        # transform points
        points = points @ bda_rot.t().numpy()
        lidarseg[:, :3] = points
        
        # print(results['pts_filename'].split('/')[-1])
        rel_path = 'samples/{0}.npy'.format(results['pts_filename'].split('/')[-1])
        #  [z y x cls] or [z y x vx vy vz cls]
        occ = np.load(os.path.join(self.occ_path, rel_path))
        occ = occ.astype(np.float32)
        # occ = occ[..., -1:]
        
        # class 0 is 'ignore' class
        if self.use_semantic:
            occ[..., 3][occ[..., 3] == 0] = 255
        else:
            occ = occ[occ[..., 3] > 0]
            occ[..., 3] = 1

        voxel = np.zeros(self.grid_size)
        voxel[occ[:, 0].astype(np.int), occ[:, 1].astype(np.int), occ[:, 2].astype(np.int)] = occ[:, 3]
        # print(voxel.shape)

        # pcd_np_cor =  self.voxel2world(occ[..., [2,1,0]] + 0.5)
        # pcd_label = occ[..., -1:]
        # pcd_np_cor = (bda_rot @ torch.from_numpy(pcd_np_cor).unsqueeze(-1).float()).squeeze(-1).numpy()
        # pcd_np_cor = self.world2voxel(pcd_np_cor)

        # # make sure the point is in the grid
        # pcd_np_cor = np.clip(pcd_np_cor, np.array([0,0,0]), self.grid_size - 1)
        # transformed_occ = copy.deepcopy(pcd_np_cor)
        # pcd_np = np.concatenate([pcd_np_cor, pcd_label], axis=-1)

        # # 255: noise, 1-16 normal classes, 0 unoccupied
        # pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
        # pcd_np = pcd_np.astype(np.int64)
        # processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.unoccupied
        # processed_label = nb_process_label(processed_label, pcd_np)
        results['gt_occ'] = voxel
        # print(results['gt_occ'].shape)

        results['points_occ'] = torch.from_numpy(lidarseg).float()

        imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors, denorm_imgs, intrin_nerf, c2ws, img_size = results['img_inputs']
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors, denorm_imgs, aabb, intrin_nerf, c2ws, img_size)
        results['gt_depths'] = rots, trans, intrins, post_rots, post_trans, bda_rot, img_size, gt_depths

        return results

    def voxel2world(self, voxel):
        """
        voxel: [N, 3]
        """
        return voxel * self.voxel_size[None, :] + self.pc_range[:3][None, :]


    def world2voxel(self, wolrd):
        """
        wolrd: [N, 3]
        """
        return (wolrd - self.pc_range[:3][None, :]) / self.voxel_size[None, :]


    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}'
        return repr_str

    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        
        # from lidar to camera
        points = points.reshape(-1, 1, 3)
        points = points - trans.reshape(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        
        # from camera to raw pixel
        points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd

@PIPELINES.register_module()
class LoadOccupancy2(object):

    def __init__(self, to_float32=True, use_semantic=False, occ_path=None, grid_size=[512, 512, 40], unoccupied=0,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], gt_resize_ratio=1, cal_visible=False, use_vel=False, data_root='data/nuscenes',
            is_train=False, is_test_submit=False, bda_aug_conf=None, cls_metas='nuscenes.yaml'):
        self.to_float32 = to_float32
        self.use_semantic = use_semantic
        self.occ_path = occ_path
        self.cal_visible = cal_visible
        
        self.is_train = is_train
        self.is_test_submit = is_test_submit
        
        self.cls_metas = cls_metas
        with open(cls_metas, 'r') as stream:
            nusc_cls_metas = yaml.safe_load(stream)
            self.learning_map = nusc_cls_metas['learning_map']

        self.data_root = data_root
        self.bda_aug_conf = bda_aug_conf

        self.grid_size = np.array(grid_size)
        self.unoccupied = unoccupied
        self.pc_range = np.array(pc_range)
        self.voxel_size = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        self.gt_resize_ratio = gt_resize_ratio
        self.use_vel = use_vel
    
    def sample_3d_augmentation(self):
        """Generate 3d augmentation values based on bda_config."""
        
        # Currently, we only use the flips along three directions. The rotation and scaling are not fully experimented.
        rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
        scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
        flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
        flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        flip_dz = np.random.uniform() < self.bda_aug_conf.get('flip_dz_ratio', 0.0)
        
        return rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz

    def __call__(self, results):
        if self.is_test_submit:
            imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors = results['img_inputs']
            bda_rot = torch.eye(3).float()
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors)
            
            pts_filename = results['pts_filename']
            points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(-1, 5)[..., :3]
            points_label = np.zeros((points.shape[0], 1)) # placeholder
            lidarseg = np.concatenate([points, points_label], axis=-1)
            results['points_occ'] = torch.from_numpy(lidarseg).float()
            
            return results
        
        ''' load lidarseg points '''
        lidarseg_labels_filename = os.path.join(self.data_root, results['lidarseg'])
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        pts_filename = results['pts_filename']
        
        points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(-1, 5)[..., :3]
        lidarseg = np.concatenate([points, points_label], axis=-1)
        
        pointsT = points.copy().T

        pointsT = np.array(Quaternion(results['lidar2ego_rotation']).rotation_matrix) @ pointsT
        pointsT = pointsT + np.array(results['lidar2ego_translation'])[:, np.newaxis]

        pointsT = np.array(Quaternion(results['ego2global_rotation']).rotation_matrix) @ pointsT
        pointsT = pointsT + np.array(results['ego2global_translation'])[:, np.newaxis]
        pointsT = pointsT.T

        aabb_min = torch.Tensor([pointsT[:, 0].min(), pointsT[:, 1].min(), pointsT[:, 2].min()])
        aabb_max = torch.Tensor([pointsT[:, 0].max(), pointsT[:, 1].max(), pointsT[:, 2].max()])
        aabb = torch.stack([aabb_min, aabb_max])

        ''' create multi-view projections '''
        ''' apply 3D augmentation for lidar_points (and the later generated occupancy) '''
        if self.is_train:
            rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz = self.sample_3d_augmentation()
            _, bda_rot = voxel_transform(None, rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz)
        else:
            bda_rot = torch.eye(3).float()
        
        # transform points
        points = points @ bda_rot.t().numpy()
        lidarseg[:, :3] = points

        rel_path = 'scene_{0}/occupancy/{1}.npy'.format(results['scene_token'], results['lidar_token'])
        #  [z y x cls] or [z y x vx vy vz cls]
        pcd = np.load(os.path.join(self.occ_path, rel_path))
        pcd_label = pcd[..., -1:]
        pcd_label[pcd_label==0] = 255
        pcd_np_cor = self.voxel2world(pcd[..., [2,1,0]] + 0.5)  # x y z
        untransformed_occ = copy.deepcopy(pcd_np_cor)  # N 4
        # print(untransformed_occ.shape)
        # bevdet augmentation
        pcd_np_cor = (bda_rot @ torch.from_numpy(pcd_np_cor).unsqueeze(-1).float()).squeeze(-1).numpy()
        pcd_np_cor = self.world2voxel(pcd_np_cor)

        # make sure the point is in the grid
        pcd_np_cor = np.clip(pcd_np_cor, np.array([0,0,0]), self.grid_size - 1)
        transformed_occ = copy.deepcopy(pcd_np_cor)
        pcd_np = np.concatenate([pcd_np_cor, pcd_label], axis=-1)

        # velocity
        if self.use_vel:
            pcd_vel = pcd[..., [3,4,5]]  # x y z
            pcd_vel = (bda_rot @ torch.from_numpy(pcd_vel).unsqueeze(-1).float()).squeeze(-1).numpy()
            pcd_vel = np.concatenate([pcd_np, pcd_vel], axis=-1)  # [x y z cls vx vy vz]
            results['gt_vel'] = pcd_vel

        # 255: noise, 1-16 normal classes, 0 unoccupied
        pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
        pcd_np = pcd_np.astype(np.int64)
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.unoccupied
        processed_label = nb_process_label(processed_label, pcd_np)
        results['gt_occ'] = processed_label

        if self.cal_visible:
            visible_mask = np.zeros(self.grid_size, dtype=np.uint8)
            # camera branch
            if 'img_inputs' in results.keys():
                _, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][:6]
                occ_uvds = self.project_points(torch.Tensor(untransformed_occ), 
                                                rots, trans, intrins, post_rots, post_trans)  # N 6 3
                N, n_cam, _ = occ_uvds.shape
                img_visible_mask = np.zeros((N, n_cam))
                img_h, img_w = results['img_inputs'][0].shape[-2:]
                for cam_idx in range(n_cam):
                    basic_mask = (occ_uvds[:, cam_idx, 0] >= 0) & (occ_uvds[:, cam_idx, 0] < img_w) & \
                                (occ_uvds[:, cam_idx, 1] >= 0) & (occ_uvds[:, cam_idx, 1] < img_h) & \
                                (occ_uvds[:, cam_idx, 2] >= 0)

                    basic_valid_occ = occ_uvds[basic_mask, cam_idx]  # M 3
                    M = basic_valid_occ.shape[0]  # TODO M~=?
                    basic_valid_occ[:, 2] = basic_valid_occ[:, 2] * 10
                    basic_valid_occ = basic_valid_occ.cpu().numpy()
                    basic_valid_occ = basic_valid_occ.astype(np.int16)  # TODO first round then int?
                    depth_canva = np.ones((img_h, img_w), dtype=np.uint16) * 2048
                    nb_valid_mask = np.zeros((M), dtype=np.bool)
                    nb_valid_mask = nb_process_img_points(basic_valid_occ, depth_canva, nb_valid_mask)  # M
                    img_visible_mask[basic_mask, cam_idx] = nb_valid_mask

                img_visible_mask = img_visible_mask.sum(1) > 0  # N  1:occupied  0: free
                img_visible_mask = img_visible_mask.reshape(-1, 1).astype(pcd_label.dtype) 

                img_pcd_np = np.concatenate([transformed_occ, img_visible_mask], axis=-1)
                img_pcd_np = img_pcd_np[np.lexsort((transformed_occ[:, 0], transformed_occ[:, 1], transformed_occ[:, 2])), :]
                img_pcd_np = img_pcd_np.astype(np.int64)
                img_occ_label = np.zeros(self.grid_size, dtype=np.uint8)
                voxel_img = nb_process_label(img_occ_label, img_pcd_np) 
                visible_mask = visible_mask | voxel_img
                results['img_visible_mask'] = voxel_img


            # lidar branch
            if 'points' in results.keys():
                pts = results['points'].tensor.cpu().numpy()[:, :3]
                pts_in_range = ((pts>=self.pc_range[:3]) & (pts<self.pc_range[3:])).sum(1)==3
                pts = pts[pts_in_range]
                pts = (pts - self.pc_range[:3])/self.voxel_size
                pts = np.concatenate([pts, np.ones((pts.shape[0], 1)).astype(pts.dtype)], axis=1) 
                pts = pts[np.lexsort((pts[:, 0], pts[:, 1], pts[:, 2])), :].astype(np.int64)
                pts_occ_label = np.zeros(self.grid_size, dtype=np.uint8)
                voxel_pts = nb_process_label(pts_occ_label, pts)  # W H D 1:occupied 0:free
                visible_mask = visible_mask | voxel_pts
                results['lidar_visible_mask'] = voxel_pts

            results['visible_mask'] = visible_mask
        results['points_occ'] = torch.from_numpy(lidarseg).float()

        imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors, denorm_imgs, intrin_nerf, c2ws, img_size = results['img_inputs']
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors, denorm_imgs, aabb, intrin_nerf, c2ws, img_size)
        results['gt_depths'] = rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths

        return results

    def voxel2world(self, voxel):
        """
        voxel: [N, 3]
        """
        return voxel * self.voxel_size[None, :] + self.pc_range[:3][None, :]


    def world2voxel(self, wolrd):
        """
        wolrd: [N, 3]
        """
        return (wolrd - self.pc_range[:3][None, :]) / self.voxel_size[None, :]


    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}'
        return repr_str

    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        
        # from lidar to camera
        points = points.reshape(-1, 1, 3)
        points = points - trans.reshape(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        
        # from camera to raw pixel
        points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd
    
# b1:boolean, u1: uint8, i2: int16, u2: uint16
@nb.jit('b1[:](i2[:,:],u2[:,:],b1[:])', nopython=True, cache=True, parallel=False)
def nb_process_img_points(basic_valid_occ, depth_canva, nb_valid_mask):
    # basic_valid_occ M 3
    # depth_canva H W
    # label_size = M   # for original occ, small: 2w mid: ~8w base: ~30w
    canva_idx = -1 * np.ones_like(depth_canva, dtype=np.int16)
    for i in range(basic_valid_occ.shape[0]):
        occ = basic_valid_occ[i]
        if occ[2] < depth_canva[occ[1], occ[0]]:
            if canva_idx[occ[1], occ[0]] != -1:
                nb_valid_mask[canva_idx[occ[1], occ[0]]] = False

            canva_idx[occ[1], occ[0]] = i
            depth_canva[occ[1], occ[0]] = occ[2]
            nb_valid_mask[i] = True
    return nb_valid_mask

# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label_withvel(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label


# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label

def voxel_transform(voxel_labels, rotate_angle, scale_ratio, flip_dx, flip_dy, flip_dz):
    # bird-eye-view rotation
    rotate_degree = rotate_angle
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([
        [rot_cos, -rot_sin, 0, 0],
        [rot_sin, rot_cos, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    
    # I @ flip_x @ flip_y
    flip_mat = torch.eye(4)
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([
            [-1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([
            [1, 0, 0, 0], 
            [0, -1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    
    if flip_dz:
        flip_mat = flip_mat @ torch.Tensor([
            [1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, -1, 0],
            [0, 0, 0, 1]])
    
    # denorm @ flip_x @ flip_y @ flip_z @ rotation @ normalize
    bda_mat = flip_mat @ rot_mat
    bda_mat = bda_mat[:3, :3]
    
    # apply transformation to the 3D volume, which is tensor of shape [X, Y, Z]
    if voxel_labels is not None:
        voxel_labels = voxel_labels.numpy().astype(np.uint8)
        if not np.isclose(rotate_degree, 0):
            '''
            Currently, we use a naive method for 3D rotation because we found the visualization of 
            rotate results with scipy is strange: 
                scipy.ndimage.interpolation.rotate(voxel_labels, rotate_degree, 
                        output=voxel_labels, mode='constant', order=0, 
                        cval=255, axes=(0, 1), reshape=False)
            However, we found further using BEV rotation brings no gains over 3D flips only.
            '''
            voxel_labels = custom_rotate_3d(voxel_labels, rotate_degree)
        
        if flip_dz:
            voxel_labels = voxel_labels[:, :, ::-1]
        
        if flip_dy:
            voxel_labels = voxel_labels[:, ::-1]
        
        if flip_dx:
            voxel_labels = voxel_labels[::-1]
        
        voxel_labels = torch.from_numpy(voxel_labels.copy()).long()
    
    return voxel_labels, bda_mat

def custom_rotate_3d(voxel_labels, rotate_degree):
    # rotate like images: convert to PIL Image and rotate
    is_tensor = False
    if type(voxel_labels) is torch.Tensor:
        is_tensor = True
        voxel_labels = voxel_labels.numpy().astype(np.uint8)
    
    voxel_labels_list = []
    for height_index in range(voxel_labels.shape[-1]):
        bev_labels = voxel_labels[..., height_index]
        bev_labels = Image.fromarray(bev_labels.astype(np.uint8))
        bev_labels = bev_labels.rotate(rotate_degree, resample=Image.Resampling.NEAREST, fillcolor=255)
        bev_labels = np.array(bev_labels)
        voxel_labels_list.append(bev_labels)
    voxel_labels = np.stack(voxel_labels_list, axis=-1)
    
    if is_tensor:
        voxel_labels = torch.from_numpy(voxel_labels).long()
    
    return voxel_labels