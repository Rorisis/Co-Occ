import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from projects.mmdet3d_plugin.utils.formating import cm_to_ious, format_SC_results, format_SSC_results

import pdb

@DATASETS.register_module()
class CustomNuScenesOccLSSDataset(NuScenesDataset):
    def __init__(self, occ_size, pc_range, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.occ_size = occ_size
        self.pc_range = pc_range
        
    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            
            return data
    
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        
        return example

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            # fix for running the code on different data_paths
            pts_filename=info['lidar_path'].replace('./data/nuscenes', self.data_root),
            sweeps=info['sweeps'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'],
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            lidar_token=info['lidar_token'],
        )
        
        # available for infos which are organized in scenes and prepared for video demos
        if 'scene_name' in info:
            input_dict['scene_name'] = info['scene_name']
        
        # not available for test-test
        if 'lidarseg' in info:
            input_dict['lidarseg'] = info['lidarseg']
        
        # fix data_path
        img_filenames = {}
        lidar2cam_dic = {}
        
        for cam_type, cam_info in info['cams'].items():
            cam_info['data_path'] = cam_info['data_path'].replace('./data/nuscenes', self.data_root)
            img_filenames[cam_type] = cam_info['data_path']
            
            # obtain lidar to camera transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            lidar2cam_dic[cam_type] = lidar2cam_rt.T
        
        input_dict['curr'] = info
        input_dict['img_filenames'] = img_filenames
        input_dict['lidar2cam_dic'] = lidar2cam_dic
        
        return input_dict

    def evaluate_lidarseg(self, results, logger=None, **kwargs):
        from projects.mmdet3d_plugin.utils import cm_to_ious, format_results
        eval_results = {}
        
        ''' evaluate lidar semantic segmentation '''
        ious = cm_to_ious(results['evaluation_semantic'])
        res_table, res_dic = format_results(ious, return_dic=True)
        for key, val in res_dic.items():
            eval_results['nuScenes_lidarseg_{}'.format(key)] = val
        
        if logger is not None:
            logger.info('LiDAR Segmentation Evaluation')
            logger.info(res_table)
        
        return eval_results

    def evaluate_ssc(self, results, logger=None, **kwargs):
        # though supported, it can only be evaluated by the sparse-LiDAR-generated occupancy in nusc
        
        eval_results = {}
        
        ''' evaluate SC '''
        evaluation_semantic = sum(results['SC_metric'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
        for key, val in res_dic.items():
            eval_results['SC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SC Evaluation')
            logger.info(res_table)
        
        ''' evaluate SSC '''
        evaluation_semantic = sum(results['SSC_metric'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SSC_results(ious, return_dic=True)
        for key, val in res_dic.items():
            eval_results['SSC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SSC Evaluation')
            logger.info(res_table)
        
        ''' evaluate SSC_fine '''
        if 'SSC_metric_fine' in results.keys():
            evaluation_semantic = sum(results['SSC_metric_fine'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SSC_results(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC fine Evaluation')
                logger.info(res_table)

        return eval_results

    def evaluate(self, results, logger=None, **kwargs):
        if results is None:
            logger.info('Skip Evaluation')
        eval_results = {}
        if 'evaluation_semantic' in results:
            eval_results.update(self.evaluate_lidarseg(results, logger, **kwargs))
            eval_results.update(self.evaluate_ssc(results, logger, **kwargs))
            return eval_results
        else:
            return self.evaluate_ssc(results, logger, **kwargs)
        